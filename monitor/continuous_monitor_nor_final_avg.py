# -*- coding: utf-8 -*-
import os
import json
import math
import struct
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tqdm import tqdm
from CBAM_attention import ChannelAttention, CBAMAttention

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# =========================
# demo风格：全局绘图设置
# =========================
def setup_pretty_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.sans-serif": ["Times New Roman"],
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,

        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.unicode_minus": False,
    })


setup_pretty_style()


# =========================
# GPU memory growth
# =========================
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass


# =========================
# 通用工具
# =========================
def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def linear_mapping(pre_a, pre_b, end_a, end_b, value):
    a = (end_b - end_a) / (pre_b - pre_a)
    b = end_a - a * pre_a
    return a * value + b


def idx_to_coord(idx: int, coord_min: float, coord_max: float, idx_min: int = 0, idx_max: int = 1023) -> float:
    return float(linear_mapping(idx_min, idx_max, coord_min, coord_max, float(idx)))


def xy_distance_mm(xy1: Tuple[float, float], xy2: Tuple[float, float]) -> float:
    dx = float(xy1[0]) - float(xy2[0])
    dy = float(xy1[1]) - float(xy2[1])
    return float(math.sqrt(dx * dx + dy * dy))


def normalize_by_global_absmax(wave_32xN: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    m = float(np.max(np.abs(wave_32xN)))
    if m > 0:
        return wave_32xN / (m + eps)
    return wave_32xN


def normalize_by_channel_absmax(wave_32xN: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    m = np.max(np.abs(wave_32xN), axis=1, keepdims=True).astype(np.float64)
    return wave_32xN / (m + eps)


# =========================
# SAC 流式读取器
# =========================
class SACChunkReader:
    HEADER_BYTES = 632

    def __init__(self, sac_path: str):
        self.sac_path = sac_path
        self.f = open(sac_path, "rb")

        header = self.f.read(self.HEADER_BYTES)
        if len(header) != self.HEADER_BYTES:
            raise IOError(f"Cannot read SAC header: {sac_path}")

        int_base = 70 * 4
        nvhdr_off = int_base + 6 * 4
        npts_off = int_base + 9 * 4

        nvhdr_le = struct.unpack("<i", header[nvhdr_off:nvhdr_off + 4])[0]
        npts_le = struct.unpack("<i", header[npts_off:npts_off + 4])[0]
        nvhdr_be = struct.unpack(">i", header[nvhdr_off:nvhdr_off + 4])[0]
        npts_be = struct.unpack(">i", header[npts_off:npts_off + 4])[0]

        def plausible(nvhdr, npts):
            return (0 < nvhdr < 50) and (0 < npts < 10_000_000_000)

        if plausible(nvhdr_le, npts_le):
            self.endian = "<"
            self.npts = int(npts_le)
            self._dtype = np.dtype("<f4")
        elif plausible(nvhdr_be, npts_be):
            self.endian = ">"
            self.npts = int(npts_be)
            self._dtype = np.dtype(">f4")
        else:
            self.endian = "<"
            self.npts = int(max(0, npts_le))
            self._dtype = np.dtype("<f4")

        delta = struct.unpack(self.endian + "f", header[0:4])[0]
        self.delta = float(delta) if delta > 0 else 0.0
        self.fs = float(1.0 / self.delta) if self.delta > 0 else 0.0

        self.data_offset = self.HEADER_BYTES

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def read_block(self, start: int, count: int) -> np.ndarray:
        start = int(start)
        count = int(count)
        out = np.zeros((count,), dtype=np.float32)

        if count <= 0:
            return out
        if start >= self.npts:
            return out

        end = min(start + count, self.npts)
        take = end - start
        if take <= 0:
            return out

        byte_start = self.data_offset + start * 4
        self.f.seek(byte_start, os.SEEK_SET)
        raw = self.f.read(take * 4)
        arr = np.frombuffer(raw, dtype=self._dtype)
        out[:take] = arr.astype(np.float32, copy=False)
        return out


class SACMultiChannelReader:
    def __init__(self, sac_folder: str, sac_idx_prefix: str, channel_count: int = 32):
        self.channel_count = int(channel_count)
        self.readers: List[SACChunkReader] = []

        for ch in range(self.channel_count):
            p = os.path.join(sac_folder, f"{sac_idx_prefix}{ch}.sac")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing SAC: {p}")
            self.readers.append(SACChunkReader(p))

        self.npts = min(r.npts for r in self.readers)
        self.fs = self.readers[0].fs if self.readers[0].fs > 0 else 1.0

    def close(self):
        for r in self.readers:
            r.close()

    def length(self) -> int:
        return int(self.npts)

    def read_block_all(self, start: int, count: int) -> np.ndarray:
        out = np.zeros((self.channel_count, int(count)), dtype=np.float32)
        for i, r in enumerate(self.readers):
            out[i, :] = r.read_block(start, count)
        return out


# =========================
# 流式滑窗
# =========================
def iter_stream_windows_32ch(
    reader32: SACMultiChannelReader,
    slice_len: int = 1000,
    pad_to: int = 1024,
    step: int = 100,
    chunk_size: int = 1_000_000,
):
    assert pad_to >= slice_len
    total_len = reader32.length()
    C = reader32.channel_count

    buf = np.zeros((C, 0), dtype=np.float32)
    file_read_pos = 0
    global_pos = 0

    while global_pos < total_len:
        while buf.shape[1] < slice_len and file_read_pos < total_len:
            to_read = min(int(chunk_size), total_len - file_read_pos)
            block = reader32.read_block_all(file_read_pos, to_read)
            file_read_pos += to_read
            buf = np.concatenate([buf, block], axis=1)

        take = min(slice_len, buf.shape[1])
        wave_32x1024 = np.zeros((C, pad_to), dtype=np.float32)
        if take > 0:
            wave_32x1024[:, :take] = buf[:, :take]

        yield global_pos, wave_32x1024

        global_pos += int(step)
        if buf.shape[1] > step:
            buf = buf[:, step:]
        else:
            buf = np.zeros((C, 0), dtype=np.float32)


# =========================
# 定位输入构造
# =========================
def _map_coord_repeat(v: float, scope: Tuple[float, float], length: int = 1024) -> np.ndarray:
    a = float(linear_mapping(scope[0], scope[1], 0.0, 1.0, v))
    a = float(max(0.0, min(1.0, a)))
    return np.full((length,), a, dtype=np.float32)


def build_locate_input_32x1024x8(
    wave_32x1024_norm: np.ndarray,
    stn_xyz: List[List[float]],
    scope_xy: Tuple[float, float],
    scope_z: Tuple[float, float],
) -> np.ndarray:
    w = np.asarray(wave_32x1024_norm, dtype=np.float32)
    assert w.shape == (32, 1024)
    assert len(stn_xyz) == 32

    w = normalize_by_channel_absmax(w)

    combined = list(zip(w, stn_xyz))
    by_x = sorted(combined, key=lambda it: it[1][0])
    by_y = sorted(combined, key=lambda it: it[1][1])

    def pack(sorted_list):
        waves = np.stack([it[0] for it in sorted_list], axis=0)
        xs = np.stack([_map_coord_repeat(it[1][0], scope_xy) for it in sorted_list], axis=0)
        ys = np.stack([_map_coord_repeat(it[1][1], scope_xy) for it in sorted_list], axis=0)
        zs = np.stack([_map_coord_repeat(it[1][2], scope_z) for it in sorted_list], axis=0)
        return waves, xs, ys, zs

    wx, xx, yx, zx = pack(by_x)
    wy, xy, yy, zy = pack(by_y)

    feat = np.stack([wx, xx, yx, zx, wy, xy, yy, zy], axis=-1).astype(np.float32)
    return feat


# =========================
# 模型输出整理
# =========================
def to_1024(x) -> np.ndarray:
    a = np.asarray(x)
    a = np.squeeze(a)
    if a.ndim == 1 and a.shape[0] == 1024:
        return a.astype(np.float32)
    if a.ndim == 2 and a.shape in [(1024, 1), (1, 1024)]:
        return a.reshape(-1).astype(np.float32)
    raise ValueError(f"Detect output cannot be (1024,), got {a.shape}")


def to_2x1024(x) -> np.ndarray:
    a = np.asarray(x)
    a = np.squeeze(a)
    if a.ndim == 2 and a.shape == (2, 1024):
        return a.astype(np.float32)
    if a.ndim == 2 and a.shape == (1024, 2):
        return a.T.astype(np.float32)

    if a.ndim >= 2:
        axes = list(range(a.ndim))
        ax2 = next((i for i in axes if a.shape[i] == 2), None)
        ax1024 = next((i for i in axes if a.shape[i] == 1024), None)
        if ax2 is not None and ax1024 is not None and ax2 != ax1024:
            b = np.moveaxis(a, ax2, 0)
            ax1024_new = next(i for i in range(b.ndim) if b.shape[i] == 1024 and i != 0)
            b = np.moveaxis(b, ax1024_new, 1)
            if b.ndim > 2:
                b = b.mean(axis=tuple(range(2, b.ndim)))
            if b.shape == (2, 1024):
                return b.astype(np.float32)

    raise ValueError(f"Locate output cannot be (2,1024), got {a.shape}")


# =========================
# 事件聚类：簇内 Top-K(K=min_hits) 平均位置
# =========================
@dataclass
class CandidateMeta:
    global_start: int
    arrival_global_idx: int
    arrival_local_idx: int
    detect_conf: float
    locate_conf: float
    x_mm: float
    y_mm: float
    max_x: float
    max_y: float


@dataclass
class EventResult:
    event_id: int
    arrival_global_idx: int
    arrival_time_s: float
    x_mm: float
    y_mm: float
    locate_conf: float
    detect_conf: float
    hit_count: int
    rep_global_start: int
    rep_arrival_local_idx: int
    max_x: float
    max_y: float


class EventAggregator:

    def __init__(self, fs: float, same_time_samples: int, same_xy_mm: float, min_hits: int):
        self.fs = float(fs) if fs and fs > 0 else 1.0
        self.same_time_samples = int(same_time_samples)
        self.same_xy_mm = float(same_xy_mm)
        self.min_hits = int(min_hits)

        self._events: List[EventResult] = []
        self._next_id = 1

        self._count = 0
        self._last_arrival: Optional[int] = None
        self._last_xy: Optional[Tuple[float, float]] = None

        self._cands: List[CandidateMeta] = []

    def _belongs(self, cand: CandidateMeta) -> bool:
        if self._count == 0 or self._last_arrival is None or self._last_xy is None:
            return True
        dt = abs(int(cand.arrival_global_idx) - int(self._last_arrival))
        dxy = xy_distance_mm((cand.x_mm, cand.y_mm), self._last_xy)
        return (dt < self.same_time_samples) and (dxy < self.same_xy_mm)

    def _finalize(self):
        # 当前簇结束：判断是否输出事件
        if self._count >= self.min_hits and self._cands:
            # locate_conf 从大到小排序
            cands_sorted = sorted(self._cands, key=lambda c: float(c.locate_conf), reverse=True)

            # Top-K: K=min_hits
            k = int(self.min_hits)
            topk = cands_sorted[:k]

            # 事件位置：Top-K 的 (x_mm,y_mm) 平均
            x_mean = float(np.mean([c.x_mm for c in topk]))
            y_mean = float(np.mean([c.y_mm for c in topk]))

            # 代表解：Top-1（用于 rep_start、置信度等字段）
            b = cands_sorted[0]

            self._events.append(EventResult(
                event_id=self._next_id,
                arrival_global_idx=int(b.arrival_global_idx),
                arrival_time_s=float(b.arrival_global_idx) / self.fs,
                x_mm=float(x_mean),
                y_mm=float(y_mean),
                locate_conf=float(b.locate_conf),
                detect_conf=float(b.detect_conf),
                hit_count=int(self._count),
                rep_global_start=int(b.global_start),
                rep_arrival_local_idx=int(b.arrival_local_idx),
                max_x=float(b.max_x),
                max_y=float(b.max_y),
            ))
            self._next_id += 1

        # 清空簇状态
        self._count = 0
        self._last_arrival = None
        self._last_xy = None
        self._cands = []

    def add(self, cand: CandidateMeta):
        if self._count == 0:
            self._count = 1
            self._last_arrival = int(cand.arrival_global_idx)
            self._last_xy = (float(cand.x_mm), float(cand.y_mm))
            self._cands = [cand]
            return

        if self._belongs(cand):
            self._count += 1
            self._last_arrival = int(cand.arrival_global_idx)
            self._last_xy = (float(cand.x_mm), float(cand.y_mm))
            self._cands.append(cand)
        else:
            self._finalize()
            self._count = 1
            self._last_arrival = int(cand.arrival_global_idx)
            self._last_xy = (float(cand.x_mm), float(cand.y_mm))
            self._cands = [cand]

    def finalize(self) -> List[EventResult]:
        self._finalize()
        return self._events


# ==========================================================
# ====================画图函数（demo 风格）====================
# ==========================================================
def _pad_or_trim_1d_to_len(y: Optional[np.ndarray], new_len: int) -> np.ndarray:
    if y is None:
        return np.zeros((new_len,), dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    out = np.zeros((new_len,), dtype=np.float32)
    n = min(y.size, new_len)
    if n > 0:
        out[:n] = y[:n]
    return out


def plot_fullwave_32ch_with_events(
    sac_folder: str,
    sac_idx_prefix: str,
    save_path: str,
    fs: float,
    event_arrivals_global: List[int],
    channel_count: int = 32,
    chunk_size: int = 1_000_000,
    max_points: Union[int, str, None] = 20000,
    show_progress: bool = True,

    offset: float = 2.2,
    wave_lw: float = 1.5,
    baseline_lw: float = 0.7,
    event_lw: float = 0.6,
    fig_w: float = 12.0,
    fig_h: float = 18.0,
    ch_label_fontsize: float = 16.0,
    tick_labelsize: float = 16.0,
    axis_labelsize: float = 16.0,
    title_size: float = 20.0,
    bold: bool = True,
):
    ensure_dir(os.path.dirname(save_path))

    reader = SACMultiChannelReader(sac_folder, sac_idx_prefix, channel_count=channel_count)
    total_len = reader.length()
    fs = float(fs) if fs and fs > 0 else float(reader.fs)

    if max_points is None or (isinstance(max_points, str) and str(max_points).lower() == "all"):
        stride = 1
    else:
        mp = int(max_points)
        stride = int(math.ceil(total_len / float(mp))) if total_len > mp else 1
        stride = max(1, stride)

    max_abs = np.zeros((channel_count,), dtype=np.float64)
    ds_blocks = []

    pos = 0
    total_blocks = int(math.ceil(total_len / float(chunk_size))) if chunk_size > 0 else 1
    pbar = tqdm(total=total_blocks, desc="Plot Fullwave (read blocks)", ncols=110) if show_progress else None

    while pos < total_len:
        to_read = min(int(chunk_size), total_len - pos)
        block = reader.read_block_all(pos, to_read).astype(np.float32)

        max_abs = np.maximum(max_abs, np.max(np.abs(block), axis=1))

        offset_sel = (stride - (pos % stride)) % stride
        sel = block[:, offset_sel:to_read:stride]
        if sel.shape[1] > 0:
            ds_blocks.append(sel)

        pos += to_read
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    reader.close()

    if ds_blocks:
        ds = np.concatenate(ds_blocks, axis=1)
    else:
        ds = np.zeros((channel_count, 1), dtype=np.float32)

    eps = 1e-10
    denom = (max_abs + eps).reshape(channel_count, 1)
    ds_norm = ds / denom

    t = (np.arange(ds_norm.shape[1], dtype=np.float64) * stride) / fs

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    for sp in ["left", "bottom", "top", "right"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(1.5)
        ax.spines[sp].set_color("black")

    ax.tick_params(axis="both", which="both", width=1.4, length=4.5,
                   direction="out", colors="black", labelsize=tick_labelsize)

    for ch in range(channel_count):
        ax.hlines(y=ch * offset, xmin=float(t[0]), xmax=float(t[-1]),
                  colors="black", linewidth=baseline_lw, alpha=0.95)

    for ch in range(channel_count):
        ax.plot(t, ds_norm[ch] + ch * offset, color="black", linewidth=wave_lw, alpha=1.0)

    for a in (event_arrivals_global or []):
        ax.axvline(x=float(a) / fs, color="red", linestyle="--", linewidth=event_lw, alpha=0.45)

    yticks = [ch * offset for ch in range(channel_count)]
    ylabels = [f"CH{ch}" for ch in range(channel_count)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=ch_label_fontsize, color="black")

    ax.set_xlabel("Time (s)", fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    ax.set_ylabel("Channels", fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    ax.set_title(f"Full Waveforms (32ch, stacked) | stride={stride}",
                 fontsize=title_size, fontweight=("bold" if bold else "normal"), pad=10)

    ax.set_ylim(-1.5, (channel_count - 1) * offset + 1.5)
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.grid(False)

    if bold:
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_event_slice_32ch(
    wave_32xN_norm: np.ndarray,
    vline_pos: int,
    save_path: str,
    title: str,
    pdf_pred_1024: Optional[np.ndarray] = None,
    fs: float = 3_000_000.0,
    x_len: int = 1201,
    tick_step_samples: int = 300,
    time_unit: str = "us",

    pdf_scale: float = 1.0,
    offset: float = 2.2,
    wave_lw: float = 1.5,
    baseline_lw: float = 0.7,
    pdf_lw: float = 1.5,
    vline_lw: float = 1.2,
    fig_w: float = 12.0,
    fig_h: float = 18.0,
    ch_label_fontsize: float = 16.0,
    tick_labelsize: float = 16.0,
    axis_labelsize: float = 16.0,
    title_size: float = 20.0,
    bold: bool = True,
):
    ensure_dir(os.path.dirname(save_path))

    w_in = np.asarray(wave_32xN_norm, dtype=np.float32)
    if w_in.ndim != 2 or w_in.shape[0] != 32:
        raise ValueError(f"wave must be (32, N), got {w_in.shape}")

    C = 32
    x_len = int(x_len)
    fs = float(fs) if fs and fs > 0 else 1.0

    w = np.zeros((C, x_len), dtype=np.float32)
    ncopy = min(w_in.shape[1], x_len)
    w[:, :ncopy] = w_in[:, :ncopy]

    pdf = _pad_or_trim_1d_to_len(pdf_pred_1024, x_len)
    vline_pos = int(max(0, min(x_len - 1, int(vline_pos))))

    if str(time_unit).lower() == "us":
        scale = 1e6
        x = (np.arange(x_len, dtype=np.float64) / fs) * scale
        xlab = "Time (µs)"
        tick_pos_samples = np.arange(0, x_len, int(tick_step_samples), dtype=np.int32)
        tick_pos = (tick_pos_samples / fs) * scale
        tick_labels = [f"{int(round(v))}" for v in tick_pos]
    else:
        x = (np.arange(x_len, dtype=np.float64) / fs)
        xlab = "Time (s)"
        tick_pos_samples = np.arange(0, x_len, int(tick_step_samples), dtype=np.int32)
        tick_pos = (tick_pos_samples / fs)
        tick_labels = [f"{v:.6g}" for v in tick_pos]

    vline_x = float(x[vline_pos])

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    for sp in ["left", "bottom", "top", "right"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(1.5)
        ax.spines[sp].set_color("black")

    ax.tick_params(axis="both", which="both", width=1.4, length=4.5,
                   direction="out", colors="black", labelsize=tick_labelsize)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)

    for r in range(C):
        ax.hlines(y=r * offset, xmin=float(x[0]), xmax=float(x[-1]),
                  colors="black", linewidth=baseline_lw, alpha=0.95)

    for ch in range(C):
        ax.plot(x, w[ch] + ch * offset, color="black", linewidth=wave_lw, alpha=1.0)

    pdf_y0 = C * offset
    ax.plot(x, pdf_y0 + pdf_scale * pdf, color="red", linewidth=pdf_lw, alpha=0.95)
    ax.axvline(x=vline_x, color="red", linestyle="--", linewidth=vline_lw, alpha=0.75)

    rows = C + 1
    yticks = [r * offset for r in range(rows)]
    ylabels = [f"CH{ch}" for ch in range(C)] + ["PDF"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=ch_label_fontsize, color="black")

    ax.set_xlabel(xlab, fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    ax.set_title(title, fontsize=title_size, fontweight=("bold" if bold else "normal"), pad=10)

    ax.set_ylim(-1.5, (rows - 1) * offset + (pdf_scale * 1.0 + 0.8))
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.grid(False)

    if bold:
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_detect_gaussian(
    pred_1024: np.ndarray,
    save_path: str,
    title: str,
    curve_lw: float = 1.5,
    fig_w: float = 10.0,
    fig_h: float = 3.5,
    tick_labelsize: float = 10.0,
    axis_labelsize: float = 12.0,
    title_size: float = 13.0,
    bold: bool = True,
):
    ensure_dir(os.path.dirname(save_path))

    p = np.asarray(pred_1024, dtype=np.float32).reshape(-1)
    if p.size != 1024:
        raise ValueError(f"pred_1024 must be length 1024, got {p.size}")

    p = np.clip(p, 0.0, 1.0)
    x = np.arange(1024, dtype=np.int32)

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    for sp in ["left", "bottom", "top", "right"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(1.5)
        ax.spines[sp].set_color("black")

    ax.tick_params(axis="both", which="both", width=1.4, length=4.5,
                   direction="out", colors="black", labelsize=tick_labelsize)

    ax.plot(x, p, color="red", linewidth=curve_lw, alpha=0.95)

    ax.set_xlim(0, 1023)
    ax.set_ylim(0, 1)

    yticks = [0.0, 0.5, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.2f}" for t in yticks])

    ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.8)

    ax.set_xlabel("Sample", fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    ax.set_ylabel("Probability", fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    ax.set_title(title, fontsize=title_size, fontweight=("bold" if bold else "normal"), pad=10)

    if bold:
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_locate_xy(
    loc_2x1024: np.ndarray,
    save_path: str,
    title: str,
    curve_lw: float = 1.5,
    fig_w: float = 12.5,
    fig_h: float = 6.2,
    tick_labelsize: float = 10.0,
    axis_labelsize: float = 12.0,
    title_size: float = 13.0,
    bold: bool = True,
):
    ensure_dir(os.path.dirname(save_path))

    a = np.asarray(loc_2x1024, dtype=np.float32)
    if a.shape != (2, 1024):
        raise ValueError(f"loc_2x1024 must be (2,1024), got {a.shape}")

    a = np.clip(a, 0.0, 1.0)
    x = np.arange(1024, dtype=np.int32)
    px, py = a[0], a[1]

    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

    def style_axis(ax):
        for sp in ["left", "bottom", "top", "right"]:
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_linewidth(1.5)
            ax.spines[sp].set_color("black")
        ax.tick_params(axis="both", which="both", width=1.4, length=4.5,
                       direction="out", colors="black", labelsize=tick_labelsize)
        ax.set_xlim(0, 1023)
        ax.set_ylim(0, 1)
        yticks = [0.0, 0.5, 1.0]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{t:.2f}" for t in yticks])
        ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.8)

    ax0 = axes[0]
    ax0.plot(x, px, color="red", linewidth=curve_lw, alpha=0.95)
    ax0.set_ylabel("X Probability", fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    ax0.set_title(title, fontsize=title_size, fontweight=("bold" if bold else "normal"), pad=10)
    style_axis(ax0)
    ax0.tick_params(labelbottom=False)

    ax1 = axes[1]
    ax1.plot(x, py, color="red", linewidth=curve_lw, alpha=0.95)
    ax1.set_ylabel("Y Probability", fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    ax1.set_xlabel("Sample", fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    style_axis(ax1)

    if bold:
        for ax in axes:
            plt.setp(ax.get_xticklabels(), fontweight="bold")
            plt.setp(ax.get_yticklabels(), fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_xyloc_time_segments(
    xy_list: List[Tuple[float, float]],
    event_times_s: List[float],
    total_duration_s: float,
    save_path: str,
    xy_axis_range: Tuple[float, float] = (-250.0, 250.0),

    stn_xyz: Optional[List[List[float]]] = None,
    stn_marker_size: float = 42.0,
    stn_label_fontsize: float = 7.5,
    stn_z0_alpha: float = 0.45,
    stn_z48_alpha: float = 0.22,
    stn_edge_lw: float = 0.6,
    stn_label_dx: float = 6.0,
    stn_label_dy: float = 6.0,

    event_marker_size: float = 70.0,
    event_edge_lw: float = 0.6,

    tick_labelsize: float = 10.0,
    axis_labelsize: float = 12.0,
    title_size: float = 13.0,
    bold: bool = True,
):
    ensure_dir(os.path.dirname(save_path))
    xy = np.asarray(xy_list, dtype=np.float64)
    t = np.asarray(event_times_s, dtype=np.float64)

    fig = plt.figure(figsize=(8.6, 8.0))
    ax = plt.gca()

    ax.set_xlim(xy_axis_range[0], xy_axis_range[1])
    ax.set_ylim(xy_axis_range[0], xy_axis_range[1])
    ax.set_aspect("equal", adjustable="box")

    for sp in ["left", "bottom", "top", "right"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(1.5)
        ax.spines[sp].set_color("black")

    ax.tick_params(axis="both", which="both", width=1.4, length=4.5,
                   direction="out", colors="black", labelsize=tick_labelsize)

    ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.8)

    ax.set_xlabel("X (mm)", fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    ax.set_ylabel("Y (mm)", fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    ax.set_title("Representative Events (time-segment colored)",
                 fontsize=title_size, fontweight=("bold" if bold else "normal"), pad=10)

    if stn_xyz is not None:
        stn = np.asarray(stn_xyz, dtype=np.float64)
        if stn.shape != (32, 3):
            raise ValueError(f"stn_xyz must be (32,3), got {stn.shape}")

        sx, sy, sz = stn[:, 0], stn[:, 1], stn[:, 2]
        m0 = (sz == 0)
        m48 = (sz == 48)

        if np.any(m0):
            ax.scatter(sx[m0], sy[m0], s=stn_marker_size, marker="^",
                       c="black", alpha=stn_z0_alpha,
                       edgecolors="black", linewidths=stn_edge_lw,
                       zorder=1, label="Top stations")

        if np.any(m48):
            ax.scatter(sx[m48], sy[m48], s=stn_marker_size, marker="^",
                       c="black", alpha=stn_z48_alpha,
                       edgecolors="black", linewidths=stn_edge_lw,
                       zorder=1, label="Bottom stations")

        for i in range(32):
            alpha_text = (stn_z0_alpha * 0.95) if sz[i] == 0 else (stn_z48_alpha * 0.95)
            ax.text(float(sx[i] + stn_label_dx), float(sy[i] + stn_label_dy),
                    f"Stn{i}", fontsize=stn_label_fontsize, color="black", alpha=alpha_text,
                    ha="left", va="bottom", zorder=2)

    if xy.size != 0:
        colors = ["gold", "limegreen", "dodgerblue", "red"]
        edges = np.linspace(0.0, float(total_duration_s), 5)
        labels = [
            f"{edges[0]:.2f}–{edges[1]:.2f}s",
            f"{edges[1]:.2f}–{edges[2]:.2f}s",
            f"{edges[2]:.2f}–{edges[3]:.2f}s",
            f"{edges[3]:.2f}–{edges[4]:.2f}s",
        ]
        for k in range(4):
            m = (t >= edges[k]) & (t < edges[k + 1]) if k < 3 else (t >= edges[k]) & (t <= edges[k + 1])
            if np.any(m):
                ax.scatter(xy[m, 0], xy[m, 1], s=event_marker_size, marker="o",
                           c=colors[k], edgecolor="black", linewidths=event_edge_lw,
                           alpha=0.9, label=f"Events {labels[k]}", zorder=3)

    leg = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=8,
        markerscale=0.8,
        labelspacing=0.25,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.3,
    )
    frame = leg.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(0.9)
    frame.set_facecolor("none")
    frame.set_alpha(1.0)

    if bold:
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_event_frequency(
    event_times_s: List[float],
    total_duration_s: float,
    divisions: int,
    save_path: str,
    line_lw: float = 1.6,
    marker_size: float = 4.5,
    tick_labelsize: float = 10.0,
    axis_labelsize: float = 12.0,
    title_size: float = 13.0,
    bold: bool = True,
):
    ensure_dir(os.path.dirname(save_path))
    t = np.asarray(event_times_s, dtype=np.float64)

    divisions = int(max(1, divisions))
    edges = np.linspace(0.0, float(total_duration_s), divisions + 1)
    counts, _ = np.histogram(t, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = plt.figure(figsize=(11.5, 3.8))
    ax = plt.gca()

    for sp in ["left", "bottom", "top", "right"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(1.5)
        ax.spines[sp].set_color("black")

    ax.tick_params(axis="both", which="both", width=1.4, length=4.5,
                   direction="out", colors="black", labelsize=tick_labelsize)

    ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.8)

    ax.plot(centers, counts, color="black", marker="o",
            linewidth=line_lw, markersize=marker_size, alpha=0.95)

    ax.set_xlim(0, float(total_duration_s))
    ax.set_ylim(0, max(1, int(np.max(counts)) + 1 if counts.size else 1))

    ax.set_xlabel("Time (s)", fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    ax.set_ylabel("Event count", fontsize=axis_labelsize, fontweight=("bold" if bold else "normal"))
    ax.set_title(f"Event Frequency (bin = {float(total_duration_s)/divisions:.3g} s)",
                 fontsize=title_size, fontweight=("bold" if bold else "normal"), pad=10)

    if bold:
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# 主流程（检测/定位/聚类；聚类 finalize 改为簇内 Top-K 平均位置）
# =========================
def run_pipeline(
    sac_folder_path: str,
    sac_idx_prefix: str,
    stn_xyz: List[List[float]],
    detect_model_path: str,
    locate_model_path: str,
    base_output_dir: str,

    slice_len: int = 1000,
    pad_to: int = 1024,
    step: int = 100,
    chunk_size: int = 1_000_000,

    detect_conf_th: float = 0.7,

    locate_conf_th_x: float = 0.7,
    locate_conf_th_y: float = 0.7,

    same_time_samples: int = 1200,
    same_xy_mm: float = 10.0,
    min_hits: int = 2,

    out_xy_minmax: Tuple[float, float] = (-312.5, 312.5),

    scope_xy_for_feat: Tuple[float, float] = (-250.0, 250.0),
    scope_z_for_feat: Tuple[float, float] = (0.0, 48.0),

    detect_batch_windows: int = 256,
    detect_batch_size: int = 64,
    locate_batch_size: int = 16,

    xy_plot_range: Tuple[float, float] = (-250.0, 250.0),
    freq_divisions: int = 10,

    fullwave_max_points: Union[int, str, None] = 20000,
    show_tqdm: bool = True,

    slice_plot_len: int = 1201,
    slice_plot_tick_step_samples: int = 300,
    slice_plot_time_unit: str = "us",
):
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    base_output_dir = os.path.abspath(base_output_dir) if base_output_dir else os.path.abspath("./outputs")
    ensure_dir(base_output_dir)

    dir_fullwave = os.path.join(base_output_dir, "01_fullwave")
    dir_slices = os.path.join(base_output_dir, "02_event_slices")
    dir_detect = os.path.join(base_output_dir, "03_detect_pred")
    dir_locate = os.path.join(base_output_dir, "04_locate_pred")
    ensure_dir(dir_fullwave)
    ensure_dir(dir_slices)
    ensure_dir(dir_detect)
    ensure_dir(dir_locate)

    path_events_txt = os.path.join(base_output_dir, "events_catalog.txt")
    path_events_jsonl = os.path.join(base_output_dir, "events_catalog.jsonl")
    path_all_events_png = os.path.join(base_output_dir, "all_events.png")
    path_freq_png = os.path.join(base_output_dir, "event_frequency.png")

    reader32 = SACMultiChannelReader(sac_folder_path, sac_idx_prefix, channel_count=32)
    fs = float(reader32.fs) if reader32.fs > 0 else 1.0
    total_len = reader32.length()
    total_duration_s = float(total_len) / fs if total_len > 0 else 0.0
    total_windows_est = max(1, int(math.ceil(total_len / float(step))))

    detect_model = keras.models.load_model(
        detect_model_path,
        custom_objects={'ChannelAttention': ChannelAttention, 'CBAMAttention': CBAMAttention},
        compile=False
    )

    locate_model = keras.models.load_model(
        locate_model_path,
        custom_objects={'ChannelAttention': ChannelAttention, 'CBAMAttention': CBAMAttention},
        compile=False
    )

    aggregator = EventAggregator(
        fs=fs,
        same_time_samples=same_time_samples,
        same_xy_mm=same_xy_mm,
        min_hits=min_hits
    )

    batch_waves = []
    batch_starts = []
    passed_candidates = 0

    def flush_detect_batch(pbar: Optional[tqdm] = None):
        nonlocal passed_candidates, batch_waves, batch_starts
        if not batch_waves:
            return

        B = len(batch_waves)
        det_in = np.stack(batch_waves, axis=0).astype(np.float32)
        det_in = det_in[..., np.newaxis]
        det_out = detect_model.predict(det_in, batch_size=max(1, detect_batch_size), verbose=0)

        cand_waves = []
        cand_meta = []

        for i in range(B):
            global_start = int(batch_starts[i])
            pred = to_1024(det_out[i])
            arrival_local = int(np.argmax(pred))
            detect_conf = float(pred[arrival_local])
            if detect_conf < float(detect_conf_th):
                continue
            arrival_global = int(global_start + arrival_local)
            cand_waves.append(batch_waves[i])
            cand_meta.append((global_start, arrival_local, arrival_global, detect_conf))

        if cand_waves:
            loc_inputs = []
            for w in cand_waves:
                feat = build_locate_input_32x1024x8(
                    wave_32x1024_norm=w,
                    stn_xyz=stn_xyz,
                    scope_xy=scope_xy_for_feat,
                    scope_z=scope_z_for_feat,
                )
                loc_inputs.append(feat)

            loc_in = np.stack(loc_inputs, axis=0).astype(np.float32)
            loc_out = locate_model.predict(loc_in, batch_size=max(1, locate_batch_size), verbose=0)

            for k in range(len(cand_meta)):
                global_start, arrival_local, arrival_global, detect_conf = cand_meta[k]
                arr2 = to_2x1024(loc_out[k])

                px, py = arr2[0], arr2[1]
                ix = int(np.argmax(px)); max_x = float(px[ix])
                iy = int(np.argmax(py)); max_y = float(py[iy])

                locate_conf = 0.5 * (max_x + max_y)

                # 双阈值：必须同时过
                if (max_x < float(locate_conf_th_x)) or (max_y < float(locate_conf_th_y)):
                    continue

                xy_min, xy_max = float(out_xy_minmax[0]), float(out_xy_minmax[1])
                x_mm = idx_to_coord(ix, xy_min, xy_max, 0, 1023)
                y_mm = idx_to_coord(iy, xy_min, xy_max, 0, 1023)

                aggregator.add(CandidateMeta(
                    global_start=int(global_start),
                    arrival_global_idx=int(arrival_global),
                    arrival_local_idx=int(arrival_local),
                    detect_conf=float(detect_conf),
                    locate_conf=float(locate_conf),
                    x_mm=float(x_mm),
                    y_mm=float(y_mm),
                    max_x=float(max_x),
                    max_y=float(max_y),
                ))
                passed_candidates += 1

        if pbar is not None:
            pbar.set_postfix_str(f"cand_pass={passed_candidates}")

        batch_waves.clear()
        batch_starts.clear()

    pbar = tqdm(total=total_windows_est, desc="Detect+Locate (stream)", ncols=110) if show_tqdm else None

    try:
        for global_start, wave_32x1024_raw in iter_stream_windows_32ch(
            reader32=reader32,
            slice_len=slice_len,
            pad_to=pad_to,
            step=step,
            chunk_size=chunk_size,
        ):
            # detect 输入：仍然是全局 absmax 归一化
            wave_norm = normalize_by_global_absmax(wave_32x1024_raw)

            batch_waves.append(wave_norm.astype(np.float32))
            batch_starts.append(int(global_start))

            if len(batch_waves) >= int(max(1, detect_batch_windows)):
                flush_detect_batch(pbar)

            if pbar is not None:
                pbar.update(1)

        flush_detect_batch(pbar)

    finally:
        reader32.close()
        if pbar is not None:
            pbar.close()

    events = aggregator.finalize()

    # 输出 catalog
    with open(path_events_jsonl, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev.__dict__, ensure_ascii=False) + "\n")

    with open(path_events_txt, "w", encoding="utf-8") as f:
        f.write("EventID\tArrivalGlobalIdx\tArrivalTime_s\tX_mm\tY_mm\tLocateConf\tMaxX\tMaxY\tDetectConf\tHitCount\n")
        for ev in events:
            f.write(
                f"{ev.event_id}\t{ev.arrival_global_idx}\t{ev.arrival_time_s:.6f}\t"
                f"{ev.x_mm:.3f}\t{ev.y_mm:.3f}\t{ev.locate_conf:.3f}\t"
                f"{ev.max_x:.3f}\t{ev.max_y:.3f}\t{ev.detect_conf:.3f}\t{ev.hit_count}\n"
            )

    print(f"[Done] representative events={len(events)}  | saved:\n  {path_events_txt}\n  {path_events_jsonl}")

    # ===== fullwave =====
    fullwave_png = os.path.join(dir_fullwave, "fullwave_32ch_with_events.png")
    rep_arrivals = [int(ev.arrival_global_idx) for ev in events]
    plot_fullwave_32ch_with_events(
        sac_folder=sac_folder_path,
        sac_idx_prefix=sac_idx_prefix,
        save_path=fullwave_png,
        fs=fs,
        event_arrivals_global=rep_arrivals,
        channel_count=32,
        chunk_size=chunk_size,
        max_points=fullwave_max_points,
        show_progress=show_tqdm
    )
    print(f"[Saved] {fullwave_png}")

    # ===== per-event plots =====
    reader32b = SACMultiChannelReader(sac_folder_path, sac_idx_prefix, channel_count=32)
    it_events = tqdm(events, desc="Save per-event plots", ncols=110) if show_tqdm else events

    for ev in it_events:
        s0 = int(ev.rep_global_start)

        block_model = reader32b.read_block_all(s0, slice_len)
        wave_model_32x1024 = np.zeros((32, pad_to), dtype=np.float32)
        wave_model_32x1024[:, :block_model.shape[1]] = block_model[:, :block_model.shape[1]]
        wave_model_norm = normalize_by_global_absmax(wave_model_32x1024)

        det_in = wave_model_norm[np.newaxis, ..., np.newaxis].astype(np.float32)
        det_out = detect_model.predict(det_in, batch_size=1, verbose=0)
        pred_1024 = to_1024(det_out[0])
        arrival_local = int(np.argmax(pred_1024))

        feat = build_locate_input_32x1024x8(
            wave_32x1024_norm=wave_model_norm,
            stn_xyz=stn_xyz,
            scope_xy=scope_xy_for_feat,
            scope_z=scope_z_for_feat,
        )
        loc_in = feat[np.newaxis, ...].astype(np.float32)
        loc_out = locate_model.predict(loc_in, batch_size=1, verbose=0)
        loc_2x1024 = to_2x1024(loc_out[0])

        block_plot = reader32b.read_block_all(s0, int(slice_plot_len))
        wave_plot_32x1201 = np.zeros((32, int(slice_plot_len)), dtype=np.float32)
        wave_plot_32x1201[:, :block_plot.shape[1]] = block_plot[:, :block_plot.shape[1]]
        wave_plot_norm = normalize_by_global_absmax(wave_plot_32x1201)

        plot_event_slice_32ch(
            wave_32xN_norm=wave_plot_norm,
            vline_pos=arrival_local,
            pdf_pred_1024=pred_1024,
            fs=fs,
            x_len=int(slice_plot_len),
            tick_step_samples=int(slice_plot_tick_step_samples),
            time_unit=str(slice_plot_time_unit),
            save_path=os.path.join(dir_slices, f"event_{ev.event_id:04d}_slice.png"),
            title=f"Event Slice | Event {ev.event_id} | rep_start={s0} | local_arrival={arrival_local}",
        )

        plot_detect_gaussian(
            pred_1024=pred_1024,
            save_path=os.path.join(dir_detect, f"event_{ev.event_id:04d}_detect.png"),
            title=f"Detect Gaussian | Event {ev.event_id}",
        )

        plot_locate_xy(
            loc_2x1024=loc_2x1024,
            save_path=os.path.join(dir_locate, f"event_{ev.event_id:04d}_locateXY.png"),
            title=f"Locate Output (X/Y) | Event {ev.event_id}",
        )

    reader32b.close()

    xy_list = [(float(ev.x_mm), float(ev.y_mm)) for ev in events]
    t_list = [float(ev.arrival_time_s) for ev in events]

    plot_xyloc_time_segments(
        xy_list=xy_list,
        event_times_s=t_list,
        total_duration_s=total_duration_s,
        save_path=path_all_events_png,
        xy_axis_range=xy_plot_range,
        stn_xyz=stn_xyz,
    )
    print(f"[Saved] {path_all_events_png}")

    plot_event_frequency(
        event_times_s=t_list,
        total_duration_s=total_duration_s,
        divisions=freq_divisions,
        save_path=path_freq_png,
    )
    print(f"[Saved] {path_freq_png}")

    print(f"[All Done] base_output_dir={base_output_dir}")
    return events, fs, base_output_dir


if __name__ == "__main__":
    sac_folder_path = r"F:\event_sac_decay\idx0"
    sac_idx_prefix = "idx0_"
    detect_model_path = r"E:\lgx\code\AEMNet\detect\model\detect_model20_decay_0.2.hdf5"
    locate_model_path = r"E:\lgx\code\AEMNet\locate\model\locate_model20_xy_50_950_SNR_0.03.hdf5"
    base_output_dir = r"F:\results/event0_0.7_0.4_0.4_200_2_SNR_avg_0.2"

    stn_xyz = [
        [83, -155, 0], [-29, -63, 0], [-162, 81, 0], [4, -170, 0], [-167, -4, 0], [-75, -167, 0],
        [-173, -87, 0], [-132, -159, 0],
        [233, -212, 48], [170, -148, 48], [106, -85, 48], [42, -21, 48], [-21, 42, 48], [-85, 106, 48],
        [-148, 170, 48], [-212, 233, 48], [-212, 233, 0], [-163, 184, 0], [-113, 134, 0], [-64, 85, 0],
        [-14, 35, 0], [35, -14, 0], [85, -64, 0], [134, -113, 0], [184, -162, 0], [233, -212, 0],
        [144, 140, 0], [83, -165, 48], [-45, -168, 48], [-91, -86, 48], [-110, 22, 48], [-180, -18, 48]
    ]

    run_pipeline(
        sac_folder_path=sac_folder_path,
        sac_idx_prefix=sac_idx_prefix,
        stn_xyz=stn_xyz,
        detect_model_path=detect_model_path,
        locate_model_path=locate_model_path,
        base_output_dir=base_output_dir,

        slice_len=1000,
        pad_to=1024,
        step=100,
        chunk_size=1_000_000,

        detect_conf_th=0.7,

        locate_conf_th_x=0.4,
        locate_conf_th_y=0.4,

        same_time_samples=100,
        same_xy_mm=200,
        min_hits=2,

        out_xy_minmax=(-312.5, 312.5),
        scope_xy_for_feat=(-250.0, 250.0),
        scope_z_for_feat=(0.0, 48.0),

        detect_batch_windows=256,
        detect_batch_size=64,
        locate_batch_size=16,

        xy_plot_range=(-250.0, 250.0),
        freq_divisions=10,

        fullwave_max_points=1000000,
        show_tqdm=True,

        slice_plot_len=1201,
        slice_plot_tick_step_samples=300,
        slice_plot_time_unit="us",
    )
