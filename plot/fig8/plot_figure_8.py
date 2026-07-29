# -*- coding: utf-8 -*-
"""
Draw Figure 8 panels e-g for AE event catalogs.

Figure logic:
    (e) Event rate: all detected events
    (f) Distance-to-fault quantiles: high-confidence events
    (g) Along-fault space-time heatmap: high-confidence events

High-confidence condition:
    DetectConf >= 0.95
    LocateConf >= 0.60

Input files must contain at least these columns:
    ArrivalTime_s, X_mm, Y_mm, DetectConf, LocateConf

How to use:
    1. Modify CATALOG_PATHS and OUTPUT_DIR below.
    2. Double-click this .py file or run it in Python.

Optional command-line use:
    python plot_figure8_efg.py events_catalog0.txt events_catalog1.txt events_catalog2.txt events_catalog3.txt output_folder
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, PowerNorm


# =============================================================================
# 1. Input and output settings
# =============================================================================

# 修改这里：4 个检测定位结果文件路径，顺序对应 Slip-1 至 Slip-4
CATALOG_PATHS = [
    r"E:\桌面\图片\paper_fig\fig7\events_catalog0.txt",
    r"E:\桌面\图片\paper_fig\fig7\events_catalog1.txt",
    r"E:\桌面\图片\paper_fig\fig7\events_catalog2.txt",
    r"E:\桌面\图片\paper_fig\fig7\events_catalog3.txt",
]

# 修改这里：输出文件夹
OUTPUT_DIR = r"E:\桌面\图片\paper_fig\fig8"

# 输出文件名前缀
OUTPUT_BASENAME = "figure8_efg"

# 四条曲线名称
SLIP_NAMES = ["Slip-1", "Slip-2", "Slip-3", "Slip-4"]


# =============================================================================
# 2. Plot parameters
# =============================================================================

# 高置信度筛选阈值
DETECT_CONF_THRESHOLD = 0.95
LOCATE_CONF_THRESHOLD = 0.60

# 时间范围和 bin 宽度
TIME_MIN = 0.0
TIME_MAX = 10.0
TIME_BIN_WIDTH = 0.5

# 沿断层方向 bin 宽度，单位 mm
S_BIN_WIDTH = 20.0

# g 图颜色加深速度。gamma < 1 表示颜色加深更快
POWER_GAMMA = 0.65

# g 图颜色：低值为浅灰蓝，高值为深蓝
SOFT_BLUE_COLORS = [
    "#eaf0f6",
    "#d7e4f0",
    "#b6d1e6",
    "#7fb3d5",
    "#3f88c5",
    "#0b3c6f",
]

# 图尺寸
FIGSIZE = (11.4, 6.17)

# 输出分辨率
DPI = 300


# =============================================================================
# 3. Functions
# =============================================================================

def parse_command_line_args():
    """
    Optional command-line mode:
        python plot_figure8_efg.py file0 file1 file2 file3 output_dir
    """
    if len(sys.argv) == 6:
        catalog_paths = sys.argv[1:5]
        output_dir = sys.argv[5]
        return catalog_paths, output_dir

    return CATALOG_PATHS, OUTPUT_DIR


def read_catalogs(catalog_paths, slip_names):
    frames = []

    if len(catalog_paths) != 4:
        raise ValueError("CATALOG_PATHS must contain exactly 4 file paths.")

    for path_str, slip_name in zip(catalog_paths, slip_names):
        path = Path(path_str)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        df = pd.read_csv(path, sep="\t")
        df["Slip"] = slip_name
        df["SourceFile"] = str(path)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    required_cols = ["ArrivalTime_s", "X_mm", "Y_mm", "DetectConf", "LocateConf"]
    missing = [col for col in required_cols if col not in all_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required_cols:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    all_df = all_df.dropna(subset=required_cols)
    all_df = all_df[
        (all_df["ArrivalTime_s"] >= TIME_MIN)
        & (all_df["ArrivalTime_s"] <= TIME_MAX)
    ].copy()

    if len(all_df) == 0:
        raise ValueError("No valid events remain after time filtering.")

    return all_df


def add_fault_coordinates(df):
    """
    Approximate fault trace:
        x + y = 0

    Distance to fault:
        d = |x + y| / sqrt(2)

    Along-fault coordinate:
        s = (x - y) / sqrt(2)
    """
    df = df.copy()
    df["DistanceToFault_mm"] = np.abs(df["X_mm"] + df["Y_mm"]) / np.sqrt(2.0)
    df["AlongFault_mm"] = (df["X_mm"] - df["Y_mm"]) / np.sqrt(2.0)
    return df


def setup_matplotlib_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["xtick.major.width"] = 0.9
    plt.rcParams["ytick.major.width"] = 0.9
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def main():
    catalog_paths, output_dir = parse_command_line_args()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df = read_catalogs(catalog_paths, SLIP_NAMES)

    # 原图逻辑：
    # e 图使用全部事件
    # f、g 图使用高置信度事件
    hc_df = all_df[
        (all_df["DetectConf"] >= DETECT_CONF_THRESHOLD)
        & (all_df["LocateConf"] >= LOCATE_CONF_THRESHOLD)
    ].copy()

    if len(hc_df) == 0:
        raise ValueError(
            "No high-confidence events remain. "
            "Please check DetectConf/LocateConf thresholds."
        )

    all_df = add_fault_coordinates(all_df)
    hc_df = add_fault_coordinates(hc_df)

    time_edges = np.arange(TIME_MIN, TIME_MAX + TIME_BIN_WIDTH, TIME_BIN_WIDTH)
    time_centers = (time_edges[:-1] + time_edges[1:]) / 2.0

    # -------------------------------------------------------------------------
    # Panel e: Event rate, all events
    # -------------------------------------------------------------------------
    event_rate_by_slip = {}
    for slip_name in SLIP_NAMES:
        sub = all_df[all_df["Slip"] == slip_name]
        counts, _ = np.histogram(sub["ArrivalTime_s"].to_numpy(), bins=time_edges)
        event_rate_by_slip[slip_name] = counts / TIME_BIN_WIDTH

    # -------------------------------------------------------------------------
    # Panel f: Distance-to-fault quantiles, high-confidence events
    # -------------------------------------------------------------------------
    q25, q50, q75 = [], [], []

    for i in range(len(time_edges) - 1):
        left, right = time_edges[i], time_edges[i + 1]

        if i < len(time_edges) - 2:
            sub = hc_df[
                (hc_df["ArrivalTime_s"] >= left)
                & (hc_df["ArrivalTime_s"] < right)
            ]
        else:
            sub = hc_df[
                (hc_df["ArrivalTime_s"] >= left)
                & (hc_df["ArrivalTime_s"] <= right)
            ]

        if len(sub) > 0:
            q25.append(float(sub["DistanceToFault_mm"].quantile(0.25)))
            q50.append(float(sub["DistanceToFault_mm"].quantile(0.50)))
            q75.append(float(sub["DistanceToFault_mm"].quantile(0.75)))
        else:
            q25.append(np.nan)
            q50.append(np.nan)
            q75.append(np.nan)

    q25 = np.array(q25, dtype=float)
    q50 = np.array(q50, dtype=float)
    q75 = np.array(q75, dtype=float)

    # -------------------------------------------------------------------------
    # Panel g: Along-fault heatmap, high-confidence events
    # -------------------------------------------------------------------------
    s_min = np.floor(hc_df["AlongFault_mm"].min() / S_BIN_WIDTH) * S_BIN_WIDTH
    s_max = np.ceil(hc_df["AlongFault_mm"].max() / S_BIN_WIDTH) * S_BIN_WIDTH
    s_edges = np.arange(s_min, s_max + S_BIN_WIDTH, S_BIN_WIDTH)

    heatmap_counts, t_edges, s_edges = np.histogram2d(
        hc_df["ArrivalTime_s"].to_numpy(),
        hc_df["AlongFault_mm"].to_numpy(),
        bins=[time_edges, s_edges],
    )

    cmap = LinearSegmentedColormap.from_list(
        "soft_blue_journal",
        SOFT_BLUE_COLORS
    )

    setup_matplotlib_style()

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 1.22],
        height_ratios=[1.0, 1.0],
    )

    ax_e = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[1, 0])
    ax_g = fig.add_subplot(gs[:, 1])

    # -------------------------------------------------------------------------
    # Draw panel e
    # -------------------------------------------------------------------------
    for slip_name in SLIP_NAMES:
        ax_e.plot(
            time_centers,
            event_rate_by_slip[slip_name],
            linewidth=1.5,
            label=slip_name,
        )

    ax_e.set_xlim(TIME_MIN, TIME_MAX)
    ax_e.set_ylim(bottom=0)
    ax_e.set_ylabel("Event rate (s$^{-1}$)")
    ax_e.set_xlabel("Time in final 10 s before failure (s)")
    ax_e.text(
        0.02,
        0.96,
        "(a)",
        transform=ax_e.transAxes,
        ha="left",
        va="top",
        fontsize=13,
    )
    ax_e.legend(
        frameon=False,
        fontsize=9,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.09, 1.02),
    )
    ax_e.tick_params(direction="out", length=3.5, width=0.9)

    # -------------------------------------------------------------------------
    # Draw panel f
    # -------------------------------------------------------------------------
    ax_f.fill_between(
        time_centers,
        q25,
        q75,
        alpha=0.20,
        label="25th–75th percentile",
    )
    ax_f.plot(time_centers, q25, linewidth=1.25, label="25th percentile")
    ax_f.plot(time_centers, q50, linewidth=1.70, label="Median")
    ax_f.plot(time_centers, q75, linewidth=1.25, label="75th percentile")

    ax_f.set_xlim(TIME_MIN, TIME_MAX)
    ax_f.set_ylim(bottom=0)
    ax_f.set_xlabel("Time in final 10 s before failure (s)")
    ax_f.set_ylabel("Distance to fault trace (mm)")
    ax_f.text(
        0.02,
        0.96,
        "(b)",
        transform=ax_f.transAxes,
        ha="left",
        va="top",
        fontsize=13,
    )
    ax_f.legend(
        frameon=False,
        fontsize=8.8,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.08, 1.02),
    )
    ax_f.tick_params(direction="out", length=3.5, width=0.9)

    # -------------------------------------------------------------------------
    # Draw panel g
    # -------------------------------------------------------------------------
    max_count = np.nanmax(heatmap_counts)
    if max_count <= 0:
        norm = None
    else:
        norm = PowerNorm(gamma=POWER_GAMMA, vmin=0, vmax=max_count)

    mesh = ax_g.pcolormesh(
        t_edges,
        s_edges,
        heatmap_counts.T,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )

    ax_g.set_xlim(TIME_MIN, TIME_MAX)
    ax_g.set_xlabel("Time in final 10 s before failure (s)")
    ax_g.set_ylabel("Along-fault coordinate, $s$ (mm)")
    ax_g.text(
        0.02,
        0.96,
        "(c)",
        transform=ax_g.transAxes,
        ha="left",
        va="top",
        fontsize=13,
    )
    ax_g.tick_params(direction="out", length=3.5, width=0.9)

    cbar = fig.colorbar(mesh, ax=ax_g, fraction=0.046, pad=0.025)
    cbar.set_label("Event count")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    out_png = output_dir / f"{OUTPUT_BASENAME}.png"
    out_pdf = output_dir / f"{OUTPUT_BASENAME}.pdf"
    out_svg = output_dir / f"{OUTPUT_BASENAME}.svg"

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("Done.")
    print(f"All events used for panel e: {len(all_df)}")
    print(f"High-confidence events used for panels f and g: {len(hc_df)}")
    print(f"Output PNG: {out_png}")
    print(f"Output PDF: {out_pdf}")
    print(f"Output SVG: {out_svg}")


if __name__ == "__main__":
    main()
