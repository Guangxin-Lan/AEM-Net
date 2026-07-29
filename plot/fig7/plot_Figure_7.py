# -*- coding: utf-8 -*-
"""
将 4 个检测定位结果文件绘制为单独一张 2×2 的定位图。

功能说明：
1. 读取 4 个结果文件（制表符分隔 txt）
2. 保留原始定位图画法：
   - 高置信度事件筛选：LocateConf >= 0.60 且 DetectConf >= 0.95
   - 散点颜色表示 LocateConf
   - 断层迹线：y = -x
   - DBSCAN 聚类红圈：eps = 12 mm, min_samples = 10
   - 分阶段中位位置：0–6, 6–8.5, 8.5–9, 9–9.5, 9.5–10 s
   - 用箭头连接阶段中位位置
3. 输出 PNG / PDF / SVG 三种格式

使用方法：
- 直接修改下面“用户输入区”的 4 个文件路径和输出文件夹路径
- 保存后直接运行该 .py 文件即可（不需要命令行）
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from sklearn.cluster import DBSCAN


# =============================================================================
# 用户输入区（只需要修改这里）
# =============================================================================

# 4 个结果文件路径（按 Slip-1, Slip-2, Slip-3, Slip-4 的顺序填写）
CATALOG_PATHS = [
    r"E:\桌面\图片\paper_fig\fig7\events_catalog0.txt",
    r"E:\桌面\图片\paper_fig\fig7\events_catalog1.txt",
    r"E:\桌面\图片\paper_fig\fig7\events_catalog2.txt",
    r"E:\桌面\图片\paper_fig\fig7\events_catalog3.txt",
]

# 输出文件夹路径
OUTPUT_DIR = r"E:\桌面\图片\paper_fig\fig9"

# 输出文件名前缀
OUTPUT_BASENAME = "figure_spatial_4maps_2x2"

# 每个文件对应的名称
SLIP_NAMES = ["Slip-1", "Slip-2", "Slip-3", "Slip-4"]


# =============================================================================
# 参数设置
# =============================================================================

# 高置信度筛选阈值
LOCATE_CONF_THRESHOLD = 0.60
DETECT_CONF_THRESHOLD = 0.95

# 时间范围（秒）
TIME_MIN = 0.0
TIME_MAX = 10.0

# DBSCAN 参数（与之前图一致）
DBSCAN_EPS_MM = 12.0
DBSCAN_MIN_SAMPLES = 10

# 分阶段时间窗（与之前图一致）
STAGE_BINS = [
    (0.0, 6.0),
    (6.0, 8.5),
    (8.5, 9.0),
    (9.0, 9.5),
    (9.5, 10.0),
]
STAGE_LABELS = ["0-6", "6-8.5", "8.5-9", "9-9.5", "9.5-10"]
STAGE_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

# 坐标范围（mm）
XY_MIN = -250
XY_MAX = 250

# 输出分辨率
PNG_DPI = 600


# =============================================================================
# 画图风格
# =============================================================================

def setup_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["font.size"] = 10.8
    plt.rcParams["axes.linewidth"] = 0.95
    plt.rcParams["xtick.major.width"] = 0.95
    plt.rcParams["ytick.major.width"] = 0.95
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


# =============================================================================
# 数据读取与处理
# =============================================================================

def read_catalog(path, slip_name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    df = pd.read_csv(path, sep="\t")
    df["Catalog"] = slip_name

    required_cols = ["ArrivalTime_s", "X_mm", "Y_mm", "LocateConf", "DetectConf"]
    for col in required_cols + ["HitCount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"文件 {path.name} 缺少必要列: {missing_cols}")

    df = df.dropna(subset=required_cols)
    df = df[(df["ArrivalTime_s"] >= TIME_MIN) & (df["ArrivalTime_s"] <= TIME_MAX)].copy()
    return df


def read_all_catalogs(catalog_paths, slip_names):
    if len(catalog_paths) != 4:
        raise ValueError("必须传入 4 个结果文件路径。")
    if len(slip_names) != 4:
        raise ValueError("SLIP_NAMES 必须有 4 个名称。")

    dfs = []
    for p, n in zip(catalog_paths, slip_names):
        dfs.append(read_catalog(p, n))
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def filter_high_confidence(df):
    hc = df[
        (df["LocateConf"] >= LOCATE_CONF_THRESHOLD) &
        (df["DetectConf"] >= DETECT_CONF_THRESHOLD)
    ].copy()
    return hc


# =============================================================================
# 聚类与阶段中位位置计算
# =============================================================================

def compute_cluster_and_stage_info(hc_df, slip_names):
    cluster_rows = []
    stage_rows = []

    for slip_name in slip_names:
        df = hc_df[hc_df["Catalog"] == slip_name].copy()

        # ---------- DBSCAN 聚类 ----------
        if len(df) >= DBSCAN_MIN_SAMPLES:
            coords = df[["X_mm", "Y_mm"]].to_numpy()
            labels = DBSCAN(eps=DBSCAN_EPS_MM, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(coords)
            df["cluster"] = labels

            valid = df[df["cluster"] >= 0].copy()
            for cid, sub in valid.groupby("cluster"):
                cx = float(sub["X_mm"].median())
                cy = float(sub["Y_mm"].median())
                radius = float(np.sqrt((sub["X_mm"] - cx) ** 2 + (sub["Y_mm"] - cy) ** 2).quantile(0.95))
                radius = max(radius + 7.0, 13.0)
                cluster_rows.append({
                    "Catalog": slip_name,
                    "ClusterID": int(cid),
                    "Count": int(len(sub)),
                    "CenterX_mm": cx,
                    "CenterY_mm": cy,
                    "Radius_mm": radius,
                })

        # ---------- 分阶段中位位置 ----------
        for (t0, t1), lab in zip(STAGE_BINS, STAGE_LABELS):
            if t1 < TIME_MAX:
                sub = df[(df["ArrivalTime_s"] >= t0) & (df["ArrivalTime_s"] < t1)].copy()
            else:
                sub = df[(df["ArrivalTime_s"] >= t0) & (df["ArrivalTime_s"] <= t1)].copy()

            stage_rows.append({
                "Catalog": slip_name,
                "Stage": lab,
                "t0": t0,
                "t1": t1,
                "N": int(len(sub)),
                "MedianX_mm": float(sub["X_mm"].median()) if len(sub) else np.nan,
                "MedianY_mm": float(sub["Y_mm"].median()) if len(sub) else np.nan,
            })

    cluster_df = pd.DataFrame(cluster_rows)
    stage_df = pd.DataFrame(stage_rows)
    return cluster_df, stage_df


# =============================================================================
# 绘图主函数
# =============================================================================

def plot_spatial_4maps(catalog_paths, output_dir, output_basename="figure_spatial_4maps_2x2"):
    setup_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df = read_all_catalogs(catalog_paths, SLIP_NAMES)
    hc_df = filter_high_confidence(all_df)

    if len(hc_df) == 0:
        raise ValueError("筛选后没有高置信度事件，请检查数据或阈值设置。")

    cluster_df, stage_df = compute_cluster_and_stage_info(hc_df, SLIP_NAMES)

    # 与原图一致：对灰度色图进行截断加深
    base_greys = plt.cm.get_cmap("Greys")
    greys_darker = LinearSegmentedColormap.from_list(
        "Greys_trunc_025_100",
        base_greys(np.linspace(0.25, 1.0, 256))
    )
    conf_norm = Normalize(vmin=0.6, vmax=1.0)

    # 与原图一致的文字偏移
    label_offsets = {
        "Slip-1": [(10, 10), (10, 0), (10, 8), (-75, -10), (8, 0)],
        "Slip-2": [(10, 8), (10, -5), (10, 8), (10, -2), (10, -4)],
        "Slip-3": [(10, 8), (10, -4), (10, 6), (10, -8), (10, -6)],
        "Slip-4": [(10, 8), (10, -5), (10, 6), (10, -5), (10, 0)],
    }

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.15), constrained_layout=True, facecolor="white")
    axes = axes.ravel()

    xline = np.linspace(XY_MIN, XY_MAX, 300)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    last_sc = None

    for i, (ax, slip_name) in enumerate(zip(axes, SLIP_NAMES)):
        df = hc_df[hc_df["Catalog"] == slip_name].copy()
        local_clusters = cluster_df[cluster_df["Catalog"] == slip_name] if len(cluster_df) else pd.DataFrame()
        local_stages = stage_df[stage_df["Catalog"] == slip_name].dropna(subset=["MedianX_mm", "MedianY_mm"]).reset_index(drop=True)

        # 散点定位结果
        last_sc = ax.scatter(
            df["X_mm"], df["Y_mm"],
            c=df["LocateConf"], cmap=greys_darker, norm=conf_norm,
            s=10.0, alpha=0.72, linewidths=0, zorder=2
        )

        # 断层迹线 y = -x
        ax.plot(xline, -xline, ls="--", lw=0.9, color="0.45", zorder=1)

        # DBSCAN 聚类圈
        for _, row in local_clusters.iterrows():
            ax.add_patch(plt.Circle(
                (row["CenterX_mm"], row["CenterY_mm"]), row["Radius_mm"],
                fill=False, ec="red", lw=1.05, zorder=3
            ))

        # 分阶段中位位置点
        for j, row in local_stages.iterrows():
            ax.scatter(
                row["MedianX_mm"], row["MedianY_mm"],
                s=34, facecolor=STAGE_COLORS[j], edgecolor="black", linewidth=0.45, zorder=6
            )

        # 阶段迁移箭头
        for j in range(len(local_stages) - 1):
            x0, y0 = local_stages.loc[j, ["MedianX_mm", "MedianY_mm"]]
            x1, y1 = local_stages.loc[j + 1, ["MedianX_mm", "MedianY_mm"]]
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
                zorder=5
            )

        # 阶段标签
        offsets = label_offsets.get(slip_name, [(8, 8)] * len(local_stages))
        for j, row in local_stages.iterrows():
            dx, dy = offsets[j]
            ax.text(
                row["MedianX_mm"] + dx,
                row["MedianY_mm"] + dy,
                STAGE_LABELS[j],
                fontsize=7.3, color="black", zorder=7
            )

        # 面板号与标题
        ax.text(0.035, 0.055, panel_labels[i], transform=ax.transAxes,
                ha="left", va="bottom", fontsize=14.6)
        ax.text(0.145, 0.055, slip_name, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=12.0)

        # 坐标轴设置
        ax.set_xlim(XY_MIN, XY_MAX)
        ax.set_ylim(XY_MIN, XY_MAX)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([-200, -100, 0, 100, 200])
        ax.set_yticks([-200, -100, 0, 100, 200])
        ax.tick_params(direction="out", length=3.6, width=0.9, pad=2)
        ax.set_xlabel("X (mm)", labelpad=2)
        ax.set_ylabel("Y (mm)", labelpad=2)

    # 公共色棒
    cbar = fig.colorbar(last_sc, ax=axes.tolist(), fraction=0.035, pad=0.025, shrink=0.88)
    cbar.set_label("Locate confidence")
    cbar.set_ticks([0.6, 0.7, 0.8, 0.9, 1.0])

    # 输出
    out_png = output_dir / f"{output_basename}.png"
    out_pdf = output_dir / f"{output_basename}.pdf"
    out_svg = output_dir / f"{output_basename}.svg"

    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("绘图完成。")
    print(f"全部事件数: {len(all_df)}")
    print(f"高置信度事件数: {len(hc_df)}")
    print(f"PNG 输出: {out_png}")
    print(f"PDF 输出: {out_pdf}")
    print(f"SVG 输出: {out_svg}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    plot_spatial_4maps(
        catalog_paths=CATALOG_PATHS,
        output_dir=OUTPUT_DIR,
        output_basename=OUTPUT_BASENAME,
    )
