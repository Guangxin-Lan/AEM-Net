#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Figure S2 plotting script: STA/LTA–AIC vs DNet

How to use on Windows
---------------------
1. Install dependencies once:
       pip install numpy pandas matplotlib
2. Right-click this file and select "Run with Python" (or double-click it).
3. Select, in sequence:
       (1) STA/LTA–AIC result file: associated_event_detections.txt
       (2) DNet event catalog file: e.g., 10events_catalog[...].txt
       (3) Output folder
4. The script automatically writes PNG, PDF, SVG, and CSV result tables.

No command-line arguments are required.

Current fixed analysis settings
-------------------------------
- DNet filter: DetectConf >= 0.95
- One-to-one matching tolerance: +/- 0.5 ms
- Event-count bin width: 0.5 s
- Panel (f): Zero residual = solid line; Median residual = dashed line;
  legend = lower left.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# Analysis settings: only edit these values if you intentionally change them.
# ============================================================================
DNET_CONFIDENCE_THRESHOLD = 0.95
MATCH_TOLERANCE_MS = 0.5
COUNT_BIN_WIDTH_S = 0.5
OUTPUT_PREFIX = "Figure_S2_STA_LTA_AIC_vs_DNet"


# ============================================================================
# User interface
# ============================================================================
def select_paths_with_dialog() -> Tuple[Path | None, Path | None, Path | None]:
    """
    Ask the user to select the two result files and one output folder.

    Returns
    -------
    sta_path, dnet_path, output_dir
        All values are None when the user cancels any selection step.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        sta_file = filedialog.askopenfilename(
            title="Step 1/3: Select STA/LTA–AIC result file",
            filetypes=[
                ("Text/CSV files", "*.txt *.tsv *.csv"),
                ("All files", "*.*"),
            ],
        )
        if not sta_file:
            root.destroy()
            return None, None, None

        dnet_file = filedialog.askopenfilename(
            title="Step 2/3: Select DNet event catalog file",
            filetypes=[
                ("Text/CSV files", "*.txt *.tsv *.csv"),
                ("All files", "*.*"),
            ],
        )
        if not dnet_file:
            root.destroy()
            return None, None, None

        output_folder = filedialog.askdirectory(
            title="Step 3/3: Select output folder for Figure S2"
        )
        root.destroy()

        if not output_folder:
            return None, None, None

        return Path(sta_file), Path(dnet_file), Path(output_folder)

    except Exception as error:
        raise RuntimeError(
            "The file-selection window could not be opened. "
            f"Reason: {error}"
        ) from error


def show_message(title: str, message: str, error: bool = False) -> None:
    """Show a concise popup message; console output remains available as fallback."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if error:
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)

        root.destroy()
    except Exception:
        pass


def pause_before_exit() -> None:
    """Keep a Windows console window open after right-click 'Run with Python'."""
    try:
        input("\nPress Enter to close this window...")
    except (EOFError, KeyboardInterrupt):
        pass


# ============================================================================
# Input/output helpers
# ============================================================================
def read_catalog(file_path: Path) -> pd.DataFrame:
    """Read a tab-, comma-, or whitespace-separated result table."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist:\n{file_path}")

    try:
        dataframe = pd.read_csv(file_path, sep=None, engine="python")
    except Exception as error:
        raise RuntimeError(
            f"Cannot read the result table:\n{file_path}\nReason: {error}"
        ) from error

    if dataframe.empty:
        raise ValueError(f"The input table is empty:\n{file_path}")

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    return dataframe


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    catalog_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(
            f"{catalog_name} is missing required column(s): {missing_columns}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


def as_numeric_and_sort(
    dataframe: pd.DataFrame,
    numeric_columns: Iterable[str],
    time_column: str = "ArrivalTime_s",
) -> pd.DataFrame:
    """Convert requested columns to numeric values and sort by arrival time."""
    output = dataframe.copy()

    for column in numeric_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    output = (
        output.dropna(subset=[time_column])
        .sort_values(time_column)
        .reset_index(drop=True)
    )

    if output.empty:
        raise ValueError("No valid events remain after conversion of ArrivalTime_s.")

    return output


# ============================================================================
# One-to-one time matching
# ============================================================================
def greedy_time_match(
    sta_times_s: np.ndarray,
    dnet_times_s: np.ndarray,
    tolerance_s: float,
) -> Tuple[list[tuple[int, int, float]], Set[int], Set[int]]:
    """
    Match events one-to-one within a symmetric time tolerance.

    Each candidate event pair satisfies:
        |t_DNet - t_STA| <= tolerance_s

    Candidate pairs are ranked by absolute timing difference. The closest
    unused pair is selected first, and each event can be used only once.

    Returns
    -------
    matches
        (STA index, DNet index, signed residual in seconds)
        signed residual = t_DNet - t_STA.
    used_sta, used_dnet
        Index sets of participating events.
    """
    sta_times_s = np.asarray(sta_times_s, dtype=float)
    dnet_times_s = np.asarray(dnet_times_s, dtype=float)

    candidates: list[tuple[float, int, int, float]] = []

    for sta_index, sta_time in enumerate(sta_times_s):
        first_index = np.searchsorted(
            dnet_times_s,
            sta_time - tolerance_s,
            side="left",
        )
        last_index = np.searchsorted(
            dnet_times_s,
            sta_time + tolerance_s,
            side="right",
        )

        for dnet_index in range(first_index, last_index):
            residual_s = dnet_times_s[dnet_index] - sta_time
            candidates.append(
                (abs(residual_s), sta_index, dnet_index, residual_s)
            )

    candidates.sort(key=lambda row: row[0])

    used_sta: Set[int] = set()
    used_dnet: Set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for _, sta_index, dnet_index, residual_s in candidates:
        if sta_index not in used_sta and dnet_index not in used_dnet:
            used_sta.add(sta_index)
            used_dnet.add(dnet_index)
            matches.append((sta_index, dnet_index, residual_s))

    matches.sort(key=lambda row: row[0])
    return matches, used_sta, used_dnet


def make_match_table(
    sta: pd.DataFrame,
    dnet: pd.DataFrame,
    matches: list[tuple[int, int, float]],
) -> pd.DataFrame:
    """Build the table used by panels (d) and (f)."""
    rows = []

    for sta_index, dnet_index, residual_s in matches:
        sta_row = sta.iloc[sta_index]
        dnet_row = dnet.iloc[dnet_index]

        sta_id = (
            int(sta_row["EventID"])
            if "EventID" in sta.columns and pd.notna(sta_row["EventID"])
            else sta_index + 1
        )
        dnet_id = (
            int(dnet_row["EventID"])
            if "EventID" in dnet.columns and pd.notna(dnet_row["EventID"])
            else dnet_index + 1
        )

        rows.append(
            {
                "STA_EventID": sta_id,
                "DNet_EventID": dnet_id,
                "STA_ArrivalTime_s": float(sta_row["ArrivalTime_s"]),
                "DNet_ArrivalTime_s": float(dnet_row["ArrivalTime_s"]),
                "DNet_minus_STA_us": float(residual_s) * 1e6,
                "Abs_Diff_us": abs(float(residual_s)) * 1e6,
            }
        )

    output = pd.DataFrame(rows)

    if not output.empty:
        output = output.sort_values("STA_ArrivalTime_s").reset_index(drop=True)

    return output


# ============================================================================
# Statistics and plotting
# ============================================================================
def calculate_binned_counts(
    sta_times_s: np.ndarray,
    dnet_times_s: np.ndarray,
    bin_width_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate matching event-count series in fixed-width time bins."""
    maximum_time_s = max(float(np.max(sta_times_s)), float(np.max(dnet_times_s)))
    maximum_edge_s = math.ceil(maximum_time_s / bin_width_s) * bin_width_s

    edges = np.arange(
        0.0,
        maximum_edge_s + bin_width_s + 1e-12,
        bin_width_s,
    )
    if len(edges) < 2:
        edges = np.array([0.0, bin_width_s])

    centers = 0.5 * (edges[:-1] + edges[1:])
    sta_counts, _ = np.histogram(sta_times_s, bins=edges)
    dnet_counts, _ = np.histogram(dnet_times_s, bins=edges)

    return centers, sta_counts, dnet_counts


def set_publication_style() -> None:
    """Apply restrained formatting suitable for a JGR/AGU supplementary figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def create_figure(
    sta: pd.DataFrame,
    dnet: pd.DataFrame,
    matched: pd.DataFrame,
    sta_only: pd.DataFrame,
    dnet_only: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Create the final six-panel Figure S2.

    (a) Cumulative event counts
    (b) 0.5 s event-count comparison
    (c) Catalog-size comparison
    (d) Matched-pair arrival-time-difference histogram
    (e) Consecutive inter-event intervals
    (f) Matched-event timing residuals
    """
    if matched.empty:
        raise ValueError(
            "No events were matched within +/- "
            f"{MATCH_TOLERANCE_MS:g} ms. Check the result files or adjust "
            "MATCH_TOLERANCE_MS at the beginning of this script."
        )

    set_publication_style()

    n_sta = len(sta)
    n_dnet = len(dnet)
    n_matched = len(matched)
    n_sta_only = len(sta_only)
    n_dnet_only = len(dnet_only)

    residual_us = matched["DNet_minus_STA_us"].to_numpy(dtype=float)
    median_abs_us = float(matched["Abs_Diff_us"].median())
    q05_us, median_residual_us, q95_us = np.percentile(
        residual_us,
        [5, 50, 95],
    )

    centers, sta_counts, dnet_counts = calculate_binned_counts(
        sta["ArrivalTime_s"].to_numpy(dtype=float),
        dnet["ArrivalTime_s"].to_numpy(dtype=float),
        COUNT_BIN_WIDTH_S,
    )

    pearson_r = (
        float(np.corrcoef(sta_counts, dnet_counts)[0, 1])
        if np.std(sta_counts) > 0 and np.std(dnet_counts) > 0
        else np.nan
    )

    sta_intervals_ms = np.diff(
        sta["ArrivalTime_s"].to_numpy(dtype=float)
    ) * 1e3
    dnet_intervals_ms = np.diff(
        dnet["ArrivalTime_s"].to_numpy(dtype=float)
    ) * 1e3

    maximum_time_s = max(
        float(sta["ArrivalTime_s"].max()),
        float(dnet["ArrivalTime_s"].max()),
    )
    x_limit_s = math.ceil(maximum_time_s)
    tolerance_us = MATCH_TOLERANCE_MS * 1e3

    fig = plt.figure(figsize=(10.5, 11.0))
    grid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        left=0.08,
        right=0.98,
        bottom=0.07,
        top=0.97,
        hspace=0.38,
        wspace=0.28,
    )

    # (a) Cumulative event counts
    axis = fig.add_subplot(grid[0, 0])
    axis.plot(
        sta["ArrivalTime_s"],
        np.arange(1, n_sta + 1),
        linewidth=1.2,
        label="STA/LTA–AIC",
    )
    axis.plot(
        dnet["ArrivalTime_s"],
        np.arange(1, n_dnet + 1),
        linewidth=1.2,
        label="DNet",
    )
    axis.set_xlim(0, x_limit_s)
    axis.set_xlabel("Interseismic time (s)")
    axis.set_ylabel("Cumulative event count")
    axis.set_title("(a) Cumulative event counts", loc="left", fontsize=10)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.tick_params(direction="out", length=3)

    # (b) Event-count comparison
    axis = fig.add_subplot(grid[0, 1])
    axis.plot(
        centers,
        sta_counts,
        marker="o",
        markersize=2.2,
        linewidth=1.0,
        label="STA/LTA–AIC",
    )
    axis.plot(
        centers,
        dnet_counts,
        marker="o",
        markersize=2.2,
        linewidth=1.0,
        label="DNet",
    )
    axis.set_xlim(0, x_limit_s)
    axis.set_xlabel("Interseismic time (s)")
    axis.set_ylabel(f"Event count / {COUNT_BIN_WIDTH_S:g} s")
    axis.set_title("(b) Temporal count comparison", loc="left", fontsize=10)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.text(
        0.98,
        0.94,
        f"r = {pearson_r:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    axis.tick_params(direction="out", length=3)

    # (c) Catalog-size comparison
    axis = fig.add_subplot(grid[1, 0])
    labels = ["STA/LTA–AIC", "DNet", "Matched", "STA-only", "DNet-only"]
    values = [n_sta, n_dnet, n_matched, n_sta_only, n_dnet_only]
    bars = axis.bar(labels, values)

    axis.set_ylabel("Number of events")
    axis.set_title("(c) Catalog-size comparison", loc="left", fontsize=10)
    axis.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.015,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.tick_params(direction="out", length=3)

    # (d) Matched-pair time differences
    axis = fig.add_subplot(grid[1, 1])
    histogram_step_us = max(1.0, tolerance_us / 20.0)
    histogram_bins = np.arange(
        -tolerance_us,
        tolerance_us + histogram_step_us + 1e-12,
        histogram_step_us,
    )
    axis.hist(
        residual_us,
        bins=histogram_bins,
        edgecolor="black",
        linewidth=0.4,
    )
    axis.axvline(0, linestyle="--", linewidth=0.8)
    axis.set_xlim(-tolerance_us, tolerance_us)
    axis.set_xlabel("DNet arrival time − STA/LTA–AIC arrival time (μs)")
    axis.set_ylabel("Matched event pairs")
    axis.set_title("(d) Matched-pair time differences", loc="left", fontsize=10)
    axis.text(
        0.98,
        0.94,
        f"Median |Δt| = {median_abs_us:.1f} μs",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    axis.tick_params(direction="out", length=3)

    # (e) Consecutive inter-event intervals
    axis = fig.add_subplot(grid[2, 0])
    interval_plot_limit_ms = 5.0
    interval_bins_ms = np.linspace(0, interval_plot_limit_ms, 51)

    axis.hist(
        sta_intervals_ms,
        bins=interval_bins_ms,
        histtype="step",
        linewidth=1.2,
        label="STA/LTA–AIC",
    )
    axis.hist(
        dnet_intervals_ms,
        bins=interval_bins_ms,
        histtype="step",
        linewidth=1.2,
        label="DNet",
    )
    axis.set_xlim(0, interval_plot_limit_ms)
    axis.set_xlabel("Consecutive inter-event interval (ms)")
    axis.set_ylabel("Counts")
    axis.set_title("(e) Consecutive inter-event intervals", loc="left", fontsize=10)
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    axis.tick_params(direction="out", length=3)

    # (f) Timing residuals
    axis = fig.add_subplot(grid[2, 1])
    axis.scatter(
        matched["STA_ArrivalTime_s"],
        residual_us,
        s=10,
        label="Matched event pair",
    )
    # Zero residual is a solid line.
    axis.axhline(
        0,
        linestyle="-",
        linewidth=0.8,
        label="Zero residual",
    )
    # Median residual is a dashed line.
    axis.axhline(
        median_residual_us,
        linestyle="--",
        linewidth=0.9,
        label="Median residual",
    )
    axis.fill_between(
        [0, x_limit_s],
        q05_us,
        q95_us,
        alpha=0.18,
        label="5%–95% residual range",
    )
    y_limit_us = max(
        tolerance_us * 1.10,
        float(np.nanmax(np.abs(residual_us))) * 1.10,
    )
    axis.set_xlim(0, x_limit_s)
    axis.set_ylim(-y_limit_us, y_limit_us)
    axis.set_xlabel("STA/LTA–AIC arrival time (s)")
    axis.set_ylabel("DNet − STA/LTA–AIC (μs)")
    axis.set_title("(f) Timing residuals of matched events", loc="left", fontsize=10)
    axis.legend(
        frameon=True,
        fontsize=7.5,
        loc="lower left",
        borderpad=0.45,
        handlelength=1.5,
        labelspacing=0.35,
    )
    axis.tick_params(direction="out", length=3)

    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{OUTPUT_PREFIX}.png"
    pdf_path = output_dir / f"{OUTPUT_PREFIX}.pdf"
    svg_path = output_dir / f"{OUTPUT_PREFIX}.svg"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.csv"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        [
            ["STA/LTA–AIC associated events", n_sta],
            ["DNet events after DetectConf threshold", n_dnet],
            [f"Matched events (±{MATCH_TOLERANCE_MS:g} ms)", n_matched],
            ["STA-only events", n_sta_only],
            ["DNet-only events", n_dnet_only],
            [f"Pearson r of {COUNT_BIN_WIDTH_S:g} s event counts", pearson_r],
            ["Median absolute timing difference (μs)", median_abs_us],
            ["Median signed DNet − STA residual (μs)", median_residual_us],
            ["5th percentile signed residual (μs)", q05_us],
            ["95th percentile signed residual (μs)", q95_us],
        ],
        columns=["Metric", "Value"],
    )
    summary.to_csv(summary_path, index=False)

    return {
        "png": png_path,
        "pdf": pdf_path,
        "svg": svg_path,
        "summary": summary_path,
    }


# ============================================================================
# Main workflow
# ============================================================================
def main() -> None:
    print("=" * 72)
    print("Figure S2: STA/LTA–AIC vs DNet comparison")
    print("No command line is required.")
    print("=" * 72)

    sta_path, dnet_path, output_dir = select_paths_with_dialog()

    if sta_path is None or dnet_path is None or output_dir is None:
        print("[INFO] No file or output folder selected. Program cancelled.")
        pause_before_exit()
        return

    print(f"[INFO] STA/LTA–AIC file: {sta_path}")
    print(f"[INFO] DNet file: {dnet_path}")
    print(f"[INFO] Output folder: {output_dir}")
    print(f"[INFO] DNet threshold: DetectConf >= {DNET_CONFIDENCE_THRESHOLD}")
    print(f"[INFO] Matching tolerance: +/- {MATCH_TOLERANCE_MS} ms")

    sta_raw = read_catalog(sta_path)
    dnet_raw = read_catalog(dnet_path)

    validate_required_columns(
        sta_raw,
        required_columns=["ArrivalTime_s"],
        catalog_name="STA/LTA–AIC result file",
    )
    validate_required_columns(
        dnet_raw,
        required_columns=["ArrivalTime_s", "DetectConf"],
        catalog_name="DNet result file",
    )

    sta = as_numeric_and_sort(
        sta_raw,
        numeric_columns=["ArrivalTime_s", "EventID", "HitCount", "ArrivalSpan_s"],
    )
    dnet_all = as_numeric_and_sort(
        dnet_raw,
        numeric_columns=[
            "ArrivalTime_s",
            "EventID",
            "DetectConf",
            "LocateConf",
            "HitCount",
        ],
    )

    dnet = dnet_all.loc[
        dnet_all["DetectConf"] >= DNET_CONFIDENCE_THRESHOLD
    ].copy().reset_index(drop=True)

    if dnet.empty:
        raise ValueError(
            "No DNet events remain after applying "
            f"DetectConf >= {DNET_CONFIDENCE_THRESHOLD}."
        )

    matches, used_sta, used_dnet = greedy_time_match(
        sta["ArrivalTime_s"].to_numpy(dtype=float),
        dnet["ArrivalTime_s"].to_numpy(dtype=float),
        tolerance_s=MATCH_TOLERANCE_MS * 1e-3,
    )

    matched = make_match_table(sta, dnet, matches)
    sta_only = sta.drop(index=list(used_sta)).reset_index(drop=True)
    dnet_only = dnet.drop(index=list(used_dnet)).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    matched_path = output_dir / f"{OUTPUT_PREFIX}_matched_pairs.csv"
    sta_only_path = output_dir / f"{OUTPUT_PREFIX}_STA_only.csv"
    dnet_only_path = output_dir / f"{OUTPUT_PREFIX}_DNet_only.csv"

    matched.to_csv(matched_path, index=False, float_format="%.9f")
    sta_only.to_csv(sta_only_path, index=False, float_format="%.9f")
    dnet_only.to_csv(dnet_only_path, index=False, float_format="%.9f")

    output_files = create_figure(
        sta=sta,
        dnet=dnet,
        matched=matched,
        sta_only=sta_only,
        dnet_only=dnet_only,
        output_dir=output_dir,
    )

    summary_message = (
        "Processing completed successfully.\n\n"
        f"STA/LTA–AIC events: {len(sta)}\n"
        f"DNet events: {len(dnet)}\n"
        f"Matched events: {len(matched)}\n\n"
        "Generated files:\n"
        f"{output_files['png'].name}\n"
        f"{output_files['pdf'].name}\n"
        f"{output_files['svg'].name}\n"
        f"{output_files['summary'].name}\n"
        f"{matched_path.name}\n"
        f"{sta_only_path.name}\n"
        f"{dnet_only_path.name}"
    )

    print("\n" + summary_message)
    show_message("Figure S2 completed", summary_message, error=False)
    pause_before_exit()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        error_message = f"Processing failed:\n{error}"
        print(f"\n[ERROR] {error_message}", file=sys.stderr)
        show_message("Figure S2 failed", error_message, error=True)
        pause_before_exit()
        sys.exit(1)
