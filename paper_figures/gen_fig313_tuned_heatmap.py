# gen_fig313_tuned_heatmap.py
# Fig 3.13: Sector-Matching Hypothesis test — after independent tuning.
# Heatmap: 7 regions x 3 columns (Baseline delta=0, Tuned delta-RMSE, Tuned delta-%).
# Green = improvement (negative delta), Red = degradation (positive delta), White = zero.
# Style strictly matches Fig 3.9 (star/fig_ch3_matching_heatmap.svg).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import numpy as np
import os, sys

fm.fontManager.addfont("C:/Windows/Fonts/arial.ttf")
fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["Arial", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import nature_figure_config

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "plots", "paper")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from thesis_data import REGIONS, PHASE4_TUNED


def plot_tuned_heatmap(save_png, save_svg):
    # Row order: Atlantic group, then Pacific, then Core (identical to Fig 3.9)
    row_order = ["barents", "greenland", "kara", "laptev", "bering", "chukchi", "central"]
    region_names = {r["id"]: r["name"] for r in REGIONS}
    region_sectors = {r["id"]: r["sector"] for r in REGIONS}

    row_labels = [region_names[rid] for rid in row_order]
    sector_row = [region_sectors[rid] for rid in row_order]

    # Build raw data matrix: 7 rows x 3 columns
    # Col 0: baseline (Δ=0), Col 1: tuned_rmse - baseline, Col 2: delta_pct
    data_raw = np.zeros((7, 3))
    for i, rid in enumerate(row_order):
        d = PHASE4_TUNED[rid]
        bl = d["baseline_rmse"]
        data_raw[i, 0] = 0.0                        # baseline column always 0
        data_raw[i, 1] = d["tuned_rmse"] - bl       # tuned delta RMSE (M km²)
        data_raw[i, 2] = d["delta_pct"]             # tuned delta %

    col_labels = ["基线 (Δ=0)", "调优后 ΔRMSE", "调优后 Δ%"]

    # ---- Per-column independent normalisation ----
    # Col 0 (baseline) and Col 1 (RMSE delta) share RMSE-scale norm.
    # Col 2 (percentage delta) has its own norm to avoid being dominated.
    vmax_rmse = max(abs(data_raw[:, 1].min()), abs(data_raw[:, 1].max()), 1e-6)
    vmax_pct  = max(abs(data_raw[:, 2].min()), abs(data_raw[:, 2].max()), 1e-6)

    # Display matrix: each column scaled independently to [-1, 1]
    data_display = np.zeros((7, 3))
    data_display[:, 0] = 0.0
    data_display[:, 1] = data_raw[:, 1] / vmax_rmse
    data_display[:, 2] = data_raw[:, 2] / vmax_pct

    # Symmetric colour limits for display
    norm = plt.Normalize(vmin=-1, vmax=1)

    # Custom colormap: green(improve) -> white(neutral) -> red(degrade)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "green_white_red",
        ["#2E9E44", "white", "#E53935"],
        N=256,
    )

    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(data_display, cmap=cmap, norm=norm, aspect="auto")

    # ---- Annotate each cell with raw values ----
    for i in range(7):
        for j in range(3):
            raw_val = data_raw[i, j]
            disp_val = data_display[i, j]  # normalised for text colour decision
            if j == 0:
                text = "0"
            elif j == 1:
                sign = "+" if raw_val > 0 else ""
                text = f"{sign}{raw_val:.4f}"
            else:
                sign = "+" if raw_val > 0 else ""
                text = f"{sign}{raw_val:.1f}%"
            # Text colour: white on dark background (|disp| > 0.5), black on light
            if abs(disp_val) > 0.5:
                text_color = "white"
            else:
                text_color = "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    fontweight="bold", color=text_color)

    # ---- Highlight row(s) ----
    # Kara Sea: best improvement (-9.9%), highlight by green border
    kara_idx = row_order.index("kara")
    rect = mpatches.Rectangle(
        (-0.5, kara_idx - 0.5), 3, 1,
        linewidth=2.5, edgecolor="#2E9E44", facecolor="none",
        linestyle="-",
    )
    ax.add_patch(rect)
    ax.text(
        3.15, kara_idx,
        "最大改善", ha="left", va="center",
        fontsize=8, color="#2E9E44", fontweight="bold",
    )

    # Laptev Sea: only degradation (+3.3%), highlight by red border
    laptev_idx = row_order.index("laptev")
    rect2 = mpatches.Rectangle(
        (-0.5, laptev_idx - 0.5), 3, 1,
        linewidth=2.5, edgecolor="#E53935", facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect2)
    ax.text(
        3.15, laptev_idx,
        "唯一退化", ha="left", va="center",
        fontsize=8, color="#E53935", fontweight="bold",
    )

    # ---- Axes ----
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels(row_labels, fontsize=10)

    # Sector group labels on right side
    for i, rid in enumerate(row_order):
        sec_cn = region_sectors[rid]
        if sec_cn == "Atlantic":
            sec_cn = "大西洋"
        elif sec_cn == "Pacific":
            sec_cn = "太平洋"
        else:
            sec_cn = "核心区"
        ax.text(3.15, i, sec_cn, ha="left", va="center", fontsize=7, color="gray")

    # ---- Colorbar ----
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("归一化改善幅度 (每列独立标准化至 [−1, 1])", fontsize=8)

    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(save_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(save_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Console report
    print("=== Figure 3.13 Console Summary ===")
    print(f"  {'Region':12s} {'Baseline':>10s} {'Tuned_delta':>12s} {'Delta%':>8s}  Improved?")
    print("  " + "-" * 56)
    for i, rid in enumerate(row_order):
        d = PHASE4_TUNED[rid]
        bl = d["baseline_rmse"]
        md = d["tuned_rmse"] - bl
        dp = d["delta_pct"]
        improved = "YES" if dp < 0 else ("DEGRADED" if dp > 0 else "no")
        print(f"  {region_names[rid]:8s}  {bl:.4f}      {md:+.4f}        {dp:+.1f}%       {improved}")
    print(f"\n  Saved: {save_png}")
    print(f"  Saved: {save_svg}")


if __name__ == "__main__":
    plot_tuned_heatmap(
        os.path.join(OUTPUT_DIR, "fig_ch3_tuned_heatmap.png"),
        os.path.join(OUTPUT_DIR, "fig_ch3_tuned_heatmap.svg"),
    )
