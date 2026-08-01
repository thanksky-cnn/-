# gen_fig39_matching_heatmap.py
# Fig 3.9: Sector-Matching Hypothesis test.
# Heatmap: 7 regions x 3 columns (Baseline delta=0, Match delta-RMSE, Mismatch delta-RMSE).
# Green = improvement (negative delta), Red = degradation (positive delta), White = zero.
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

from thesis_data import REGIONS, PHASE4_UNTUNED

SECTOR_LABEL = ["大西洋扇区", "大西洋扇区", "大西洋扇区", "大西洋扇区", "太平洋扇区", "太平洋扇区", "核心区"]


def plot_matching_heatmap(save_png, save_svg):
    # Build data matrix: rows = regions (sorted by sector then name),
    #                    cols = [baseline_delta, match_delta, mismatch_delta]
    #
    # We keep a fixed row order: Atlantic group, then Pacific, then Core
    row_order = ["barents", "greenland", "kara", "laptev", "bering", "chukchi", "central"]
    region_names = {r["id"]: r["name"] for r in REGIONS}
    region_sectors = {r["id"]: r["sector"] for r in REGIONS}
    match_aux_map = {r["id"]: r["match_idx"] for r in REGIONS}

    row_labels = [region_names[rid] for rid in row_order]
    sector_row = [region_sectors[rid] for rid in row_order]

    data = np.zeros((7, 3))
    for i, rid in enumerate(row_order):
        d = PHASE4_UNTUNED[rid]
        bl = d["baseline_rmse"]
        data[i, 0] = 0.0  # baseline column always 0
        data[i, 1] = d["match_rmse"] - bl      # match delta
        data[i, 2] = d["mismatch_rmse"] - bl   # mismatch delta

    col_labels = ["基线 (Δ=0)", f"匹配 (ΔRMSE)", f"非匹配 (ΔRMSE)"]

    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine colour limits symmetric around 0
    vmax = max(abs(data.min()), abs(data.max()))
    # Normalize so 0 maps to white
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    # Custom colormap: green(improve) → white(neutral) → red(degrade)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "green_white_red",
        ["#2E9E44", "white", "#E53935"],
        N=256,
    )

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # ---- Annotate each cell ----
    for i in range(7):
        for j in range(3):
            val = data[i, j]
            if j == 0:
                text = "0"
            else:
                sign = "+" if val > 0 else ""
                text = f"{sign}{val:.4f}"
            # Determine text colour: black on light bg, white on dark bg
            if abs(val) > vmax * 0.4:
                text_color = "white"
            else:
                text_color = "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    fontweight="bold", color=text_color)

    # ---- Barents row highlight (only region supporting hypothesis) ----
    barents_idx = row_order.index("barents")
    rect = mpatches.Rectangle(
        (-0.5, barents_idx - 0.5), 3, 1,
        linewidth=2.5, edgecolor="#E53935", facecolor="none",
        linestyle="-",
    )
    ax.add_patch(rect)
    # annotate
    ax.text(
        3.15, barents_idx,
        "唯一支持", ha="left", va="center",
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

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ΔRMSE (百万平方公里)", fontsize=9)

    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(save_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(save_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Console report
    print("=== Figure 3.9 Console Summary ===")
    print(f"{'Region':12s} {'Baseline':>10s} {'Match_delta':>12s} {'Mismatch_delta':>14s}  Matched?")
    for rid in row_order:
        d = PHASE4_UNTUNED[rid]
        bl = d["baseline_rmse"]
        md = d["match_rmse"] - bl
        mmd = d["mismatch_rmse"] - bl
        # Matched if match improves more than mismatch
        matched = "YES" if (md < mmd and md < 0) else "no"
        print(f"  {region_names[rid]:8s}  {bl:.4f}      {md:+.4f}        {mmd:+.4f}          {matched}")
    print(f"  Saved: {save_png}")
    print(f"  Saved: {save_svg}")


if __name__ == "__main__":
    plot_matching_heatmap(
        os.path.join(OUTPUT_DIR, "fig_ch3_matching_heatmap.png"),
        os.path.join(OUTPUT_DIR, "fig_ch3_matching_heatmap.svg"),
    )
