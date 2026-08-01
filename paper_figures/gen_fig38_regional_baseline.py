# gen_fig38_regional_baseline.py
# Fig 3.8: Spatial heterogeneity baseline — 7 Arctic regional seas.
# (a) Absolute RMSE bar chart (sorted low→high)
# (b) Normalized RMSE = RMSE / mean_area x 100 (%)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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

# ---------------------------------------------------------------------------
# Colour map: Pacific sector → blue, Atlantic → orange/pink, Core → grey
# ---------------------------------------------------------------------------
SECTOR_COLOR = {
    "Pacific":  "#7884B4",
    "Atlantic": "#F0C0CC",
    "Core":     "#D8D8D8",
}

SECTOR_CN = {
    "Pacific":  "太平洋扇区",
    "Atlantic": "大西洋扇区",
    "Core":     "核心区",
}


def plot_regional_baseline(save_png, save_svg):
    regions_sorted = sorted(REGIONS, key=lambda r: PHASE4_UNTUNED[r["id"]]["baseline_rmse"])
    names = [r["name"] for r in regions_sorted]
    baselines = [PHASE4_UNTUNED[r["id"]]["baseline_rmse"] for r in regions_sorted]
    mean_areas = [r["mean_area"] for r in regions_sorted]
    norm_rmse = [b / a * 100 for b, a in zip(baselines, mean_areas)]
    sectors = [r["sector"] for r in regions_sorted]
    bar_colors_abs = [SECTOR_COLOR[s] for s in sectors]

    # ---- Normalised RMSE sorted independently ----
    # Build a separate sort order for panel (b)
    norm_data = sorted(
        zip(names, norm_rmse, sectors),
        key=lambda x: x[1]
    )
    names_norm, norm_vals, sectors_norm = zip(*norm_data)
    bar_colors_norm = [SECTOR_COLOR[s] for s in sectors_norm]

    # ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- (a) Absolute RMSE ----
    x1 = np.arange(len(names))
    bars1 = ax1.bar(x1, baselines, color=bar_colors_abs, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(names, fontsize=10, fontfamily='SimHei')
    ax1.set_ylabel("均方根误差 (百万平方公里)", fontsize=10)
    ax1.set_ylim(0, max(baselines) * 1.18)

    # value labels
    for xi, v in zip(x1, baselines):
        ax1.text(xi, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax1.text(0.03, 0.97, "(a)", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top")

    # ---- (b) Normalised RMSE (%) ----
    x2 = np.arange(len(names_norm))
    ax2.bar(x2, norm_vals, color=bar_colors_norm, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(names_norm, fontsize=10, fontfamily='SimHei')
    ax2.set_ylabel("归一化均方根误差 (%)", fontsize=10)
    ax2.set_ylim(0, max(norm_vals) * 1.18)

    for xi, v in zip(x2, norm_vals):
        ax2.text(xi, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax2.text(0.03, 0.97, "(b)", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top")

    # ---- Legend for sector colours ----
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#7884B4", label="太平洋扇区"),
        Patch(facecolor="#F0C0CC", label="大西洋扇区"),
        Patch(facecolor="#D8D8D8", label="核心区"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, frameon=False, fontsize=9)

    # Finalise
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(save_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Console report
    print("=== Figure 3.8 Console Summary ===")
    for r in regions_sorted:
        rid = r["id"]
        b = PHASE4_UNTUNED[rid]["baseline_rmse"]
        nrm = b / r["mean_area"] * 100
        print(f"  {r['name']:8s}  abs-RMSE={b:.4f}  norm={nrm:.1f}%  sector={r['sector']}")
    print(f"  Saved: {save_png}")
    print(f"  Saved: {save_svg}")


if __name__ == "__main__":
    plot_regional_baseline(
        os.path.join(OUTPUT_DIR, "fig_ch3_regional_baseline.png"),
        os.path.join(OUTPUT_DIR, "fig_ch3_regional_baseline.svg"),
    )
