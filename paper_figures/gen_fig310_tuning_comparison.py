# gen_fig310_tuning_comparison.py
# Fig 3.10: Independent tuning before/after — paired bar chart.
# 7 regions x 2 groups (unified params / independently tuned).
# Green bars = improvement; Red bars = no improvement / degradation.
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

from thesis_data import REGIONS, PHASE4_UNTUNED, PHASE4_TUNED

# Colour map
SECTOR_COLOR = {
    "Pacific":  "#7884B4",
    "Atlantic": "#F0C0CC",
    "Core":     "#D8D8D8",
}
IMPROVE_GREEN = "#2E9E44"
DEGRADE_RED   = "#E53935"


def plot_tuning_comparison(save_png, save_svg):
    # Sort by delta_pct (most improved first)
    region_ids = [r["id"] for r in REGIONS]
    tuned_list = [
        (rid, PHASE4_UNTUNED[rid]["baseline_rmse"], PHASE4_TUNED[rid]["tuned_rmse"],
         PHASE4_TUNED[rid]["delta_pct"], REGIONS[[r["id"] for r in REGIONS].index(rid)])
        for rid in region_ids
    ]
    # Actually let me just build the list properly
    tuned_data = []
    for r in REGIONS:
        rid = r["id"]
        untuned_rmse = PHASE4_UNTUNED[rid]["baseline_rmse"]
        tuned_rmse = PHASE4_TUNED[rid]["tuned_rmse"]
        delta_pct = PHASE4_TUNED[rid]["delta_pct"]
        tuned_data.append({
            "id": rid,
            "name": r["name"],
            "sector": r["sector"],
            "untuned": untuned_rmse,
            "tuned": tuned_rmse,
            "delta_pct": delta_pct,
        })

    # Sort: most negative delta_pct (most improved) first
    tuned_data.sort(key=lambda x: x["delta_pct"])

    names = [d["name"] for d in tuned_data]
    untuned_vals = [d["untuned"] for d in tuned_data]
    tuned_vals = [d["tuned"] for d in tuned_data]
    delta_pcts = [d["delta_pct"] for d in tuned_data]
    sectors = [d["sector"] for d in tuned_data]

    # Bar colours: gradient by improvement/degradation magnitude
    def gradient_tune(dp):
        if dp < 0:
            intensity = min(abs(dp) / 10.0, 1.0)  # normalize to ~10%
            r = int(0x2E + intensity * 0x10)
            g = int(0x9E - intensity * 0x40)
            b = int(0x44 - intensity * 0x24)
            return f'#{min(r,255):02x}{max(g,0):02x}{max(b,0):02x}'
        else:
            intensity = min(abs(dp) / 5.0, 1.0)
            r = 0xE5
            g = int(0x39 - intensity * 0x30)
            b = int(0x35 - intensity * 0x28)
            return f'#{r:02x}{max(g,0):02x}{max(b,0):02x}'
    bar_colors = [gradient_tune(dp) for dp in delta_pcts]

    # Sector-based edge colour for Pacific / Atlantic grouping
    edge_colors_map = {
        "Pacific": "#5B6EA5",
        "Atlantic": "#D4A0AA",
        "Core": "#B0B0B0",
    }

    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(names))
    bar_width = 0.35

    # Unified-param bars (lighter fill, the 'before')
    bars1 = ax.bar(
        x - bar_width / 2, untuned_vals, bar_width,
        label="统一参数", color="#CCCCCC", edgecolor="#999999", linewidth=0.8,
    )
    # Tuned bars (coloured by improvement direction)
    bars2 = ax.bar(
        x + bar_width / 2, tuned_vals, bar_width,
        label="独立调优", color=bar_colors, edgecolor="white", linewidth=0.5,
    )

    # ---- Value labels on bars ----
    for xi, (u, t) in enumerate(zip(untuned_vals, tuned_vals)):
        ax.text(xi - bar_width / 2, u + 0.002, f"{u:.4f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(xi + bar_width / 2, t + 0.002, f"{t:.4f}", ha="center", va="bottom", fontsize=7.5)

    # ---- Percentage change annotation between bars ----
    for xi, dp in enumerate(delta_pcts):
        sign = "+" if dp > 0 else ""
        color = DEGRADE_RED if dp > 0 else IMPROVE_GREEN
        ax.text(xi, max(untuned_vals[xi], tuned_vals[xi]) + 0.010,
                f"{sign}{dp:.1f}%", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=color)

    # ---- Axes ----
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("均方根误差 (百万平方公里)", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    # Sector colour strip under x-axis labels
    for xi, sec in enumerate(sectors):
        ax.text(xi, -0.012, "—", ha="center", va="top", fontsize=14,
                color=edge_colors_map[sec], fontweight="bold",
                transform=ax.get_xaxis_transform())

    # Sector legend using manual text
    # We add a note below the x-axis
    ax.text(
        0.5, -0.10,
        "——  太平洋扇区          ——  大西洋扇区          ——  核心区",
        transform=ax.transAxes, ha="center", fontsize=8, color="gray",
    )

    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(save_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(save_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Console report
    print("=== Figure 3.10 Console Summary ===")
    print(f"{'Region':12s} {'Unified':>10s} {'Tuned':>10s} {'Delta%':>8s}  {'Sector'}")
    for d in tuned_data:
        print(f"  {d['name']:8s}  {d['untuned']:.4f}      {d['tuned']:.4f}     {d['delta_pct']:+.1f}%    {d['sector']}")
    print(f"  Saved: {save_png}")
    print(f"  Saved: {save_svg}")


if __name__ == "__main__":
    plot_tuning_comparison(
        os.path.join(OUTPUT_DIR, "fig_ch3_tuning_comparison.png"),
        os.path.join(OUTPUT_DIR, "fig_ch3_tuning_comparison.svg"),
    )
