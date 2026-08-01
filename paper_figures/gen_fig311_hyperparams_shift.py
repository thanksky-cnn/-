# gen_fig311_hyperparams_shift.py
# Fig 3.11: Hyperparameter shift after independent tuning.
# 3 scatter subplots: (a) learning_rate (log scale), (b) aux_seq_len, (c) aux_dropout.
# Each point = one region, labelled with region name. Dashed y=x diagonal.
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

from thesis_data import REGIONS, PHASE4_TUNED, UNIFIED_PARAMS

SECTOR_COLOR = {
    "Pacific":  "#7884B4",
    "Atlantic": "#F0C0CC",
    "Core":     "#D8D8D8",
}


def plot_hyperparams_shift(save_png, save_svg):
    region_ids = [r["id"] for r in REGIONS]
    region_names = {r["id"]: r["name"] for r in REGIONS}
    region_sectors = {r["id"]: r["sector"] for r in REGIONS}

    U = UNIFIED_PARAMS  # shorthand

    # Gather tuned values
    names_list = []
    sectors_list = []
    lr_unified = []
    lr_tuned = []
    seq_unified = []
    seq_tuned = []
    dp_unified = []
    dp_tuned = []

    for rid in region_ids:
        t = PHASE4_TUNED[rid]
        names_list.append(region_names[rid])
        sectors_list.append(region_sectors[rid])
        lr_unified.append(U["lr"])
        lr_tuned.append(t["lr"])
        seq_unified.append(U["aux_seq_len"])
        seq_tuned.append(t["aux_seq_len"])
        dp_unified.append(U["aux_dropout"])
        dp_tuned.append(t["aux_dropout"])

    n = len(names_list)
    # Map sector to marker/colour
    colors = [SECTOR_COLOR[s] for s in sectors_list]
    # Use a slightly darker edge for visibility
    edge_colors = {
        "Pacific": "#5B6EA5",
        "Atlantic": "#D19BA6",
        "Core": "#B0B0B0",
    }
    edges = [edge_colors[s] for s in sectors_list]

    # ----------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.2))

    # ---- (a) learning rate (log scale) ----
    ax1.scatter(lr_unified, lr_tuned, c=colors, edgecolors=edges, s=100, zorder=5, linewidths=0.8)
    # Diagonal
    lr_min = min(min(lr_unified), min(lr_tuned)) * 0.5
    lr_max = max(max(lr_unified), max(lr_tuned)) * 2
    ax1.plot([lr_min, lr_max], [lr_min, lr_max], "k--", linewidth=0.8, alpha=0.7, zorder=1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(lr_min, lr_max)
    ax1.set_ylim(lr_min, lr_max)
    ax1.set_xlabel("统一学习率", fontsize=10)
    ax1.set_ylabel("调优学习率", fontsize=10)

    # Label points
    for i in range(n):
        ax1.annotate(
            names_list[i],
            (lr_unified[i], lr_tuned[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="#333333",
            ha="left",
        )

    ax1.text(0.03, 0.97, "(a)", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top")

    # ---- (b) aux_seq_len ----
    ax2.scatter(seq_unified, seq_tuned, c=colors, edgecolors=edges, s=100, zorder=5, linewidths=0.8)
    seq_all = seq_unified + seq_tuned
    seq_min = min(seq_all) - 1
    seq_max = max(seq_all) + 1
    ax2.plot([seq_min, seq_max], [seq_min, seq_max], "k--", linewidth=0.8, alpha=0.7, zorder=1)
    ax2.set_xlim(seq_min, seq_max)
    ax2.set_ylim(seq_min, seq_max)
    ax2.set_xlabel("统一辅助序列长度", fontsize=10)
    ax2.set_ylabel("调优辅助序列长度", fontsize=10)
    # integer ticks
    from matplotlib.ticker import MaxNLocator
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    for i in range(n):
        ax2.annotate(
            names_list[i],
            (seq_unified[i], seq_tuned[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="#333333",
            ha="left",
        )

    ax2.text(0.03, 0.97, "(b)", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top")

    # ---- (c) aux_dropout ----
    ax3.scatter(dp_unified, dp_tuned, c=colors, edgecolors=edges, s=100, zorder=5, linewidths=0.8)
    dp_all = dp_unified + dp_tuned
    dp_pad = 0.06
    ax3.plot([0 - dp_pad, 1 + dp_pad], [0 - dp_pad, 1 + dp_pad], "k--", linewidth=0.8, alpha=0.7, zorder=1)
    ax3.set_xlim(0 - dp_pad, 1 + dp_pad)
    ax3.set_ylim(0 - dp_pad, 1 + dp_pad)
    ax3.set_xlabel("统一辅助 dropout", fontsize=10)
    ax3.set_ylabel("调优辅助 dropout", fontsize=10)

    for i in range(n):
        ax3.annotate(
            names_list[i],
            (dp_unified[i], dp_tuned[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="#333333",
            ha="left",
        )

    ax3.text(0.03, 0.97, "(c)", transform=ax3.transAxes, fontsize=14, fontweight="bold", va="top")

    # ---- Shared legend for sectors ----
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#7884B4", label="太平洋扇区"),
        Patch(facecolor="#F0C0CC", label="大西洋扇区"),
        Patch(facecolor="#D8D8D8", label="核心区"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, frameon=False, fontsize=9)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(save_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Console report
    print("=== Figure 3.11 Console Summary ===")
    print(f"{'Region':12s} {'LR(uni)':>10s} {'LR(tune)':>10s} {'Seq(uni)':>8s} {'Seq(tune)':>9s} {'DO(uni)':>7s} {'DO(tune)':>8s}")
    for i in range(n):
        print(f"  {names_list[i]:8s}  {lr_unified[i]:.6f}    {lr_tuned[i]:.6f}     {seq_unified[i]}         {seq_tuned[i]}         {dp_unified[i]:.1f}       {dp_tuned[i]:.1f}")
    print(f"  Saved: {save_png}")
    print(f"  Saved: {save_svg}")


if __name__ == "__main__":
    plot_hyperparams_shift(
        os.path.join(OUTPUT_DIR, "fig_ch3_hyperparams_shift.png"),
        os.path.join(OUTPUT_DIR, "fig_ch3_hyperparams_shift.svg"),
    )
