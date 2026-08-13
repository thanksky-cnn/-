# -*- coding: utf-8 -*-
"""LSTM 网络与记忆单元结构示意图（润色版）.

左图：LSTM 按时序展开的网络结构；
右图：LSTM 单元内部的门控、记忆状态与输出路径。
输出：outputs/plots/paper/fig_lstm_structure.png + .svg
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.unicode_minus": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

import nature_figure_config

# 仓库配置以 Arial 开头，本图中文优先，覆盖回 SimHei 在前。
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial", "DejaVu Sans"]

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "plots", "paper"
)
os.makedirs(OUT_DIR, exist_ok=True)

INK = "#27272E"
GRAY = "#6C7280"
SOFT_GRAY = "#D8D8D8"
BLUE = "#7884B4"
DARK_BLUE = "#484878"
LIGHT_BLUE = "#E4E4F0"
PAPER = "#FCFDFE"


def draw_box(ax, cx, cy, w, h, text, fc="white", ec=INK, fs=8, lw=0.9,
             tc=INK, rounded=0.08, zorder=3):
    patch = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounded}",
        fc=fc, ec=ec, lw=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=zorder + 1)


def draw_node(ax, cx, cy, r, text, fc="white", ec=INK, fs=8.5, lw=0.9,
              zorder=4):
    patch = mpatches.Circle((cx, cy), r, fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            zorder=zorder + 1)


def draw_arrow(ax, p1, p2, color=GRAY, lw=0.9, rad=0.0, ls="-", zorder=5):
    patch = mpatches.FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0, zorder=zorder,
    )
    ax.add_patch(patch)


def style_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_unrolled_network(ax):
    ax.set_xlim(-0.3, 10.8)
    ax.set_ylim(-0.2, 5.6)
    ax.set_aspect("equal")
    style_axis(ax)

    ax.text(0.03, 0.97, "(a) LSTM 网络展开结构",
            transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    cell_xs = [2.9, 4.9, 6.9]
    for x in cell_xs:
        draw_box(ax, x, 2.6, 1.45, 1.55, "LSTM", fc=LIGHT_BLUE,
                 ec=BLUE, fs=9, lw=0.9, tc=DARK_BLUE)

    draw_box(ax, 1.0, 2.6, 1.05, 0.7, r"$h_0$",
             fc="white", ec=BLUE, fs=8, zorder=3)
    draw_box(ax, 9.3, 2.6, 1.05, 0.7, r"$h_3$",
             fc="white", ec=BLUE, fs=8, zorder=3)

    draw_arrow(ax, (1.52, 2.6), (2.17, 2.6))
    draw_arrow(ax, (3.62, 2.6), (4.17, 2.6))
    draw_arrow(ax, (5.62, 2.6), (6.17, 2.6))
    draw_arrow(ax, (7.62, 2.6), (8.78, 2.6))

    ax.text(3.9, 3.05, r"$h_1$", ha="center", va="bottom",
            fontsize=7.5, color=DARK_BLUE)
    ax.text(5.9, 3.05, r"$h_2$", ha="center", va="bottom",
            fontsize=7.5, color=DARK_BLUE)

    for i, x in enumerate(cell_xs, start=1):
        draw_arrow(ax, (x, 1.05), (x, 1.82))
        draw_arrow(ax, (x, 3.38), (x, 4.15))
        ax.text(x, 0.75, f"$x_{i}$", ha="center", va="center",
                fontsize=7.5, color=DARK_BLUE)
        ax.text(x, 4.45, f"$y_{i}$", ha="center", va="center",
                fontsize=7.5, color=DARK_BLUE)


def plot_lstm_cell(ax):
    ax.set_xlim(-0.3, 13.0)
    ax.set_ylim(-0.35, 8.85)
    ax.set_aspect("equal")
    style_axis(ax)

    ax.text(0.03, 0.97, "(b) LSTM 单元内部结构",
            transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    outer = mpatches.FancyBboxPatch(
        (0.3, 0.18), 12.45, 8.15,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        fc=PAPER, ec=SOFT_GRAY, lw=0.8, zorder=1,
    )
    ax.add_patch(outer)

    # 外部输入与拼接
    draw_box(ax, 1.1, 6.9, 1.70, 0.75, r"$C_{t-1}$",
             fc="white", ec=BLUE, fs=8, zorder=3)
    draw_box(ax, 1.1, 4.4, 1.70, 0.75, r"$h_{t-1}$",
             fc="white", ec=BLUE, fs=8, zorder=3)
    draw_box(ax, 1.1, 1.6, 1.70, 0.75, r"$x_t$",
             fc="white", ec=BLUE, fs=8, zorder=3)
    draw_box(ax, 2.8, 2.8, 2.10, 0.75, r"拼接 $[h_{t-1}, x_t]$",
             fc=LIGHT_BLUE, ec=BLUE, fs=7, zorder=3)

    # 拼接分支点
    draw_node(ax, 4.35, 2.8, 0.10, "", fc=DARK_BLUE, ec=DARK_BLUE,
              fs=1, zorder=4)

    # 四个门控节点
    gate_y = {"f": 6.5, "i": 4.9, "c": 3.3, "o": 1.7}
    gate_text = {"f": r"$\sigma$", "i": r"$\sigma$",
                 "c": r"$\tanh$", "o": r"$\sigma$"}
    for key, y in gate_y.items():
        draw_box(ax, 5.7, y, 1.80, 0.95, gate_text[key],
                 fc="white", ec=BLUE, fs=8.5, zorder=3)

    ax.text(5.7, 7.22, "遗忘门", ha="center", va="center",
            fontsize=7, color=GRAY)
    ax.text(5.7, 4.35, "输入门", ha="center", va="center",
            fontsize=7, color=GRAY)
    ax.text(5.7, 2.75, "候选记忆", ha="center", va="center",
            fontsize=7, color=GRAY)
    ax.text(5.7, 1.15, "输出门", ha="center", va="center",
            fontsize=7, color=GRAY)

    # 运算节点
    draw_node(ax, 8.0, 6.9, 0.35, r"$\times$", fs=9)
    draw_node(ax, 9.9, 5.4, 0.35, r"$\times$", fs=9)
    draw_node(ax, 11.0, 6.6, 0.35, r"$+$", fs=9)
    draw_node(ax, 10.8, 4.0, 0.48, r"$\tanh$", fs=7.5)
    draw_node(ax, 8.8, 1.9, 0.35, r"$\times$", fs=9)

    # 输入到拼接
    draw_arrow(ax, (1.95, 4.03), (2.18, 3.18))
    draw_arrow(ax, (1.95, 1.98), (2.18, 2.42))
    draw_arrow(ax, (3.85, 2.8), (4.10, 2.8))

    # 拼接分支到四个门
    for key, y in gate_y.items():
        draw_arrow(ax, (4.45, 2.8), (4.80, y), rad=-0.08)

    # 记忆状态更新路径
    draw_arrow(ax, (1.95, 6.9), (7.65, 6.9))
    draw_arrow(ax, (6.60, 6.5), (7.82, 6.68))
    draw_arrow(ax, (8.35, 6.9), (10.65, 6.6))
    draw_arrow(ax, (11.35, 6.6), (11.95, 6.6))
    ax.text(12.25, 6.6, r"$C_t$", ha="left", va="center",
            fontsize=8, color=DARK_BLUE)

    # 输入门与候选记忆
    draw_arrow(ax, (6.60, 4.9), (9.55, 5.4))
    draw_arrow(ax, (6.60, 3.3), (9.90, 5.05))
    draw_arrow(ax, (10.25, 5.4), (10.85, 6.28))
    draw_arrow(ax, (11.00, 6.25), (10.95, 4.48))

    # 隐藏状态输出路径
    draw_arrow(ax, (6.60, 1.7), (8.45, 1.9))
    draw_arrow(ax, (11.28, 4.0), (9.15, 2.25))
    draw_arrow(ax, (9.15, 1.9), (10.20, 0.8))
    draw_box(ax, 10.9, 0.8, 1.20, 0.60, r"$h_t$",
             fc=LIGHT_BLUE, ec=BLUE, fs=8, zorder=3)

    # 更新公式
    ax.text(9.8, 7.55, r"$C_t=f_t\odot C_{t-1}+i_t\odot\tilde C_t$",
            ha="center", va="center", fontsize=7, color=GRAY, zorder=4)
    ax.text(10.2, 0.08, r"$h_t=o_t\odot\tanh(C_t)$",
            ha="center", va="center", fontsize=7, color=GRAY, zorder=4)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.9))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.18)

    plot_unrolled_network(ax1)
    plot_lstm_cell(ax2)

    for fmt in ["png", "svg"]:
        fpath = os.path.join(OUT_DIR, f"fig_lstm_structure.{fmt}")
        fig.savefig(fpath, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {fpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()
