# gen_fig_ch2_data_overview.py
# Figure for Chapter 2: Data Overview Panel
# (a) Sea ice area interannual variability (1979-2025)
# (b) Sea ice area seasonal cycle (boxplot + mean line)
# (c) Climate indices time series (AO, NAO, PNA, SST, Nino3.4)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import os, sys

fm.fontManager.addfont("C:/Windows/Fonts/arial.ttf")
fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["Arial", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import nature_figure_config

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "plots", "paper")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = config.BASE_DIR

# =============================================================================
# Colour palette (matching star/ reference figures)
# =============================================================================
BLUE_PURPLE   = "#7884B4"   # main blue-purple
DEEP_BLUE     = "#484878"   # deep blue
LIGHT_PURPLE  = "#B4C0E4"   # light purple
PINK          = "#F0C0CC"   # pink
GREEN         = "#2E9E44"   # improvement green
RED           = "#E53935"   # degradation red
GRAY          = "#D8D8D8"   # gray
LIGHT_GRAY    = "#cccccc"   # light gray
MEDIUM_GRAY   = "#9e9e9e"   # medium gray

# Climate index colours (distinct per index)
INDEX_COLORS = {
    "AO":       "#7884B4",  # blue-purple
    "NAO":      "#484878",  # deep blue
    "PNA":      "#F0C0CC",  # pink
    "SST":      "#E53935",  # red
    "Nino3.4":  "#2E9E44",  # green
}


def load_sie_data():
    """Load all 12 monthly SIE CSV files, return DataFrame with col: year, month, area."""
    frames = []
    for m in range(1, 13):
        fpath = os.path.join(BASE_DIR, "data", "raw", f"N_{m:02d}_extent_v4.0.csv")
        df = pd.read_csv(fpath, skipinitialspace=True)
        df = df[["year", "mo", "area"]].copy()
        df.columns = ["year", "month", "area"]
        frames.append(df)
    sie = pd.concat(frames, ignore_index=True)
    # Replace missing-data sentinel -9999 with NaN and drop
    sie.loc[sie["area"] < -100, "area"] = np.nan
    sie = sie.sort_values(["year", "month"]).reset_index(drop=True)
    # Filter to 1979-2025
    sie = sie[(sie["year"] >= 1979) & (sie["year"] <= 2025)]
    return sie


def load_climate_index(fname, col_name):
    """Load a climate index CSV (year, month, <col_name>)."""
    fpath = os.path.join(BASE_DIR, "data", fname)
    df = pd.read_csv(fpath)
    # standardise column order
    df = df[["year", "month", col_name]].copy()
    df = df[(df["year"] >= 1979) & (df["year"] <= 2025)]
    return df


def plot_data_overview(save_png, save_svg):
    sie = load_sie_data()

    # ---- Build annual aggregates for SIE ----
    sie_annual = sie.groupby("year")["area"].mean().reset_index()
    sie_annual.columns = ["year", "annual_mean_area"]

    # September minimum (month=9) and March maximum (month=3)
    sie_sep = sie[sie["month"] == 9][["year", "area"]].copy()
    sie_sep.columns = ["year", "sep_area"]
    sie_mar = sie[sie["month"] == 3][["year", "area"]].copy()
    sie_mar.columns = ["year", "mar_area"]

    # Linear trend of annual mean
    yrs = sie_annual["year"].values
    ann_mean = sie_annual["annual_mean_area"].values
    coeffs = np.polyfit(yrs, ann_mean, 1)
    trend_line = np.polyval(coeffs, yrs)
    trend_per_decade = coeffs[0] * 10  # change per decade
    print(f"  SIE annual mean trend: {coeffs[0]:.4f} M km2/yr = {trend_per_decade:.4f} M km2/decade")

    # ---- Load climate indices ----
    ao_df  = load_climate_index("ao_monthly.csv", "ao")
    nao_df = load_climate_index("nao_monthly.csv", "nao")
    pna_df = load_climate_index("pna_monthly.csv", "pna")
    nino_df = load_climate_index("nino34_monthly.csv", "nino34")
    sst_df = load_climate_index("arctic_sst_monthly.csv", "sst")

    # ---- Build monthly climatology for boxplots ----
    sie_monthly_groups = [sie[sie["month"] == m]["area"].dropna().values for m in range(1, 13)]

    # =========================================================================
    # Figure layout: 3 panels, stacked vertically
    # =========================================================================
    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.85, 1.0], hspace=0.32)

    ax_a = fig.add_subplot(gs[0])   # (a) SIE interannual
    ax_b = fig.add_subplot(gs[1])   # (b) SIE seasonal cycle
    ax_c = fig.add_subplot(gs[2])   # (c) Climate indices

    # =========================================================================
    # Panel (a): Sea ice area interannual variability
    # =========================================================================
    # Monthly values as light gray thin line
    for yr in range(1979, 2026):
        sy = sie[sie["year"] == yr]
        if len(sy) == 12:
            ax_a.plot(range(1, 13), sy["area"].values,
                      color=LIGHT_GRAY, linewidth=0.4, alpha=0.7)

    # Annual mean as bold blue-purple line
    ax_a.plot(yrs, ann_mean, color=BLUE_PURPLE, linewidth=2.0, label="年均值")

    # Linear trend as dashed red line
    ax_a.plot(yrs, trend_line, color=RED, linewidth=1.5, linestyle="--",
              label=f"线性趋势 ({trend_per_decade:+.2f}/10年)")

    # September minimum points (red)
    ax_a.scatter(sie_sep["year"], sie_sep["sep_area"],
                 color=RED, s=12, zorder=5, label="9月极小值")

    # March maximum points (blue)
    ax_a.scatter(sie_mar["year"], sie_mar["mar_area"],
                 color=DEEP_BLUE, s=12, zorder=5, label="3月极大值")

    ax_a.set_xlim(1978.5, 2025.5)
    ax_a.set_xlabel("年份", fontsize=11)
    ax_a.set_ylabel("海冰面积 (百万平方公里)", fontsize=11)
    ax_a.legend(loc="lower left", frameon=False, fontsize=9, ncol=2)
    ax_a.text(0.03, 0.97, "(a)", transform=ax_a.transAxes,
              fontsize=14, fontweight="bold", va="top")

    # =========================================================================
    # Panel (b): Seasonal cycle boxplots + mean line
    # =========================================================================
    bp = ax_b.boxplot(sie_monthly_groups, positions=range(1, 13),
                       patch_artist=True, widths=0.55,
                       medianprops={"color": "black", "linewidth": 0.8},
                       whiskerprops={"color": MEDIUM_GRAY, "linewidth": 0.6},
                       capprops={"color": MEDIUM_GRAY, "linewidth": 0.6},
                       boxprops={"facecolor": GRAY, "edgecolor": MEDIUM_GRAY,
                                 "linewidth": 0.6},
                       flierprops={"marker": "o", "markerfacecolor": MEDIUM_GRAY,
                                   "markersize": 3, "markeredgewidth": 0.3})

    # Monthly mean line
    monthly_means = [np.nanmean(g) for g in sie_monthly_groups]
    ax_b.plot(range(1, 13), monthly_means, color=DEEP_BLUE, linewidth=2.0,
              marker="o", markersize=5, markerfacecolor=DEEP_BLUE, label="月均值")

    ax_b.set_xticks(range(1, 13))
    ax_b.set_xticklabels([f"{m}月" for m in range(1, 13)], fontsize=10)
    ax_b.set_xlabel("月份", fontsize=11)
    ax_b.set_ylabel("海冰面积 (百万平方公里)", fontsize=11)
    ax_b.legend(loc="upper right", frameon=False, fontsize=9)
    ax_b.text(0.03, 0.97, "(b)", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold", va="top")

    # =========================================================================
    # Panel (c): Climate indices time series (shared x-axis 1979-2025)
    # =========================================================================
    # Compute annual means for each index for cleaner display
    def annual_mean_ts(df, col):
        """Return (years, annual_mean) arrays — only complete years (12 months)."""
        ann = df.groupby("year")[col].agg(["mean", "count"]).reset_index()
        ann.columns = ["year", col, "n"]
        ann = ann[ann["n"] == 12]
        return ann["year"].values, ann[col].values

    # Plot each index with distinct colour
    indices = [
        ("AO", ao_df, "ao", 0),
        ("NAO", nao_df, "nao", 0),
        ("PNA", pna_df, "pna", 0),
        ("SST", sst_df, "sst", 1),    # second y-axis for SST (different scale)
        ("Nino3.4", nino_df, "nino34", 0),
    ]

    lines_main = []
    labels_main = []
    ax_c2 = None  # secondary axis for SST

    for label, df, col, axis_side in indices:
        yrs_i, vals_i = annual_mean_ts(df, col)
        if axis_side == 0:
            l, = ax_c.plot(yrs_i, vals_i, color=INDEX_COLORS[label],
                           linewidth=1.2, alpha=0.85)
            lines_main.append(l)
            labels_main.append(label)
        else:
            # SST on secondary y-axis (different scale)
            if ax_c2 is None:
                ax_c2 = ax_c.twinx()
            l, = ax_c2.plot(yrs_i, vals_i, color=INDEX_COLORS[label],
                            linewidth=1.2, linestyle="--", alpha=0.85)
            lines_main.append(l)
            labels_main.append(label)

    # Zero reference line
    ax_c.axhline(y=0, color="black", linewidth=0.4, linestyle="-", alpha=0.3)

    ax_c.set_xlim(1978.5, 2025.5)
    ax_c.set_xlabel("年份", fontsize=11)
    ax_c.set_ylabel("气候指数 (标准化)", fontsize=11)
    if ax_c2:
        ax_c2.set_ylabel("SST (摄氏度)", fontsize=11, color=RED)

    # Combined legend
    ax_c.legend(lines_main, labels_main, loc="upper left", frameon=False,
                fontsize=9, ncol=3)
    ax_c.text(0.03, 0.97, "(c)", transform=ax_c.transAxes,
              fontsize=14, fontweight="bold", va="top")

    # =========================================================================
    # Save
    # =========================================================================
    fig.savefig(save_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(save_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("=== Figure: Data Overview Panel ===")
    print(f"  SIE annual mean range: {ann_mean.min():.2f} - {ann_mean.max():.2f} M km2")
    print(f"  SIE trend per decade: {trend_per_decade:+.4f} M km2")
    print(f"  September mean (1979-2025): {sie_sep['sep_area'].mean():.2f} M km2")
    print(f"  March mean (1979-2025): {sie_mar['mar_area'].mean():.2f} M km2")
    for label, df, col, _ in indices:
        _, vals = annual_mean_ts(df, col)
        print(f"  {label}: annual mean range [{vals.min():.3f}, {vals.max():.3f}]")
    print(f"  Saved: {save_png}")
    print(f"  Saved: {save_svg}")


if __name__ == "__main__":
    plot_data_overview(
        os.path.join(OUTPUT_DIR, "fig_ch2_data_overview.png"),
        os.path.join(OUTPUT_DIR, "fig_ch2_data_overview.svg"),
    )
