"""
图2.1-2.3: 数据概览 — 海冰面积时间序列(1979-2026)、季节循环、AO/SST时间序列
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import sys, os, numpy as np, pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

OUT_DIR = os.path.join(config.BASE_DIR, "outputs", "plots", "paper")
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
ice_dir = config.DATA_DIR
files = sorted([f for f in os.listdir(ice_dir) if f.endswith(".csv")])
df_list = []
for f in files:
    month = int(f.split("_")[1])
    df = pd.read_csv(os.path.join(ice_dir, f))
    df.columns = df.columns.str.strip()
    df["month"] = month
    df_list.append(df)
df_ice = pd.concat(df_list, ignore_index=True).sort_values(["year", "month"]).reset_index(drop=True)
df_ice = df_ice[df_ice["area"] > 0]

ao_path = os.path.join(config.BASE_DIR, "data", "ao_monthly.csv")
df_ao = pd.read_csv(ao_path)

sst_path = os.path.join(config.BASE_DIR, "data", "arctic_sst_monthly.csv")
df_sst = pd.read_csv(sst_path)

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# (a) Sea ice area time series
ax = axes[0]
years = df_ice["year"].values
months = df_ice["month"].values
time_dec = years + (months - 1) / 12
area = df_ice["area"].values

ax.plot(time_dec, area, color="#2166ac", linewidth=0.6, alpha=0.8)
ax.set_ylabel("海冰面积 (百万平方公里)", fontsize=10)

# Add trend line
mask = ~np.isnan(area)
z = np.polyfit(time_dec[mask], area[mask], 1)
p = np.poly1d(z)
ax.plot(time_dec, p(time_dec), color="#d73027", linewidth=1.5, linestyle="--",
        label="线性趋势 ({:.1f} 万平方公里/十年)".format(abs(z[0]) * 10 * 1e6 / 1e4))
ax.legend(loc="upper right", fontsize=8)
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
ax.grid(True, alpha=0.3)

# (b) Seasonal cycle (monthly climatology)
ax2 = axes[1]
monthly_mean = df_ice.groupby("month")["area"].mean()
monthly_std = df_ice.groupby("month")["area"].std()
months_cal = range(1, 13)
mnames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

ax2.fill_between(months_cal,
                 monthly_mean.values - monthly_std.values,
                 monthly_mean.values + monthly_std.values,
                 color="#2166ac", alpha=0.2)
ax2.plot(months_cal, monthly_mean.values, "o-", color="#2166ac", linewidth=1.5, markersize=6)
ax2.set_xlabel("月份", fontsize=10)
ax2.set_ylabel("海冰面积 (百万平方公里)", fontsize=10)
ax2.set_xticks(months_cal)
ax2.set_xticklabels(mnames)
ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top")
ax2.grid(True, alpha=0.3)

# (c) AO and SST auxiliary variables
ax3 = axes[2]  # twin axes
ax3_twin = ax3.twinx()

# AO
ao_time = df_ao["year"].values + (df_ao["month"].values - 1) / 12
ax3.plot(ao_time, df_ao["ao"].values, color="#2166ac", linewidth=0.5, alpha=0.7, label="AO")
ax3.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
ax3.set_ylabel("北极涛动 (AO) 指数", fontsize=10, color="#2166ac")

# SST
sst_time = df_sst["year"].values + (df_sst["month"].values - 1) / 12
ax3_twin.plot(sst_time, df_sst["sst"].values, color="#d73027", linewidth=0.5, alpha=0.7, label="SST")
ax3_twin.set_ylabel("海表温度 (°C)", fontsize=10, color="#d73027")

# Combine legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
ax3.text(0.02, 0.97, "(c)", transform=ax3.transAxes, fontsize=14, fontweight="bold", va="top")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig_data_overview.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
