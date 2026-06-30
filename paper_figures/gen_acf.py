"""
自相关函数图 — 展示海冰面积的12个月自相关，解释线性回归为何表现强劲
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

# Load ice data
ice_dir = config.DATA_DIR
files = sorted([f for f in os.listdir(ice_dir) if f.endswith(".csv")])
df_list = []
for f in files:
    month = int(f.split("_")[1])
    df = pd.read_csv(os.path.join(ice_dir, f))
    df.columns = df.columns.str.strip()
    df["month"] = month
    df_list.append(df)
df = pd.concat(df_list, ignore_index=True).sort_values(["year", "month"]).reset_index(drop=True)
df = df[df["area"] > 0]
area = df["area"].values

# Compute ACF (manual, max lag 36 months)
def acf(x, nlags=36):
    x = x - np.mean(x)
    n = len(x)
    result = []
    for lag in range(nlags + 1):
        c = np.sum(x[lag:] * x[:n-lag])
        result.append(c / (n - lag))
    return np.array(result) / result[0]

acf_vals = acf(area, 36)
lags = np.arange(37)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.stem(lags, acf_vals, linefmt="#2166ac", markerfmt="o", basefmt="k-")
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
ax.axhline(y=1.96/np.sqrt(len(area)), color="#d73027", linewidth=0.8, linestyle="--", alpha=0.6)
ax.axhline(y=-1.96/np.sqrt(len(area)), color="#d73027", linewidth=0.8, linestyle="--", alpha=0.6)

# Highlight lag-12
ax.annotate("lag-12: {:.3f}".format(acf_vals[12]),
            xy=(12, acf_vals[12]), xytext=(18, acf_vals[12] + 0.05),
            arrowprops=dict(arrowstyle="->", color="#d73027"),
            fontsize=11, color="#d73027", fontweight="bold")

ax.set_xlabel("滞后月数", fontsize=11)
ax.set_ylabel("自相关系数", fontsize=11)
ax.set_xlim(-0.5, 36.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig_acf.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
print(f"lag-12 ACF: {acf_vals[12]:.4f}")
print(f"lag-24 ACF: {acf_vals[24]:.4f}")
