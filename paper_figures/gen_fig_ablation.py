"""图 3.3：消融验证 — ΔRMSE 水平条形图.

从全五变量模型(E19, RMSE=0.5243)逐个移除变量。
移除PNA/NAO/SST改善性能，仅移除Nino3.4导致退化。
关键发现：多变量冗余而非信号不足是性能瓶颈。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nature_figure_config
from thesis_data import PHASE3_ABLATION, PHASE3_BASELINE_RMSE

fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["Arial", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots', 'paper')
os.makedirs(OUT_DIR, exist_ok=True)

# Sort by delta (improvement first)
entries = sorted(PHASE3_ABLATION.items(), key=lambda x: x[1]['delta'])
labels = [f"{v['label']}\n(移除 {v['removed']})" for _, v in entries]
deltas = np.array([v['delta'] for _, v in entries])
rmse_vals = np.array([v['rmse'] for _, v in entries])

E19_RMSE = PHASE3_BASELINE_RMSE

# ---- Color: gradient by delta magnitude ----
colors = []
for d in deltas:
    if d < 0:
        # Green gradient: deeper = larger improvement
        intensity = min(abs(d) / 0.012, 1.0)  # normalize to ~0.01
        r = int(0x2E - intensity * 0x1E)  # 0x2E -> 0x10
        g = int(0x9E + intensity * 0x30)  # 0x9E -> 0xCE
        b = int(0x44 - intensity * 0x20)  # 0x44 -> 0x24
        colors.append(f'#{max(r,0):02x}{min(g,255):02x}{max(b,0):02x}')
    else:
        # Red gradient: deeper = larger degradation
        intensity = min(abs(d) / 0.012, 1.0)
        r = int(0xE5)  # stay red
        g = int(0x39 - intensity * 0x30)
        b = int(0x35 - intensity * 0x20)
        colors.append(f'#{r:02x}{max(g,0):02x}{max(b,0):02x}')

fig, ax = plt.subplots(figsize=(9, 4.5))

y_pos = range(len(entries))
bars = ax.barh(y_pos, deltas, color=colors, edgecolor='#272727', linewidth=0.8, height=0.55)

# Zero line (E19 baseline)
ax.axvline(0, color='#272727', linewidth=1.5, zorder=5)

# Shaded background zones
ax.axvspan(-0.012, 0, alpha=0.04, color='#2E9E44', zorder=0)
ax.axvspan(0, 0.012, alpha=0.04, color='#E53935', zorder=0)

# Annotations
for i, (d, rmse, l) in enumerate(zip(deltas, rmse_vals, labels)):
    pct = d / E19_RMSE * 100
    color = colors[i]

    # Delta + percentage inside/outside bar
    if d < 0:
        x_text = d - 0.0004
        ha = 'right'
    else:
        x_text = d + 0.0004
        ha = 'left'
    ax.text(x_text, i, f'Δ={d:+.4f}  ({pct:+.1f}%)   |   RMSE={rmse:.4f}',
            va='center', ha=ha, fontsize=7.5, color='#272727')

# Axis labels
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlabel('相对于 E19 全五变量 (RMSE=0.5243) 的变化 (百万平方公里)', fontsize=9)
ax.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E9E44', alpha=0.7, label='移除后改善 (RMSE ↓)'),
    Patch(facecolor='#E53935', alpha=0.7, label='移除后退化 (RMSE ↑)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=7.5, frameon=False)

# Subfigure tag
ax.text(0.02, 0.97, '(b) 消融验证', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', color='#272727')

# Annotation: key finding
ax.text(0.98, 0.12,
        '← 移除极地大气指数改善\n移除热带海洋信号(Niño3.4)是唯一退化 →',
        transform=ax.transAxes, fontsize=7, color='#666666',
        ha='right', va='bottom', fontstyle='italic')

plt.tight_layout(pad=2)
for fmt in ['png', 'svg']:
    fpath = os.path.join(OUT_DIR, f'fig_ch3_ablation.{fmt}')
    if fmt == 'png':
        fig.savefig(fpath, dpi=600, bbox_inches='tight', facecolor='white')
    else:
        fig.savefig(fpath, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {fpath}')
plt.close(fig)

print('=== 图 3.3 消融验证 ===')
print(f'  E19 基线 RMSE = {E19_RMSE:.4f}')
for _, v in entries:
    pct = v['delta'] / E19_RMSE * 100
    print(f'  {v["label"]:30s}  移除 {v["removed"]:8s}  RMSE={v["rmse"]:.4f}  Δ={v["delta"]:+.4f} ({pct:+.1f}%)')
print('Done.')
