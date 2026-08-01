"""图 3.1：单变量增量 RMSE 条形图 — Phase 1 核心发现.
PNA 是最佳单变量 (-2.81%), E7v1 (AO+SST) 退化 (+3.54%).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nature_figure_config
from thesis_data import PHASE1_SINGLE_VAR

fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["Arial", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots', 'paper')
os.makedirs(OUT_DIR, exist_ok=True)

# Sort by RMSE ascending
items = sorted(PHASE1_SINGLE_VAR.items(), key=lambda x: x[1]['rmse'])
ids = [k for k, v in items]
rmse_vals = np.array([v['rmse'] for k, v in items])
deltas = np.array([v['delta'] for k, v in items])
labels = [v['label'] for k, v in items]
cats = [v['category'] for k, v in items]

E1_RMSE = PHASE1_SINGLE_VAR['E1']['rmse']

# Color mapping
color_map = {
    'baseline': '#484878',     # NMI pastel baseline_dark
    'single': '#2E9E44',       # green = improvement over baseline
    'degradation': '#E53935',  # delta_down
}
bar_colors_rmse = [color_map[c] for c in cats]

# Panel (b): gradient coloring by delta magnitude
def gradient_color(delta):
    """Deeper green for larger improvement, deeper red for larger degradation."""
    if delta < 0:
        intensity = min(abs(delta) / 0.018, 1.0)  # normalize to max improvement ~0.015
        # Green: from light (#7BC89C) to deep (#1B5E20)
        r = int(0x2E + intensity * 0x10)
        g = int(0x9E - intensity * 0x40)
        b = int(0x44 - intensity * 0x24)
        return f'#{min(r,255):02x}{max(g,0):02x}{max(b,0):02x}'
    elif delta > 0:
        intensity = min(abs(delta) / 0.020, 1.0)
        r = int(0xE5)
        g = int(0x39 - intensity * 0x30)
        b = int(0x35 - intensity * 0x28)
        return f'#{r:02x}{max(g,0):02x}{max(b,0):02x}'
    else:
        return '#D8D8D8'

bar_colors_delta = [gradient_color(d) for d in deltas]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ----- (a) Absolute RMSE -----
y_pos = range(len(ids))
bars1 = ax1.barh(y_pos, rmse_vals, color=bar_colors_rmse, edgecolor='#272727', linewidth=0.8, height=0.65)
ax1.axvline(E1_RMSE, color='#484878', linestyle='--', linewidth=1.2, alpha=0.7, label=f'E1 基线 ({E1_RMSE:.4f})')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=8)
ax1.set_xlabel('均方根误差 (百万平方公里)', fontsize=9)
ax1.legend(loc='lower right', fontsize=7, frameon=False)
ax1.invert_yaxis()
ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')

# Annotate RMSE values
for i, (v, c) in enumerate(zip(rmse_vals, cats)):
    color = 'white' if c == 'baseline' else '#272727'
    ax1.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=7.5, color=color)

# ----- (b) Delta RMSE (gradient coloring) -----
ax2.barh(y_pos, deltas, color=bar_colors_delta, edgecolor='#272727', linewidth=0.8, height=0.65)
ax2.axvline(0, color='#272727', linewidth=1.2)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=8)
ax2.set_xlabel('相对于 E1 基线的 RMSE 变化 (百万平方公里)', fontsize=9)
ax2.invert_yaxis()
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')

# Annotate delta values with percentage
for i, d in enumerate(deltas):
    pct = d / E1_RMSE * 100
    x_offset = 0.0003 if d >= 0 else -0.0003
    ha = 'left' if d >= 0 else 'right'
    ax2.text(d + x_offset, i, f'{d:+.4f} ({pct:+.1f}%)', va='center', fontsize=7.5, ha=ha)

# --- legend for (b) ---
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E9E44', label='改善 (RMSE ↓)'),
    Patch(facecolor='#E53935', label='退化 (RMSE ↑)'),
    Patch(facecolor='#D8D8D8', label='无变化'),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=7, frameon=False)

plt.tight_layout(pad=2)
for fmt in ['png', 'svg']:
    fpath = os.path.join(OUT_DIR, f'fig_ch3_single_variable.{fmt}')
    if fmt == 'png':
        fig.savefig(fpath, dpi=600, bbox_inches='tight', facecolor='white')
    else:
        fig.savefig(fpath, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {fpath}')
plt.close(fig)

print('=== 图 3.1 单变量增量 RMSE ===')
for k, v in items:
    pct = v['delta'] / E1_RMSE * 100
    print(f'  {v["label"]:25s}  RMSE={v["rmse"]:.4f}  Δ={v["delta"]:+.4f} ({pct:+.1f}%)')
print('Done.')
