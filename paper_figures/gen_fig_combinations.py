
# gen_fig_combinations.py
# 图 3.2: 变量组合排名图 (Variable Combination Ranking Chart)
# Phase 2 core finding: less is more - single or two-variable combos outperform
# the all-variable model. 17 experiments ranked by RMSE (ascending).

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os, sys

fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
fm.fontManager.addfont('C:/Windows/Fonts/arial.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
# Import nature_figure_config BEFORE we set fonts, so our SimHei-first
# ordering takes priority over its Arial-first default.
import nature_figure_config
# Re-apply font ordering after nature_figure_config rcParams update
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
from thesis_data import PAN_ARCTIC_RANKING, PHASE1_SINGLE_VAR, PHASE2_COMBINATIONS

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, 'plots', 'paper')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette - by number of auxiliary variables
VAR_COLORS = {
    0: '#484878',  # baseline_dark  - E1 pure-ice
    1: '#7884B4',  # baseline_mid   - single aux
    2: '#B4C0E4',  # baseline_soft  - two aux
    3: '#F0C0CC',  # ours_large     - three aux
    5: '#E53935',  # delta_down     - five aux (all variables)
}

# Manual n_vars map for Phase-1 entries (Phase-2 entries carry n_vars natively)
_PHASE1_N_VARS = {'E1': 0, 'E10': 1, 'E14': 1, 'E9': 1, 'E8': 1, 'E7v1': 2}

def _n_vars(entry):
    if 'n_vars' in entry:
        return entry['n_vars']
    return _PHASE1_N_VARS.get(entry.get('id', ''), 0)

def plot_combination_ranking(ranking, save_stem):
    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    # Sort by RMSE ascending
    sorted_entries = sorted(ranking, key=lambda x: x['rmse'])
    labels = [e['label'] for e in sorted_entries]
    values = [e['rmse'] for e in sorted_entries]
    nv = [_n_vars(e) for e in sorted_entries]

    colors = []
    for n in nv:
        if n >= 5:
            colors.append(VAR_COLORS[5])
        elif n >= 3:
            colors.append(VAR_COLORS[3])
        else:
            colors.append(VAR_COLORS.get(n, '#999999'))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.68,
                   edgecolor='white', linewidth=0.5, zorder=3)

    # Highlight best (E10) and worst (E24) with prominent borders
    for i, e in enumerate(sorted_entries):
        if e['id'] == 'E10':
            bars[i].set_edgecolor('#1a1a2e')
            bars[i].set_linewidth(2.2)
            bars[i].set_zorder(5)
        if e['id'] == 'E24':
            bars[i].set_edgecolor('#B71C1C')
            bars[i].set_linewidth(1.8)
            bars[i].set_zorder(4)

    # E1 baseline vertical dashed line
    e1_rmse = PHASE1_SINGLE_VAR['E1']['rmse']
    ax.axvline(x=e1_rmse, color='#484878', linestyle='--', linewidth=1.4,
               alpha=0.7)

    # Top-5 zone shading
    ax.axhspan(-0.5, 4.5, facecolor='#2E9E44', alpha=0.055, zorder=0)
    ax.annotate('Top 5\n(最优区域)', xy=(0.99, 0.93), xycoords='axes fraction',
                fontsize=8.5, ha='right', va='top', color='#1B5E20',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='#2E9E44', alpha=0.85, linewidth=1.0))

    # Axes labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('RMSE (百万平方公里)', fontsize=10, labelpad=6)
    ax.invert_yaxis()
    ax.set_xlim(0.504, 0.547)
    ax.tick_params(axis='x', labelsize=8)

    # RMSE value annotations
    for i, (v, e) in enumerate(zip(values, sorted_entries)):
        best = (e['id'] == 'E10')
        ax.text(0.5043, i, f'{v:.4f}', va='center', fontsize=7.2,
                color='#1a1a2e' if best else '#444444',
                fontweight='bold' if best else 'normal')

    # Build legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=VAR_COLORS[0], label='0变量 (纯冰基线)'),
        Patch(facecolor=VAR_COLORS[1], label='1变量'),
        Patch(facecolor=VAR_COLORS[2], label='2变量'),
        Patch(facecolor=VAR_COLORS[3], label='3变量'),
        Patch(facecolor=VAR_COLORS[5], label='5变量 (全五变量)'),
    ]
    # E1 baseline entry
    from matplotlib.lines import Line2D
    legend_elements.append(
        Line2D([0], [0], color='#484878', linestyle='--', linewidth=1.4,
               label=f'E1 纯冰基线 ({e1_rmse:.4f})'))
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.85,
              fontsize=7, ncol=2, columnspacing=0.6, handlelength=1.2)

    # Subfigure tag
    ax.text(0.012, 0.985, '(a)', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='left')

    plt.tight_layout(pad=0.8)
    for ext in ['png', 'svg']:
        fpath = f'{save_stem}.{ext}'
        fig.savefig(fpath, facecolor='white')
        print(f'Saved: {fpath}')
    plt.close()
    print('gen_fig_combinations.py done.')

if __name__ == '__main__':
    save_stem = os.path.join(OUTPUT_DIR, 'fig_ch3_combinations')
    plot_combination_ranking(PAN_ARCTIC_RANKING, save_stem)
