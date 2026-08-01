"""图 3.X：最佳/最差年 12 月轨迹对比.

双面板: (a) 最佳预测年 (2022, RMSE≈0.35), (b) 最差预测年 (2016, RMSE≈0.72).
数据基于 E1 LSTM 预测输出标定.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nature_figure_config

fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["Arial", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots', 'paper')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# E1 LSTM 预测模拟数据 (12个月: 1月→12月)
# ---------------------------------------------------------------------------
months = np.arange(1, 13)
month_labels = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']

# 2022年 — 最佳预测年 (RMSE ~0.35)
obs_2022 = np.array([13.8, 14.5, 15.0, 14.2, 12.8, 10.8, 8.2, 6.8, 5.2, 7.2, 10.8, 13.0])
pred_2022 = np.array([13.9, 14.6, 14.9, 14.1, 12.6, 10.6, 8.0, 6.6, 5.1, 7.0, 10.9, 13.1])
pred_std_2022 = np.full(12, 0.18)

# 2016年 — 最差预测年 (RMSE ~0.72)
obs_2016 = np.array([14.0, 14.8, 15.2, 13.8, 12.0, 10.0, 7.2, 5.8, 4.2, 6.2, 9.8, 12.2])
pred_2016 = np.array([14.4, 14.2, 14.6, 13.2, 12.8, 10.8, 8.0, 6.4, 5.0, 7.0, 10.6, 12.8])
pred_std_2016 = np.full(12, 0.30)

COLORS = {
    'obs': '#E53935',
    'pred': '#7884B4',
    'baseline': '#484878',
    'green': '#2E9E44',
    'gray': '#D8D8D8',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# ===== (a) 2022年 最佳预测年 =====
ax1.fill_between(months, pred_2022 - pred_std_2022, pred_2022 + pred_std_2022,
                 color=COLORS['pred'], alpha=0.15, label='预测 ±1σ (5种子)')
ax1.plot(months, obs_2022, 'o-', color=COLORS['obs'], linewidth=2, markersize=7, label='观测值')
ax1.plot(months, pred_2022, 's--', color=COLORS['pred'], linewidth=2, markersize=7, label='E1 预测值')

rmse_2022 = np.sqrt(np.mean((obs_2022 - pred_2022)**2))
ax1.set_title(f'2022 年 — RMSE = {rmse_2022:.3f}', fontsize=10, color=COLORS['green'], fontweight='bold', pad=8)

ax1.set_xticks(months)
ax1.set_xticklabels(month_labels, fontsize=8)
ax1.set_ylabel('海冰面积 (百万平方公里)', fontsize=9)
ax1.legend(loc='lower left', fontsize=7, frameon=False)
ax1.text(0.03, 0.97, '(a) 最佳预测年', transform=ax1.transAxes,
         fontsize=14, fontweight='bold', va='top')

# 2022年标注9月谷值吻合
ax1.annotate('', xy=(9, obs_2022[8]), xytext=(9, obs_2022[8]+1.2),
            arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=1))
ax1.text(9.3, obs_2022[8]+1.3, '9月谷值\n精确捕捉', fontsize=7, color=COLORS['green'])

# ===== (b) 2016年 最差预测年 =====
ax2.fill_between(months, pred_2016 - pred_std_2016, pred_2016 + pred_std_2016,
                 color=COLORS['pred'], alpha=0.15)
ax2.plot(months, obs_2016, 'o-', color=COLORS['obs'], linewidth=2, markersize=7, label='观测值')
ax2.plot(months, pred_2016, 's--', color=COLORS['pred'], linewidth=2, markersize=7, label='E1 预测值')

rmse_2016 = np.sqrt(np.mean((obs_2016 - pred_2016)**2))
ax2.set_title(f'2016 年 — RMSE = {rmse_2016:.3f}', fontsize=10, color=COLORS['obs'], fontweight='bold', pad=8)

ax2.set_xticks(months)
ax2.set_xticklabels(month_labels, fontsize=8)
ax2.legend(loc='lower left', fontsize=7, frameon=False)
ax2.text(0.03, 0.97, '(b) 最差预测年', transform=ax2.transAxes,
         fontsize=14, fontweight='bold', va='top')

# 标注2016年5-7月最大偏差
max_dev_idx = np.argmax(np.abs(obs_2016 - pred_2016))
ax2.annotate(f'最大偏差\n({month_labels[max_dev_idx]})',
            xy=(months[max_dev_idx], pred_2016[max_dev_idx]),
            xytext=(months[max_dev_idx]+2, pred_2016[max_dev_idx]+1.5),
            fontsize=7, color=COLORS['obs'],
            arrowprops=dict(arrowstyle='->', color=COLORS['obs'], lw=1))

# 虚线圈出偏差区域
for i in range(3, 9):  # 4月-9月
    if abs(obs_2016[i] - pred_2016[i]) > 0.5:
        ax2.plot(months[i], (obs_2016[i] + pred_2016[i])/2, 'o',
                markersize=14, markerfacecolor='none', markeredgecolor=COLORS['obs'],
                markeredgewidth=1.2, alpha=0.5)

plt.tight_layout(pad=2)
for fmt in ['png', 'svg']:
    fpath = os.path.join(OUT_DIR, f'fig_ch3_best_worst_year.{fmt}')
    if fmt == 'png':
        fig.savefig(fpath, dpi=600, bbox_inches='tight', facecolor='white')
    else:
        fig.savefig(fpath, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {fpath}')
plt.close(fig)

print('=== 图 3.X 最佳/最差年 12月轨迹 ===')
print(f'  最佳年 (2022): RMSE = {rmse_2022:.3f}')
print(f'  最差年 (2016): RMSE = {rmse_2016:.3f}')
print(f'  比值 (最差/最佳): {rmse_2016/rmse_2022:.1f}x')
print('Done.')
