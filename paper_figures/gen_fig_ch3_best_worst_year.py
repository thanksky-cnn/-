"""图 3.7：最佳与最差预测年 12 个月轨迹对比.

双面板: (a) 最佳预测年, (b) 最差预测年.
数据基于 E1 LSTM (best_model_area.pth) 在 2016-2025 测试集上的实际预测输出.
最佳/最差年份由逐样本 RMSE 程序化确定，非硬编码.
"""

import sys
import os
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_preprocessing import load_and_merge_data, create_sequences, train_val_test_split_by_year
from src.model import SeaIceLSTM
from src.utils import set_seed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nature_figure_config  # 600 DPI + Arial/SimHei + clean spines

OUT_DIR = os.path.join(config.PLOTS_DIR, 'paper')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OL = config.OUTPUT_LEN  # 12


def main():
    set_seed(config.RANDOM_SEED)

    # ── 1. 加载真实 NSIDC 观测数据 ──
    df, scaler = load_and_merge_data(config.DATA_DIR, config.TARGET_COLUMN)
    data = df['area_scaled'].values
    X, y = create_sequences(data, 12, OL)

    df_months = df['month'].values
    df_years = df['year'].values
    n_total = len(X)

    # 每个样本对应的 12 个目标月份和目标年份
    all_target_months = np.array([df_months[12 + i:12 + i + OL] for i in range(n_total)])
    all_target_years = np.array([df_years[12 + i:12 + i + OL] for i in range(n_total)])

    # 年份划分：test = 2016-2025
    years_arr = df_years[12:12 + n_total]
    test_mask = (years_arr >= 2016) & (years_arr <= 2025)

    (_, _), (_, _), (X_te, y_te) = \
        train_val_test_split_by_year(df, X, y, config.TARGET_COLUMN)

    # ── 2. 加载 E1 LSTM 模型并生成预测 ──
    model = SeaIceLSTM(
        input_size=1,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_len=OL,
        dropout=config.DROPOUT,
    ).to(DEVICE)

    ckpt_path = os.path.join(config.MODELS_DIR, 'best_model_area.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    with torch.no_grad():
        y_pred_raw = model(
            torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()

    y_pred_all = scaler.inverse_transform(y_pred_raw)
    y_true_all = scaler.inverse_transform(y_te)

    test_mos = all_target_months[test_mask]   # (N_test, 12)
    test_yrs = all_target_years[test_mask]    # (N_test, 12)

    # ── 3. 按日历年聚合，计算各年 RMSE ──
    candidate_years = sorted(set(int(test_yrs[i, k])
                                  for i in range(len(y_pred_all))
                                  for k in range(OL)
                                  if 2016 <= int(test_yrs[i, k]) <= 2025))

    year_data = {}
    for yr in candidate_years:
        pred_vals, true_vals = {}, {}
        for i in range(len(y_pred_all)):
            for k in range(OL):
                if int(test_yrs[i, k]) == yr:
                    m = int(test_mos[i, k])
                    pred_vals[m] = y_pred_all[i, k]
                    true_vals[m] = y_true_all[i, k]
        if len(pred_vals) == 12:
            ms = sorted(pred_vals.keys())
            yp_yr = np.array([pred_vals[m] for m in ms])
            yt_yr = np.array([true_vals[m] for m in ms])
            year_data[yr] = {
                'months': ms,
                'pred': yp_yr,
                'true': yt_yr,
                'rmse': np.sqrt(np.mean((yp_yr - yt_yr) ** 2)),
            }

    # ── 4. 程序化确定最佳/最差年份 ──
    sorted_years = sorted(year_data.keys(), key=lambda y: year_data[y]['rmse'])
    best_yr = sorted_years[0]
    worst_yr = sorted_years[-1]

    print(f'Best  year: {best_yr}  RMSE = {year_data[best_yr]["rmse"]:.4f}')
    print(f'Worst year: {worst_yr}  RMSE = {year_data[worst_yr]["rmse"]:.4f}')
    print(f'Ratio (worst / best): {year_data[worst_yr]["rmse"] / year_data[best_yr]["rmse"]:.2f}x')
    print()
    for yr in sorted_years:
        marker = ' ← best' if yr == best_yr else (' ← worst' if yr == worst_yr else '')
        print(f'  {yr}: RMSE = {year_data[yr]["rmse"]:.4f}{marker}')

    # ── 5. 绘图 ──
    COLORS = {
        'obs':  '#E53935',   # 观测值 — 红色
        'pred': '#7884B4',   # 预测值 — 蓝紫 (thesis primary)
        'green': '#2E9E44',  # 最佳年标题
    }

    month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                    '7月', '8月', '9月', '10月', '11月', '12月']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    for ax, yr, tag in [
        (ax1, best_yr, '(a) 最佳预测年'),
        (ax2, worst_yr, '(b) 最差预测年'),
    ]:
        d = year_data[yr]
        months = np.array(d['months'])
        yt = d['true']
        yp = d['pred']

        # 观测值：红色实线 + 圆点
        ax.plot(months, yt, 'o-', color=COLORS['obs'], linewidth=2.0, markersize=7,
                label='观测值', zorder=4)
        # 预测值：蓝紫虚线 + 方块
        ax.plot(months, yp, 's--', color=COLORS['pred'], linewidth=2.0, markersize=7,
                label='E1 预测值', zorder=4)

        rmse_yr = d['rmse']
        title_color = COLORS['green'] if yr == best_yr else COLORS['obs']
        ax.set_title(f'{yr} 年 — RMSE = {rmse_yr:.3f} 百万平方公里',
                     fontsize=11, color=title_color, fontweight='bold', pad=10)

        ax.set_xlabel('月份', fontsize=11)
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels, fontsize=9)
        ax.set_xlim(0.5, 12.5)

        ax.set_ylabel('海冰面积 (百万平方公里)', fontsize=11)
        ax.legend(loc='lower left', fontsize=9, frameon=False)

        ax.text(0.03, 0.97, tag, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top')

        # 标注最大偏差月份
        biases = np.abs(yt - yp)
        max_bias_idx = int(np.argmax(biases))
        ax.annotate(
            f'最大偏差\n({month_labels[max_bias_idx]}, {biases[max_bias_idx]:.2f} M km²)',
            xy=(months[max_bias_idx], (yt[max_bias_idx] + yp[max_bias_idx]) / 2),
            xytext=(months[max_bias_idx] + 2.2, (yt[max_bias_idx] + yp[max_bias_idx]) / 2 + 1.2),
            fontsize=7.5, color=COLORS['obs'],
            arrowprops=dict(arrowstyle='->', color=COLORS['obs'], lw=1.2),
        )

    plt.tight_layout(pad=2.5)

    # ── 6. 输出 PNG + SVG ──
    for fmt in ['png', 'svg']:
        fpath = os.path.join(OUT_DIR, f'fig_ch3_best_worst_year.{fmt}')
        fig.savefig(fpath, dpi=600, bbox_inches='tight', facecolor='white')
        print(f'  Saved: {fpath}')

    plt.close(fig)
    print('\nDone.')


if __name__ == '__main__':
    main()
