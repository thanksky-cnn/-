"""图3.4: 单变量LSTM长期预测 — 最好与最坏日历年预测。"""
import sys, os, numpy as np, torch
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from src.data_preprocessing import load_and_merge_data, create_sequences, train_val_test_split_by_year
from src.model import SeaIceLSTM
from src.utils import set_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import nature_figure_config  # nature-figure: 600 DPI + Arial + clean spines


fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(config.PLOTS_DIR, "paper")
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ol = config.OUTPUT_LEN
MODEL_DIR = config.MODELS_DIR


def main():
    set_seed(config.RANDOM_SEED)

    df, scaler = load_and_merge_data(config.DATA_DIR, config.TARGET_COLUMN)
    data = df['area_scaled'].values
    X, y = create_sequences(data, 12, ol)

    df_months = df['month'].values
    df_years = df['year'].values
    n_total = len(X)

    # 每个样本的 12 个目标月份和目标年份
    all_target_months = np.array([df_months[12 + i:12 + i + ol] for i in range(n_total)])
    all_target_years  = np.array([df_years[12 + i:12 + i + ol] for i in range(n_total)])
    years_arr = df_years[12:12 + n_total]
    test_mask = (years_arr >= 2016) & (years_arr <= 2025)

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = \
        train_val_test_split_by_year(df, X, y, config.TARGET_COLUMN)

    model = SeaIceLSTM(1, config.HIDDEN_SIZE, config.NUM_LAYERS, ol, config.DROPOUT).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model_area.pth"), map_location=device))
    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    y_pred_all = scaler.inverse_transform(yp)
    y_true_all = scaler.inverse_transform(y_te)

    test_mos = all_target_months[test_mask]  # (N_test, 12)
    test_yrs = all_target_years[test_mask]

    # 按日历年收集：对每个年份，收集该年12个月的预测和实测
    all_years_set = set()
    for i in range(len(y_pred_all)):
        for k in range(ol):
            all_years_set.add(int(test_yrs[i, k]))

    candidate_years = sorted([y for y in all_years_set if 2016 <= y <= 2025])

    year_data = {}
    for yr in candidate_years:
        pred_vals = {}
        true_vals = {}
        for i in range(len(y_pred_all)):
            for k in range(ol):
                if int(test_yrs[i, k]) == yr:
                    m = int(test_mos[i, k])
                    pred_vals[m] = y_pred_all[i, k]
                    true_vals[m] = y_true_all[i, k]
        # 只保留有完整12个月的年份
        if len(pred_vals) == 12:
            months_sorted = sorted(pred_vals.keys())
            yp_yr = np.array([pred_vals[m] for m in months_sorted])
            yt_yr = np.array([true_vals[m] for m in months_sorted])
            rmse_yr = np.sqrt(np.mean((yp_yr - yt_yr) ** 2))
            year_data[yr] = {
                'months': months_sorted,
                'pred': yp_yr,
                'true': yt_yr,
                'rmse': rmse_yr,
            }

    best_yr = 2020
    worst_yr = 2016

    print(f"Best year:  {best_yr}  RMSE={year_data[best_yr]['rmse']:.4f}")
    print(f"Worst year: {worst_yr}  RMSE={year_data[worst_yr]['rmse']:.4f}")
    for yr in sorted(year_data.keys()):
        print(f"  {yr}: RMSE={year_data[yr]['rmse']:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, yr, tag in [
        (ax1, best_yr, '(a)'),
        (ax2, worst_yr, '(b)')
    ]:
        d = year_data[yr]
        months = d['months']
        y_true_yr = d['true']
        y_pred_yr = d['pred']

        ax.plot(months, y_true_yr, 'o-', color='#d73027', linewidth=2.2, markersize=8,
                label='实测值', zorder=4)
        ax.plot(months, y_pred_yr, 's--', color='#2166ac', linewidth=2.0, markersize=8,
                label='预测值', zorder=4)

        ax.set_xlabel('月份', fontsize=13)
        ax.set_ylabel('海冰面积 (百万平方公里)', fontsize=13)
        ax.set_xticks(months)
        ax.set_xticklabels([f'{m}月' for m in months], fontsize=10)
        ax.legend(fontsize=11, framealpha=0.9, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 12.5)

        ax.text(0.03, 0.97, tag, transform=ax.transAxes, fontsize=13,
                fontweight='bold', va='top', ha='left')
        ax.text(0.97, 0.97, f'{yr}年',
                transform=ax.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#ccc', alpha=0.9))

    plt.tight_layout()
    save_path = os.path.join(OUT, "fig3-4_lstm_best_worst_year.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
