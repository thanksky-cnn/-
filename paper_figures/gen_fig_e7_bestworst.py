"""E7双编码器 — 最好与最坏预测年。"""
import sys, os, json, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
from src.data_preprocessing import load_dual_encoder_data, create_dual_targets
from src.model import SeaIceDualEncoderLSTM
from src.utils import set_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

LAGGED_CSV = os.path.join(config.BASE_DIR, 'data', 'lagged_features.csv')
OUT = os.path.join(config.PLOTS_DIR, "paper")
os.makedirs(OUT, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ol = config.OUTPUT_LEN
MODEL_DIR = config.MODELS_DIR


def main():
    set_seed(config.RANDOM_SEED)

    X_main, X_aux, df_de, scaler_de = load_dual_encoder_data(LAGGED_CSV, config.TARGET_COLUMN)
    aux_sl = 3
    y_de, years_de = create_dual_targets(df_de, 12, ol, aux_sl, config.TARGET_COLUMN)
    n = min(len(X_main), len(y_de))
    X_main, X_aux, y_de, years_de = X_main[:n], X_aux[:n], y_de[:n], years_de[:n]

    model = SeaIceDualEncoderLSTM(1, 128, 32, 7, aux_sl, 1, ol, 0.1, 0.6).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model_de.pth'), map_location=device))
    model.eval()

    # 最好年：2022，使用完整日历年聚合
    df_months = df_de['month'].values
    df_years = df_de['year'].values
    n_total = len(X_main)
    all_target_years = np.array([df_years[12 + i : 12 + i + ol] for i in range(n_total)])
    all_target_months = np.array([df_months[12 + i : 12 + i + ol] for i in range(n_total)])
    years_arr = df_years[12 : 12 + n_total]
    test_mask = (years_arr >= 2016) & (years_arr <= 2025)

    with torch.no_grad():
        yp_all = model(torch.tensor(X_main[test_mask], dtype=torch.float32).to(device),
                       torch.tensor(X_aux[test_mask, -aux_sl:, :], dtype=torch.float32).to(device)).cpu().numpy()
    y_pred_all = scaler_de.inverse_transform(yp_all)
    y_true_all = scaler_de.inverse_transform(y_de[test_mask])
    test_yrs = all_target_years[test_mask]
    test_mos = all_target_months[test_mask]

    # 日历年聚合
    def get_year_data(yr):
        pred_vals, true_vals = {}, {}
        for i in range(len(y_pred_all)):
            for k in range(ol):
                if int(test_yrs[i, k]) == yr:
                    m = int(test_mos[i, k])
                    pred_vals[m] = y_pred_all[i, k]
                    true_vals[m] = y_true_all[i, k]
        if len(pred_vals) == 12:
            ms = sorted(pred_vals.keys())
            return np.array([pred_vals[m] for m in ms]), np.array([true_vals[m] for m in ms])
        return None, None

    pred_2022, true_2022 = get_year_data(2022)
    pred_2016, true_2016 = get_year_data(2016)

    print(f"2022 RMSE={np.sqrt(np.mean((pred_2022-true_2022)**2)):.4f}")
    print(f"2016 RMSE={np.sqrt(np.mean((pred_2016-true_2016)**2)):.4f}")
    for m in range(1, 13):
        print(f"  {m}月: true={true_2016[m-1]:.2f} pred={pred_2016[m-1]:.2f} diff={pred_2016[m-1]-true_2016[m-1]:+.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    months_label = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
    x12 = np.arange(1, 13)
    tags = ['(a)', '(b)']
    years_label = ['2022年', '2016年']

    for i, (ax, pred, true, yr_label) in enumerate(zip(axes,
            [pred_2022, pred_2016], [true_2022, true_2016], years_label)):
        ax.fill_between(x12, true, alpha=0.10, color='#e74c3c')
        ax.plot(x12, true, 'o-', color='#e74c3c', linewidth=2.8, markersize=9, label='实测值')
        ax.plot(x12, pred, 's--', color='#1b7837', linewidth=2.5, markersize=9, label='预测值')
        ax.set_xticks(x12)
        ax.set_xticklabels(months_label, fontsize=11)
        ax.set_xlabel('月份', fontsize=13)
        ax.set_ylabel('海冰面积 (百万平方公里)', fontsize=13)
        ax.legend(fontsize=11, loc='lower left')
        ax.grid(alpha=0.3)
        ax.set_ylim(1, 14)
        ax.text(0.03, 0.97, tags[i], transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='top', ha='left')
        ax.text(0.97, 0.97, yr_label, transform=ax.transAxes, fontsize=10,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9))

    plt.tight_layout()
    save_path = os.path.join(OUT, "fig_e7_best_worst.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
