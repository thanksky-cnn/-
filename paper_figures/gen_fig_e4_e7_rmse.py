"""图4.4: 三变量E4与双编码器E7v1逐月RMSE对比（按日历月）。"""
import sys, os, json
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from src.data_preprocessing import (
    load_trivariate_data, load_dual_encoder_data,
    create_sequences, create_dual_targets, train_val_test_split_by_year
)
from src.model import SeaIceLSTM, SeaIceDualEncoderLSTM
from src.utils import set_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import nature_figure_config  # nature-figure: 600 DPI + Arial + clean spines


fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

AO_CSV = os.path.join(config.BASE_DIR, "data", "ao_monthly.csv")
SST_CSV = os.path.join(config.BASE_DIR, "data", "arctic_sst_monthly.csv")
LAGGED_CSV = os.path.join(config.BASE_DIR, "data", "lagged_features.csv")
MODEL_DIR = config.MODELS_DIR
OUT = os.path.join(config.PLOTS_DIR, "paper")
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_monthly_rmse(y_pred, y_true, target_months):
    """按日历月计算 RMSE。"""
    rmse_by_month = {}
    for m in range(1, 13):
        mask = target_months == m
        if mask.sum() > 0:
            rmse_by_month[m] = np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2))
        else:
            rmse_by_month[m] = 0.0
    return np.array([rmse_by_month[m] for m in range(1, 13)])


def get_e7v1_monthly_rmse():
    """E7v1 双编码器逐月 RMSE。"""
    set_seed(config.RANDOM_SEED)
    ol = config.OUTPUT_LEN
    X_main, X_aux, df_de, scaler_de = load_dual_encoder_data(LAGGED_CSV, config.TARGET_COLUMN)
    y_de, years_de = create_dual_targets(df_de, 12, ol, 3, config.TARGET_COLUMN)
    n = min(len(X_main), len(y_de))
    X_main, X_aux, y_de, years_de = X_main[:n], X_aux[:n], y_de[:n], years_de[:n]

    df_months = df_de['month'].values
    df_years = df_de['year'].values
    n_total = len(X_main)
    all_target_months = np.array([df_months[12 + i : 12 + i + ol] for i in range(n_total)])
    years_arr = df_years[12 : 12 + n_total]
    test_mask = (years_arr >= 2016) & (years_arr <= 2025)
    test_target_months = all_target_months[test_mask]

    model = SeaIceDualEncoderLSTM(1, 128, 32, 7, 3, 1, ol, 0.1, 0.6).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model_de.pth"), map_location=device))
    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(X_main[test_mask], dtype=torch.float32).to(device),
                   torch.tensor(X_aux[test_mask, -3:, :], dtype=torch.float32).to(device)).cpu().numpy()
    y_pred = scaler_de.inverse_transform(yp).flatten()
    y_true = scaler_de.inverse_transform(y_de[test_mask]).flatten()
    months_flat = test_target_months.flatten()
    return compute_monthly_rmse(y_pred, y_true, months_flat)


def get_e4_monthly_rmse():
    """E4 三变量逐月 RMSE。"""
    set_seed(config.RANDOM_SEED)
    ol = config.OUTPUT_LEN
    df3, sc_ice, _, _ = load_trivariate_data(config.DATA_DIR, AO_CSV, SST_CSV, config.TARGET_COLUMN)
    di = df3[['area_scaled', 'ao_scaled', 'sst_scaled']].values
    dt = df3['area_scaled'].values
    X, y = create_sequences(di, 12, ol, target_data=dt)

    df_months = df3['month'].values
    df_years = df3['year'].values
    n_total = len(X)
    all_target_months = np.array([df_months[12 + i : 12 + i + ol] for i in range(n_total)])
    years_arr = df_years[12 : 12 + n_total]
    test_mask_all = (years_arr >= 2016) & (years_arr <= 2025)
    test_target_months = all_target_months[test_mask_all]

    (_, _), (_, _), (X_te, y_te) = train_val_test_split_by_year(df3, X, y, config.TARGET_COLUMN)

    model = SeaIceLSTM(3, 64, 1, ol, 0.1).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "paper_e4.pth"), map_location=device))
    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    y_pred = sc_ice.inverse_transform(yp).flatten()
    y_true = sc_ice.inverse_transform(y_te).flatten()
    months_flat = test_target_months.flatten()
    # 确保长度一致
    n_min = min(len(y_pred), len(months_flat))
    return compute_monthly_rmse(y_pred[:n_min], y_true[:n_min], months_flat[:n_min])


def main():
    print("Computing E7v1 monthly RMSE...")
    e7_rmse = get_e7v1_monthly_rmse()
    print("Computing E4 monthly RMSE...")
    e4_rmse = get_e4_monthly_rmse()

    x12 = np.arange(1, 13)

    # 打印
    print(f"\n{'月份':<6} {'E4 RMSE':>10} {'E7v1 RMSE':>10}")
    print("-" * 28)
    for m in x12:
        print(f"{m}月     {e4_rmse[m-1]:>8.4f}   {e7_rmse[m-1]:>8.4f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(x12, e4_rmse, 's-', color='#2166ac', linewidth=2.2, markersize=9,
            label='三变量 E4', zorder=4)
    ax.plot(x12, e7_rmse, 'o-', color='#1b7837', linewidth=2.2, markersize=9,
            label='双编码器 E7v1', zorder=4)
    ax.set_xlabel('月份', fontsize=13)
    ax.set_ylabel('均方根误差 (百万平方公里)', fontsize=13)
    ax.set_xticks(x12)
    ax.set_xticklabels([f'{m}月' for m in x12], fontsize=11)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 12.5)

    plt.tight_layout()
    save_path = os.path.join(OUT, "fig4-4_e4_e7_monthly_rmse.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
