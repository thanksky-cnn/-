"""三变量 E4 — 测试集多年逐月平均预测值与实测值（mean ± 1σ）。"""
import sys, os, json
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from src.data_preprocessing import load_trivariate_data, create_sequences, train_val_test_split_by_year
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

AO_CSV = os.path.join(config.BASE_DIR, "data", "ao_monthly.csv")
SST_CSV = os.path.join(config.BASE_DIR, "data", "arctic_sst_monthly.csv")
MODEL_DIR = config.MODELS_DIR
OUT = os.path.join(config.PLOTS_DIR, "paper")
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_seed(config.RANDOM_SEED)
    ol = config.OUTPUT_LEN

    # 1. 加载三变量数据
    df3, sc_ice, sc_ao, sc_sst = load_trivariate_data(
        config.DATA_DIR, AO_CSV, SST_CSV, config.TARGET_COLUMN)
    di = df3[['area_scaled', 'ao_scaled', 'sst_scaled']].values
    dt = df3['area_scaled'].values
    X, y = create_sequences(di, 12, ol, target_data=dt)

    # 2. 年份划分 + 目标日历月
    df_months = df3['month'].values
    df_years = df3['year'].values
    n_total = len(X)

    all_target_months = np.array(
        [df_months[12 + i : 12 + i + ol] for i in range(n_total)])
    years_arr = df_years[12 : 12 + n_total]
    test_mask = (years_arr >= 2016) & (years_arr <= 2025)

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = \
        train_val_test_split_by_year(df3, X, y, config.TARGET_COLUMN)
    test_target_months = all_target_months[test_mask]

    # 3. 加载 E4 模型 (hidden_size=64, num_layers=1)
    model = SeaIceLSTM(
        input_size=3, hidden_size=64, num_layers=1,
        output_len=ol, dropout=0.1).to(device)
    model_path = os.path.join(MODEL_DIR, "paper_e4.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded: {model_path}")

    # 4. 预测
    with torch.no_grad():
        yp = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    y_pred = sc_ice.inverse_transform(yp)
    y_true = sc_ice.inverse_transform(y_te)

    # 5. 按日历月分组
    months_all = np.arange(1, 13)
    pred_by_month = {m: [] for m in months_all}
    true_by_month = {m: [] for m in months_all}

    for i in range(len(y_pred)):
        for k in range(ol):
            cal_month = int(test_target_months[i, k])
            pred_by_month[cal_month].append(y_pred[i, k])
            true_by_month[cal_month].append(y_true[i, k])

    pred_mean = np.array([np.mean(pred_by_month[m]) for m in months_all])
    pred_std  = np.array([np.std(pred_by_month[m])  for m in months_all])
    true_mean = np.array([np.mean(true_by_month[m]) for m in months_all])
    true_std  = np.array([np.std(true_by_month[m])  for m in months_all])

    # 6. 绘图
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x12 = np.arange(1, 13)

    ax.plot(x12, true_mean, 'o-', color='#d73027', linewidth=2.5, markersize=9,
            label='实测月均值', zorder=4)
    ax.fill_between(x12, true_mean - true_std, true_mean + true_std,
                     color='#d73027', alpha=0.12, edgecolor='none',
                     label='实测均值 ± 1σ')

    ax.plot(x12, pred_mean, 's--', color='#2166ac', linewidth=2.2, markersize=9,
            label='预测月均值', zorder=4)

    ax.set_xlabel('月份', fontsize=13)
    ax.set_ylabel('海冰面积 (百万平方公里)', fontsize=13)
    ax.set_xticks(x12)
    ax.set_xticklabels([f'{m}月' for m in x12], fontsize=11)
    ax.legend(fontsize=11, framealpha=0.9, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 12.5)

    plt.tight_layout()
    save_path = os.path.join(OUT, "fig_e4_trivariate_seasonal_mean_std.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # 打印
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae  = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"RMSE = {rmse:.4f}   MAE = {mae:.4f}   MAPE = {mape:.2f}%")
    print(f"\n{'月份':<6} {'实测均值':>8} {'实测±1σ':>14} {'预测均值':>8} {'偏差':>8}  {'样本数':>6}")
    print("-" * 62)
    for m in x12:
        n_m = len(true_by_month[m])
        print(f"{m}月     {true_mean[m-1]:>7.3f}  {true_mean[m-1]-true_std[m-1]:>6.3f}-{true_mean[m-1]+true_std[m-1]:<6.3f}"
              f"  {pred_mean[m-1]:>7.3f}  {pred_mean[m-1]-true_mean[m-1]:>+7.3f}  {n_m:>5}")


if __name__ == "__main__":
    main()
