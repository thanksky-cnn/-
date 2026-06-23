"""LSTM单变量长期预测 — 测试集多年逐月平均预测值与实测值（mean ± 1σ）。
   按日历月分组，避免不同季节混合平均导致曲线拉平。"""
import sys, os
import numpy as np
import torch
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

fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(config.PLOTS_DIR, "paper")
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ol = config.OUTPUT_LEN


def main():
    set_seed(config.RANDOM_SEED)

    # 1. 加载数据
    df, scaler = load_and_merge_data(config.DATA_DIR, config.TARGET_COLUMN)
    data = df[f'{config.TARGET_COLUMN}_scaled'].values
    X, y = create_sequences(data, 12, ol)

    # 2. 年份划分（使用与 short 相同的掩码逻辑）
    input_len = X.shape[1]   # 12
    output_len = y.shape[1]  # 12
    df_years = df['year'].values
    df_months = df['month'].values
    n_total = len(X)

    # 每个样本 i 的 12 个目标日历月：df_months[i+12 : i+24]
    all_target_months = np.array([df_months[i + 12 : i + 12 + ol] for i in range(n_total)])  # (N, 12)

    # split 函数内的 year 数组（与 train_val_test_split_by_year 一致）
    years = df_years[input_len : len(df) - output_len + 1]
    test_mask = (years >= 2016) & (years <= 2025)

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = \
        train_val_test_split_by_year(df, X, y, config.TARGET_COLUMN)
    test_target_months = all_target_months[test_mask]  # (N_test, 12)

    # 3. 加载模型
    model = SeaIceLSTM(
        input_size=1, hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS, output_len=ol,
        dropout=config.DROPOUT).to(device)
    model_path = os.path.join(config.MODELS_DIR, "best_model_area.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded: {model_path}")

    # 4. 预测
    with torch.no_grad():
        yp = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    y_pred = scaler.inverse_transform(yp)   # (N_test, 12)
    y_true = scaler.inverse_transform(y_te)

    # 5. 按日历月分组（12个月 × N_test 个预测值 → 聚合到12个月）
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

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae  = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

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
    save_path = os.path.join(OUT, "fig_lstm_univariate_seasonal_mean_std.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # 打印
    print(f"\n整体 RMSE = {rmse:.4f}   MAE = {mae:.4f}   MAPE = {mape:.2f}%   R² = {r2:.4f}")
    print(f"\n{'月份':<6} {'实测均值':>8} {'实测±1σ':>14} {'预测均值':>8} {'预测±1σ':>14} {'偏差':>8}  {'样本数':>6}")
    print("-" * 74)
    for m in x12:
        n_m = len(true_by_month[m])
        print(f"{m}月     {true_mean[m-1]:>7.3f}  {true_mean[m-1]-true_std[m-1]:>6.3f}-{true_mean[m-1]+true_std[m-1]:<6.3f}"
              f"  {pred_mean[m-1]:>7.3f}  {pred_mean[m-1]-pred_std[m-1]:>6.3f}-{pred_mean[m-1]+pred_std[m-1]:<6.3f}"
              f"  {pred_mean[m-1]-true_mean[m-1]:>+7.3f}  {n_m:>5}")


if __name__ == "__main__":
    main()
