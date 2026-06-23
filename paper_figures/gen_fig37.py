"""图3.7: LR/RNN/LSTM 单变量长期预测逐月RMSE对比（按日历月）。"""
import sys, os, numpy as np, torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from src.data_preprocessing import load_and_merge_data, create_sequences, train_val_test_split_by_year
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM, LinearRegressionModel, SimpleRNNModel
from src.train import train_model
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
MODEL_DIR = config.MODELS_DIR


def monthly_rmse_by_calendar(y_pred_flat, y_true_flat, months_flat):
    """按日历月（1-12）分组计算 RMSE。"""
    out = []
    for m in range(1, 13):
        mask = months_flat == m
        if mask.sum() > 0:
            out.append(np.sqrt(np.mean((y_pred_flat[mask] - y_true_flat[mask])**2)))
        else:
            out.append(0.0)
    return np.array(out)


def main():
    set_seed(config.RANDOM_SEED)

    df, scaler = load_and_merge_data(config.DATA_DIR, config.TARGET_COLUMN)
    data = df['area_scaled'].values
    X, y = create_sequences(data, 12, ol)

    df_months = df['month'].values
    df_years = df['year'].values
    n_total = len(X)
    all_target_months = np.array([df_months[12 + i:12 + i + ol] for i in range(n_total)])
    years_arr = df_years[12:12 + n_total]
    test_mask = (years_arr >= 2016) & (years_arr <= 2025)
    test_months_flat = all_target_months[test_mask].flatten()

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = \
        train_val_test_split_by_year(df, X, y, config.TARGET_COLUMN)

    results = {}
    x12 = np.arange(1, 13)

    # ---- LR ----
    print("Training LinearRegression...")
    set_seed(config.RANDOM_SEED)
    lr = LinearRegressionModel(input_len=12, output_len=ol).to(device)
    tldr = DataLoader(SeaIceDataset(X_tr, y_tr), batch_size=config.BATCH_SIZE, shuffle=True)
    vldr = DataLoader(SeaIceDataset(X_v, y_v), batch_size=config.BATCH_SIZE)
    _, _, bp, _, _ = train_model(lr, tldr, vldr, config, device)
    lr.load_state_dict(torch.load(bp))
    lr.eval()
    with torch.no_grad():
        yp = lr(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    ypf = scaler.inverse_transform(yp).flatten()
    ytf = scaler.inverse_transform(y_te).flatten()
    n_min = min(len(ypf), len(test_months_flat))
    results['LR'] = monthly_rmse_by_calendar(ypf[:n_min], ytf[:n_min], test_months_flat[:n_min])
    print(f"  LR done, RMSE={np.sqrt(np.mean((ypf-ytf)**2)):.4f}")

    # ---- RNN ----
    print("Training SimpleRNN...")
    set_seed(config.RANDOM_SEED)
    rnn = SimpleRNNModel(input_size=1, hidden_size=64, num_layers=1, output_len=ol, dropout=0.2).to(device)
    _, _, bp, _, _ = train_model(rnn, tldr, vldr, config, device)
    rnn.load_state_dict(torch.load(bp))
    rnn.eval()
    with torch.no_grad():
        yp = rnn(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    ypf = scaler.inverse_transform(yp).flatten()
    ytf = scaler.inverse_transform(y_te).flatten()
    n_min = min(len(ypf), len(test_months_flat))
    results['RNN'] = monthly_rmse_by_calendar(ypf[:n_min], ytf[:n_min], test_months_flat[:n_min])
    print(f"  RNN done, RMSE={np.sqrt(np.mean((ypf-ytf)**2)):.4f}")

    # ---- LSTM ----
    print("Loading LSTM...")
    lstm = SeaIceLSTM(1, config.HIDDEN_SIZE, config.NUM_LAYERS, ol, config.DROPOUT).to(device)
    lstm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model_area.pth"), map_location=device))
    lstm.eval()
    with torch.no_grad():
        yp = lstm(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    ypf = scaler.inverse_transform(yp).flatten()
    ytf = scaler.inverse_transform(y_te).flatten()
    n_min = min(len(ypf), len(test_months_flat))
    results['LSTM'] = monthly_rmse_by_calendar(ypf[:n_min], ytf[:n_min], test_months_flat[:n_min])
    print(f"  LSTM done, RMSE={np.sqrt(np.mean((ypf-ytf)**2)):.4f}")

    # 打印
    print(f"\n{'月份':<6} {'LR':>10} {'RNN':>10} {'LSTM':>10}")
    print("-" * 38)
    for m in x12:
        print(f"{m}月     {results['LR'][m-1]:>8.4f}   {results['RNN'][m-1]:>8.4f}   {results['LSTM'][m-1]:>8.4f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x12, results['LSTM'], 'o-', color='#d73027', linewidth=2.2, markersize=9,
            label='LSTM（单变量）', zorder=4)
    ax.plot(x12, results['LR'], 's-', color='#2166ac', linewidth=2.2, markersize=9,
            label='线性回归', zorder=4)
    ax.plot(x12, results['RNN'], '^-', color='#1b7837', linewidth=2.2, markersize=9,
            label='RNN', zorder=4)

    ax.set_xlabel('月份', fontsize=13)
    ax.set_ylabel('均方根误差 (百万平方公里)', fontsize=13)
    ax.set_xticks(x12)
    ax.set_xticklabels([f'{m}月' for m in x12], fontsize=11)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 12.5)

    plt.tight_layout()
    save_path = os.path.join(OUT, "fig3-7_three_models_monthly_rmse.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
