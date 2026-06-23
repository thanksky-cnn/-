"""
中长期预测逐月误差分解脚本

对 output_len=6（中期）和 output_len=12（长期）分别：
1. 加载对应方案的 Optuna 最优参数训练模型
2. 逐个预测月计算 RMSE / MAE
3. 生成对比表和逐月误差柱状图
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
import config
from src.data_preprocessing import load_and_merge_data, create_sequences, train_val_test_split_by_year
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM, SimpleRNNModel, LinearRegressionModel, count_parameters
from src.train import train_model, predict
from src.utils import set_seed, calculate_metrics
from compare_models import create_dataloaders, train_and_evaluate

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# 要分析的输出长度
TARGETS = [6, 12]
INPUT_LEN = 12

# 评估指标
MODEL_DEFS = [
    ("LinearRegression", LinearRegressionModel),
    ("SimpleRNN", SimpleRNNModel),
    ("LSTM", SeaIceLSTM),
]


def set_all_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_analysis(output_len):
    """训练模型，返回逐月指标和整体 metrics"""
    scheme_map = {6: "medium", 12: "long"}
    scheme = config._SCHEME_PARAMS[scheme_map[output_len]]

    # 应用方案参数
    for k, v in scheme.items():
        setattr(config, k, v)

    print(f"\n{'='*70}")
    print(f"  Output={output_len}months | hidden={config.HIDDEN_SIZE} layers={config.NUM_LAYERS}")
    print(f"  lr={config.LEARNING_RATE:.6f} batch={config.BATCH_SIZE} wd={config.WEIGHT_DECAY:.2e}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 加载数据 ----
    df, scaler = load_and_merge_data(config.DATA_DIR, config.TARGET_COLUMN)
    data_scaled = df[f'{config.TARGET_COLUMN}_scaled'].values
    X, y = create_sequences(data_scaled, INPUT_LEN, output_len)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split_by_year(
        df, X, y, config.TARGET_COLUMN
    )
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val,
                                                  config.BATCH_SIZE)

    # ---- 逐个模型训练 ----
    all_monthly = {}  # model_name → {rmse:[], mae:[]}
    all_metrics = {}

    for model_name, ModelClass in MODEL_DEFS:
        # 构建模型
        if model_name == "LinearRegression":
            model = ModelClass(INPUT_LEN, output_len)
        elif model_name == "SimpleRNN":
            model = ModelClass(1, hidden_size=64, num_layers=1, output_len=output_len,
                               dropout=0.2)
        else:  # LSTM
            model = ModelClass(1, config.HIDDEN_SIZE, config.NUM_LAYERS,
                               output_len, config.DROPOUT)
        model = model.to(device)

        print(f"\n  --- {model_name} ({count_parameters(model):,} params) ---")

        metrics, y_pred, y_true, _, _, best_epoch, train_time = train_and_evaluate(
            model, model_name, train_loader, val_loader, X_test, y_test, scaler, config, device
        )
        all_metrics[model_name] = metrics

        # ---- 逐月计算 ----
        monthly_rmse = []
        monthly_mae = []
        for m in range(output_len):
            rmse_m = np.sqrt(np.mean((y_pred[:, m] - y_true[:, m]) ** 2))
            mae_m = np.mean(np.abs(y_pred[:, m] - y_true[:, m]))
            monthly_rmse.append(rmse_m)
            monthly_mae.append(mae_m)

        all_monthly[model_name] = {"rmse": monthly_rmse, "mae": monthly_mae}

    return all_monthly, all_metrics


def print_table(all_results):
    """打印逐月对比表"""
    print(f"\n{'='*110}")
    print(f"  逐月预测误差对比")
    print(f"{'='*110}")

    for output_len in TARGETS:
        monthly = all_results[output_len]["monthly"]
        metrics = all_results[output_len]["metrics"]
        n = output_len

        print(f"\n  ── 输出={n}个月 ──")
        header = f"  {'Month':<7}"
        for mn in ["LinearRegression", "SimpleRNN", "LSTM"]:
            header += f"{mn+' RMSE':<14} {mn+' MAE':<14}"
        print(header)
        print("  " + "-" * (7 + 28 * 3))

        for m in range(n):
            row = f"  {m+1:<7}"
            for mn in ["LinearRegression", "SimpleRNN", "LSTM"]:
                row += f"{monthly[mn]['rmse'][m]:<14.4f} {monthly[mn]['mae'][m]:<14.4f}"
            print(row)

        # 整体平均
        print("  " + "-" * (7 + 28 * 3))
        row = f"  {'Avg':<7}"
        for mn in ["LinearRegression", "SimpleRNN", "LSTM"]:
            row += f"{np.mean(monthly[mn]['rmse']):<14.4f} {np.mean(monthly[mn]['mae']):<14.4f}"
        print(row)


def plot_monthly_breakdown(all_results, save_dir):
    """逐月误差柱状图（分组对比）"""
    for output_len in TARGETS:
        monthly = all_results[output_len]["monthly"]
        n = output_len
        model_names = ["LinearRegression", "SimpleRNN", "LSTM"]
        colors = ['#3498db', '#2ecc71', '#e74c3c']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        months = np.arange(1, n + 1)
        width = 0.25

        # RMSE
        ax = axes[0]
        for i, mn in enumerate(model_names):
            vals = monthly[mn]["rmse"]
            bars = ax.bar(months + i * width, vals, width, label=mn, color=colors[i],
                          edgecolor='black', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{v:.2f}', ha='center', fontsize=6, rotation=90)
        ax.set_xlabel('Forecast Month', fontsize=11)
        ax.set_ylabel('RMSE (million km²)', fontsize=11)
        ax.set_title(f'Monthly RMSE ({output_len}-Month Forecast)', fontsize=13)
        ax.set_xticks(months + width)
        ax.set_xticklabels(months)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # MAE
        ax = axes[1]
        for i, mn in enumerate(model_names):
            vals = monthly[mn]["mae"]
            bars = ax.bar(months + i * width, vals, width, label=mn, color=colors[i],
                          edgecolor='black', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{v:.2f}', ha='center', fontsize=6, rotation=90)
        ax.set_xlabel('Forecast Month', fontsize=11)
        ax.set_ylabel('MAE (million km²)', fontsize=11)
        ax.set_title(f'Monthly MAE ({output_len}-Month Forecast)', fontsize=13)
        ax.set_xticks(months + width)
        ax.set_xticklabels(months)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = f"{save_dir}/monthly_breakdown_{output_len}m.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  图表: {path}")


def save_data_tables(all_results, save_dir):
    """保存逐月数据为 CSV"""
    import pandas as pd
    for output_len in TARGETS:
        monthly = all_results[output_len]["monthly"]
        rows = []
        for m in range(output_len):
            row = {"month": m + 1}
            for mn in ["LinearRegression", "SimpleRNN", "LSTM"]:
                row[f"{mn}_rmse"] = monthly[mn]["rmse"][m]
                row[f"{mn}_mae"] = monthly[mn]["mae"][m]
            rows.append(row)
        df = pd.DataFrame(rows)
        path = f"{save_dir}/monthly_breakdown_{output_len}m.csv"
        df.to_csv(path, index=False, float_format="%.4f")
        print(f"  数据: {path}")


def main():
    set_all_seeds(config.RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    all_results = {}
    for output_len in TARGETS:
        monthly, metrics = run_analysis(output_len)
        all_results[output_len] = {"monthly": monthly, "metrics": metrics}

    # 打印表格
    print_table(all_results)

    # 保存与绘图
    save_dir = "outputs/results"
    os.makedirs(save_dir, exist_ok=True)
    save_data_tables(all_results, save_dir)
    plot_monthly_breakdown(all_results, "outputs/plots")

    # 复制到汇总文件夹
    import shutil
    dest_dir = "outputs/experiment_results_2026-05-11"
    os.makedirs(f"{dest_dir}/metrics", exist_ok=True)
    os.makedirs(f"{dest_dir}/figures", exist_ok=True)
    for output_len in TARGETS:
        csv_path = f"{save_dir}/monthly_breakdown_{output_len}m.csv"
        png_path = f"outputs/plots/monthly_breakdown_{output_len}m.png"
        if os.path.exists(csv_path):
            shutil.copy2(csv_path, f"{dest_dir}/metrics/")
        if os.path.exists(png_path):
            shutil.copy2(png_path, f"{dest_dir}/figures/")

    print(f"\n  ✅ 完成！逐月数据已保存到 outputs/results/ 和 outputs/experiment_results_2026-05-11/")


if __name__ == "__main__":
    main()
