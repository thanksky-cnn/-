"""
导出预测值与真实值逐月对比数据

对 6 个月和 12 个月预测，每个测试样本逐月输出：
  actual vs LinearRegression vs SimpleRNN vs LSTM

生成文件：
  outputs/results/predictions_6m.csv
  outputs/results/predictions_12m.csv
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import config
from src.data_preprocessing import load_and_merge_data, create_sequences, train_val_test_split_by_year
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM, SimpleRNNModel, LinearRegressionModel, count_parameters
from src.train import train_model, predict
from src.utils import set_seed, calculate_metrics
from compare_models import create_dataloaders, train_and_evaluate


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


def export_predictions(output_len):
    """训练三个模型，返回 y_true, y_preds_dict, years"""
    scheme_map = {6: "medium", 12: "long"}
    scheme = config._SCHEME_PARAMS[scheme_map[output_len]]
    for k, v in scheme.items():
        setattr(config, k, v)

    print(f"\n  Output={output_len}m | hidden={config.HIDDEN_SIZE} layers={config.NUM_LAYERS}")
    print(f"  lr={config.LEARNING_RATE:.6f} batch={config.BATCH_SIZE}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    df, scaler = load_and_merge_data(config.DATA_DIR, config.TARGET_COLUMN)
    data_scaled = df[f'{config.TARGET_COLUMN}_scaled'].values
    X, y = create_sequences(data_scaled, 12, output_len)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split_by_year(
        df, X, y, config.TARGET_COLUMN
    )

    # 提取测试集样本对应的年份（与 train_val_test_split_by_year 相同公式）
    input_len = X.shape[1]
    ol = y.shape[1]
    years_all = df['year'].values[input_len : len(df) - ol + 1]
    test_mask = (years_all >= 2016) & (years_all <= 2025)
    years_test_arr = years_all[test_mask]

    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val,
                                                  config.BATCH_SIZE)

    models_def = [
        ("LinearRegression", LinearRegressionModel(12, output_len)),
        ("SimpleRNN", SimpleRNNModel(1, hidden_size=64, num_layers=1,
                                      output_len=output_len, dropout=0.2)),
        ("LSTM", SeaIceLSTM(1, config.HIDDEN_SIZE, config.NUM_LAYERS,
                            output_len, config.DROPOUT)),
    ]

    all_preds = {}
    y_true = None

    for name, model in models_def:
        model = model.to(device)
        print(f"\n  --- {name} ---")
        metrics, y_pred, yt, _, _, best_epoch, train_time = train_and_evaluate(
            model, name, train_loader, val_loader, X_test, y_test, scaler, config, device
        )
        all_preds[name] = y_pred
        y_true = yt

    return y_true, all_preds, years_test_arr


def save_comparison_csv(y_true, all_preds, years, output_len, save_path):
    """生成逐月对比 CSV"""
    n_samples = len(y_true)
    n_months = output_len

    rows = []
    for i in range(n_samples):
        yr = int(years[i])
        for m in range(n_months):
            row = {
                "sample_id": i + 1,
                "year": yr,
                "forecast_month": m + 1,
                "actual": round(float(y_true[i, m]), 5),
            }
            for mn in ["LinearRegression", "SimpleRNN", "LSTM"]:
                row[f"pred_{mn}"] = round(float(all_preds[mn][i, m]), 5)
                row[f"error_{mn}"] = round(float(all_preds[mn][i, m] - y_true[i, m]), 5)
            rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)

    # 打印前 30 行预览
    print(f"\n  {'='*80}")
    print(f"  预测 vs 真实值 — {output_len} 个月预测（前 12 个样本）")
    print(f"  {'='*80}")
    col_order = ["sample_id", "year", "forecast_month", "actual",
                 "pred_LinearRegression", "pred_SimpleRNN", "pred_LSTM"]
    preview = df[col_order].head(12 * output_len)
    print(preview.to_string(index=False))

    df.to_csv(save_path, index=False, float_format="%.5f")
    print(f"\n  完整数据 ({len(df)} 行) 已保存: {save_path}")


def main():
    set_all_seeds(config.RANDOM_SEED)
    os.makedirs("outputs/results", exist_ok=True)

    for output_len in [6, 12]:
        print(f"\n{'='*60}")
        print(f"  导出: {output_len} 个月预测数据")
        print(f"{'='*60}")
        y_true, all_preds, years = export_predictions(output_len)
        save_comparison_csv(y_true, all_preds, years, output_len,
                           f"outputs/results/predictions_{output_len}m.csv")

    # 复制到汇总文件夹
    import shutil
    dest = "outputs/experiment_results_2026-05-11/metrics"
    os.makedirs(dest, exist_ok=True)
    for output_len in [6, 12]:
        src = f"outputs/results/predictions_{output_len}m.csv"
        if os.path.exists(src):
            shutil.copy2(src, dest + "/")

    print(f"\n  ✅ 完成！预测数据已保存到 outputs/results/ 和汇总文件夹")


if __name__ == "__main__":
    main()
