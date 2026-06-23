# ⭐ CORE: main.py — Train SeaIceLSTM + evaluate + visualize
import sys
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader  # 添加这行
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_preprocessing import load_and_merge_data, create_sequences, train_val_test_split_by_year
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM, count_parameters
from src.train import train_model, predict
from src.utils import set_seed, calculate_metrics, save_metrics
from experiments.experiment_log_writer import append_log

# ==================== 设置matplotlib全局参数 ====================
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11


def set_all_seeds(seed=42):
    """固定所有随机种子，确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ... 后续函数定义和 main() 保持不变


def set_all_seeds(seed=42):
    """固定所有随机种子，确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_loss_curve(train_losses, val_losses, save_path):
    """绘制损失曲线（英文标签）"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to: {save_path}")


def plot_predictions(y_true, y_pred, save_path, target_name="Sea Ice"):
    """绘制预测结果（英文标签）"""
    n_samples = min(5, len(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Prediction comparison
    axes[0].set_title(f'Prediction Results (First {n_samples} Test Samples)', fontsize=14)
    axes[0].set_xlabel('Forecast Month (1-12)', fontsize=12)
    axes[0].set_ylabel(f'{target_name} (million km²)', fontsize=12)

    for i in range(n_samples):
        axes[0].plot(y_true[i], 'o-', color='blue', label='Actual' if i == 0 else '', alpha=0.7)
        axes[0].plot(y_pred[i], 's--', color='red', label='Predicted' if i == 0 else '', alpha=0.7)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Scatter plot
    all_true = y_true.flatten()
    all_pred = y_pred.flatten()
    corr = np.corrcoef(all_true, all_pred)[0, 1]

    axes[1].scatter(all_true, all_pred, alpha=0.5, s=10, c='steelblue')
    axes[1].plot([all_true.min(), all_true.max()],
                 [all_true.min(), all_true.max()],
                 'r--', linewidth=2, label='Ideal Line')
    axes[1].set_xlabel('Actual (million km²)', fontsize=12)
    axes[1].set_ylabel('Predicted (million km²)', fontsize=12)
    axes[1].set_title(f'Scatter Plot (Correlation: {corr:.3f})', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved to: {save_path}")


def plot_monthly_rmse(monthly_rmse, save_path):
    """绘制逐月RMSE柱状图（英文标签）"""
    plt.figure(figsize=(10, 6))
    months = range(1, len(monthly_rmse) + 1)
    colors = ['steelblue' if rmse < np.mean(monthly_rmse) else 'coral' for rmse in monthly_rmse]
    plt.bar(months, monthly_rmse, color=colors, edgecolor='black')
    plt.axhline(y=np.mean(monthly_rmse), color='red', linestyle='--',
                label=f'Mean RMSE: {np.mean(monthly_rmse):.4f}')
    plt.xlabel('Forecast Month', fontsize=12)
    plt.ylabel('RMSE (million km²)', fontsize=12)
    plt.title('Monthly Prediction Error (RMSE)', fontsize=14)
    plt.xticks(months)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Monthly RMSE plot saved to: {save_path}")


def plot_seasonal_pattern(y_true, y_pred, save_path, target_name="Sea Ice"):
    """绘制季节性模式对比（英文标签）"""
    # Calculate multi-year average seasonal pattern
    seasonal_true = np.mean(y_true, axis=0)
    seasonal_pred = np.mean(y_pred, axis=0)
    std_true = np.std(y_true, axis=0)

    plt.figure(figsize=(12, 6))
    months = range(1, 13)

    plt.plot(months, seasonal_true, 'o-', color='blue', linewidth=2,
             markersize=8, label='Actual (Multi-year Avg)')
    plt.fill_between(months, seasonal_true - std_true, seasonal_true + std_true,
                     color='blue', alpha=0.2, label='Actual +/- 1 Std')
    plt.plot(months, seasonal_pred, 's--', color='red', linewidth=2,
             markersize=8, label='Predicted (Multi-year Avg)')

    plt.xlabel('Month', fontsize=12)
    plt.ylabel(f'{target_name} (million km²)', fontsize=12)
    plt.title('Seasonal Pattern Comparison', fontsize=14)
    plt.xticks(months)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Seasonal pattern plot saved to: {save_path}")


def main():
    # 固定随机种子（必须在导入config之后）
    set_all_seeds(config.RANDOM_SEED)

    # Get target name for display
    target_name = "Sea Ice Area" if config.TARGET_COLUMN == "area" else "Sea Ice Extent"

    print("=" * 60)
    print(f"Arctic {target_name} Prediction using LSTM")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    # ==================== 1. Load Data ====================
    print("\n" + "=" * 60)
    print("Step 1: Loading Data")
    print("=" * 60)

    df, scaler = load_and_merge_data(
        data_dir=config.DATA_DIR,
        target_column=config.TARGET_COLUMN,
        start_year=config.START_YEAR,
        end_year=config.END_YEAR
    )

    # ==================== 2. Create Sequences ====================
    print("\n" + "=" * 60)
    print("Step 2: Creating Sequences")
    print("=" * 60)

    data_scaled = df[f'{config.TARGET_COLUMN}_scaled'].values
    X, y = create_sequences(data_scaled, config.INPUT_LEN, config.OUTPUT_LEN)

    # ==================== 3. Split Dataset ====================
    print("\n" + "=" * 60)
    print("Step 3: Splitting Dataset")
    print("=" * 60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split_by_year(
        df, X, y, config.TARGET_COLUMN
    )

    # ==================== 4. Create DataLoader ====================
    print("\n" + "=" * 60)
    print("Step 4: Creating DataLoader")
    print("=" * 60)

    train_dataset = SeaIceDataset(X_train, y_train)
    val_dataset = SeaIceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # ==================== 5. Create Model ====================
    print("\n" + "=" * 60)
    print("Step 5: Creating Model")
    print("=" * 60)

    model = SeaIceLSTM(
        input_size=1,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_len=config.OUTPUT_LEN,
        dropout=config.DROPOUT
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # ==================== 6. Train Model ====================
    print("\n" + "=" * 60)
    print("Step 6: Training Model")
    print("=" * 60)

    # Set model suffix based on target column
    model_suffix = "area" if config.TARGET_COLUMN == "area" else "extent"

    train_losses, val_losses, best_model_path, best_epoch, train_time_s = train_model(
        model, train_loader, val_loader, config, device
    )

    # Rename model file
    import shutil
    default_path = best_model_path
    new_path = os.path.join(config.MODELS_DIR, f"best_model_{model_suffix}.pth")
    if os.path.exists(default_path):
        shutil.move(default_path, new_path)
    best_model_path = new_path

    # Plot loss curve
    plot_loss_curve(train_losses, val_losses,
                    f"{config.PLOTS_DIR}/loss_curve_{model_suffix}.png")

    # ==================== 7. Evaluate Model ====================
    print("\n" + "=" * 60)
    print("Step 7: Model Evaluation")
    print("=" * 60)

    # Load best model
    model.load_state_dict(torch.load(best_model_path))

    # Predict
    y_pred = predict(model, X_test, device, scaler)
    y_true = scaler.inverse_transform(y_test)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)

    # Calculate monthly RMSE
    monthly_rmse = []
    for m in range(config.OUTPUT_LEN):
        rmse_m = np.sqrt(np.mean((y_pred[:, m] - y_true[:, m]) ** 2))
        monthly_rmse.append(rmse_m)
    metrics['monthly_rmse'] = monthly_rmse

    print(f"\n{target_name} Prediction Results:")
    print(f"RMSE: {metrics['rmse']:.4f} million km²")
    print(f"MAE: {metrics['mae']:.4f} million km²")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"R2: {metrics['r2']:.4f}")

    print(f"\nMonthly RMSE:")
    for m, rmse in enumerate(monthly_rmse, 1):
        print(f"  Month {m:2d}: {rmse:.4f} million km²")

    # Save metrics
    save_metrics(metrics, f"{config.RESULTS_DIR}/metrics_{model_suffix}.txt")

    # ==================== 8. Visualization ====================
    print("\n" + "=" * 60)
    print("Step 8: Generating Visualizations")
    print("=" * 60)

    # Prediction results plot
    plot_predictions(y_true, y_pred,
                     f"{config.PLOTS_DIR}/prediction_results_{model_suffix}.png",
                     target_name)

    # Monthly RMSE plot
    plot_monthly_rmse(monthly_rmse,
                      f"{config.PLOTS_DIR}/monthly_rmse_{model_suffix}.png")

    # Seasonal pattern plot
    plot_seasonal_pattern(y_true, y_pred,
                          f"{config.PLOTS_DIR}/seasonal_pattern_{model_suffix}.png",
                          target_name)

    # ==================== Complete ====================
    print("\n" + "=" * 60)
    print(f"{target_name} Prediction Model Training Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Best model: {best_model_path}")
    print(f"  Metrics: {config.RESULTS_DIR}/metrics_{model_suffix}.txt")
    print(f"  Plots: {config.PLOTS_DIR}/")
    print("=" * 60)

    # 8. 记录实验日志（LSTM 主流程日志）
    try:
        log_entry = {
            "experiment_id": f"LSTM_{config.TARGET_COLUMN}_{config.OUTPUT_LEN}",
            "model": "SeaIceLSTM",
            "input_len": config.INPUT_LEN,
            "output_len": config.OUTPUT_LEN,
            "hidden_size": config.HIDDEN_SIZE,
            "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "reduce_lr_patience": config.REDUCE_LR_PATIENCE,
            "reduce_lr_factor": config.REDUCE_LR_FACTOR,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "train_split": config.TRAIN_SPLIT,
            "val_split": config.VAL_SPLIT,
            "target_column": config.TARGET_COLUMN,
            "scaler": "MinMaxScaler",
            "data_dir": config.DATA_DIR,
            "random_seed": config.RANDOM_SEED,
            "best_epoch": best_epoch,
            "train_time_s": train_time_s,
            "rmse": metrics["rmse"] if isinstance(metrics, dict) and "rmse" in metrics else None,
            "mae": metrics["mae"] if isinstance(metrics, dict) and "mae" in metrics else None,
            "mape": metrics["mape"] if isinstance(metrics, dict) and "mape" in metrics else None,
            "r2": metrics["r2"] if isinstance(metrics, dict) and "r2" in metrics else None,
            "monthly_rmse": monthly_rmse if 'monthly_rmse' in locals() else None
        }
        append_log(log_entry, log_path=os.path.join("experiments", "experiment_log.jsonl"))
    except Exception as e:
        print(f"Warning: failed to write experiment log: {e}")


if __name__ == "__main__":
    main()
