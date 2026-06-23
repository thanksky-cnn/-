# ⭐ CORE: compare_models.py — LinearRegression vs SimpleRNN vs LSTM baseline comparison
import sys
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_preprocessing import load_and_merge_data, create_sequences, train_val_test_split_by_year
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM, SimpleRNNModel, LinearRegressionModel, count_parameters
from src.train import train_model, predict
from experiments.experiment_log_writer import append_log
from src.utils import set_seed, calculate_metrics

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    train_dataset = SeaIceDataset(X_train, y_train)
    val_dataset = SeaIceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def train_and_evaluate(model, model_name, train_loader, val_loader, X_test, y_test, scaler, config, device):
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    print(f"Parameters: {count_parameters(model):,}")
    
    train_losses, val_losses, best_model_path, best_epoch, train_time_s = train_model(
        model, train_loader, val_loader, config, device
    )
    
    model.load_state_dict(torch.load(best_model_path))
    y_pred = predict(model, X_test, device, scaler)
    y_true = scaler.inverse_transform(y_test)
    
    metrics = calculate_metrics(y_true, y_pred)
    metrics['params'] = count_parameters(model)
    
    monthly_rmse = []
    for m in range(config.OUTPUT_LEN):
        rmse_m = np.sqrt(np.mean((y_pred[:, m] - y_true[:, m]) ** 2))
        monthly_rmse.append(rmse_m)
    metrics['monthly_rmse'] = monthly_rmse
    
    # Log experiment details for this run
    log_entry = {
        "experiment_id": f"COMPARISON_{model_name}_{config.TARGET_COLUMN}_{config.OUTPUT_LEN}",
        "model": model_name,
        "input_len": config.INPUT_LEN,
        "output_len": config.OUTPUT_LEN,
        "hidden_size": getattr(config, "HIDDEN_SIZE", None),
        "num_layers": getattr(config, "NUM_LAYERS", None),
        "dropout": getattr(config, "DROPOUT", None),
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "reduce_lr_patience": getattr(config, "REDUCE_LR_PATIENCE", None),
        "reduce_lr_factor": getattr(config, "REDUCE_LR_FACTOR", None),
        "early_stopping_patience": getattr(config, "EARLY_STOPPING_PATIENCE", None),
        "train_split": getattr(config, "TRAIN_SPLIT", None),
        "val_split": getattr(config, "VAL_SPLIT", None),
        "target_column": config.TARGET_COLUMN,
        "scaler": "MinMaxScaler",
        "data_dir": config.DATA_DIR,
        "random_seed": config.RANDOM_SEED,
        "best_epoch": best_epoch,
        "train_time_s": train_time_s,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "r2": metrics["r2"],
        "monthly_rmse": monthly_rmse
    }
    try:
        append_log(log_entry, log_path=os.path.join("experiments", "experiment_log.jsonl"))
    except Exception as _e:
        print(f"Warning: failed to write compare experiment log: {_e}")

    return metrics, y_pred, y_true, train_losses, val_losses, best_epoch, train_time_s


def plot_comparison_bar(metrics_dict, save_path):
    # Define model roles for plotting
    model_roles = ['Baseline Model', 'Primary Model', 'Validation Model']
    rmse_values = [metrics_dict['LinearRegression (Baseline)']['rmse'],
                   metrics_dict['LSTM (Primary)']['rmse'],
                   metrics_dict['SimpleRNN (Validation)']['rmse']]
    mae_values = [metrics_dict['LinearRegression (Baseline)']['mae'],
                  metrics_dict['LSTM (Primary)']['mae'],
                  metrics_dict['SimpleRNN (Validation)']['mae']]
    r2_values = [metrics_dict['LinearRegression (Baseline)']['r2'],
                 metrics_dict['LSTM (Primary)']['r2'],
                 metrics_dict['SimpleRNN (Validation)']['r2']]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    axes[0].bar(model_roles, rmse_values, color=colors, edgecolor='black')
    axes[0].set_title('RMSE Comparison (lower is better)', fontsize=12)
    axes[0].set_ylabel('RMSE (million km²)', fontsize=11)
    for i, v in enumerate(rmse_values):
        axes[0].text(i, v + 0.05, f'{v:.4f}', ha='center', fontsize=10)

    axes[1].bar(model_roles, mae_values, color=colors, edgecolor='black')
    axes[1].set_title('MAE Comparison (lower is better)', fontsize=12)
    axes[1].set_ylabel('MAE (million km²)', fontsize=11)
    for i, v in enumerate(mae_values):
        axes[1].text(i, v + 0.03, f'{v:.4f}', ha='center', fontsize=10)

    axes[2].bar(model_roles, r2_values, color=colors, edgecolor='black')
    axes[2].set_title('R² Comparison (higher is better)', fontsize=12)
    axes[2].set_ylabel('R² Score', fontsize=11)
    for i, v in enumerate(r2_values):
        axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison bar chart saved to: {save_path}")


def plot_predictions_comparison(all_preds, y_true, model_names, save_path):
    n_samples = min(5, len(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Updated colors for role-based models
    colors = {'LinearRegression (Baseline)': '#3498db',
              'LSTM (Primary)': '#e74c3c',
              'SimpleRNN (Validation)': '#2ecc71'}

    x = np.arange(1, 13)
    for idx, (name, preds) in enumerate(all_preds.items()):
        for i in range(n_samples):
            label = name.replace(' (Baseline)', '').replace(' (Primary)', '').replace(' (Validation)', '') if i == 0 else ''
            axes[0].plot(x, y_true[i], 'o-', color='black', alpha=0.5, linewidth=1.5, label='Actual' if i == 0 else '')
            axes[0].plot(x, preds[i], 's--', color=colors[name], alpha=0.7, linewidth=1.5, label=label if i == 0 else '')

    axes[0].set_xlabel('Forecast Month', fontsize=11)
    axes[0].set_ylabel('Sea Ice Area (million km²)', fontsize=11)
    axes[0].set_title('Prediction Comparison by Model Role', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    for name, preds in all_preds.items():
        display_name = name.replace(' (Baseline)', '').replace(' (Primary)', '').replace(' (Validation)', '')
        axes[1].scatter(y_true.flatten(), preds.flatten(), alpha=0.4, s=15,
                       color=colors[name], label=display_name)

    min_val = y_true.min()
    max_val = y_true.max()
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Ideal')
    axes[1].set_xlabel('Actual (million km²)', fontsize=11)
    axes[1].set_ylabel('Predicted (million km²)', fontsize=11)
    axes[1].set_title('Scatter Plot by Model Role', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction comparison saved to: {save_path}")


def plot_monthly_rmse_comparison(metrics_dict, save_path):
    plt.figure(figsize=(12, 6))

    months = range(1, 13)
    colors = {'LinearRegression (Baseline)': '#3498db',
              'LSTM (Primary)': '#e74c3c',
              'SimpleRNN (Validation)': '#2ecc71'}

    for name, metrics in metrics_dict.items():
        display_name = name.replace(' (Baseline)', '').replace(' (Primary)', '').replace(' (Validation)', '')
        plt.plot(months, metrics['monthly_rmse'], 'o-', color=colors[name],
                linewidth=2, markersize=6, label=display_name)

    plt.xlabel('Forecast Month', fontsize=12)
    plt.ylabel('RMSE (million km²)', fontsize=12)
    plt.title('Monthly RMSE by Model Role', fontsize=14)
    plt.xticks(months)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Monthly RMSE comparison saved to: {save_path}")


def main():
    set_all_seeds(config.RANDOM_SEED)
    
    target_name = "Sea Ice Area" if config.TARGET_COLUMN == "area" else "Sea Ice Extent"
    
    print("=" * 60)
    print(f"Model Comparison: LinearRegression vs SimpleRNN vs LSTM")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "=" * 60)
    print("Step 1: Loading Data")
    print("=" * 60)
    
    df, scaler = load_and_merge_data(
        data_dir=config.DATA_DIR,
        target_column=config.TARGET_COLUMN,
        start_year=config.START_YEAR,
        end_year=config.END_YEAR
    )
    
    print("\n" + "=" * 60)
    print("Step 2: Creating Sequences")
    print("=" * 60)
    
    data_scaled = df[f'{config.TARGET_COLUMN}_scaled'].values
    X, y = create_sequences(data_scaled, config.INPUT_LEN, config.OUTPUT_LEN)
    
    print("\n" + "=" * 60)
    print("Step 3: Splitting Dataset")
    print("=" * 60)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split_by_year(
        df, X, y, config.TARGET_COLUMN
    )
    
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, config.BATCH_SIZE
    )
    
    models_config = [
        ('LinearRegression (Baseline)', lambda: LinearRegressionModel(config.INPUT_LEN, config.OUTPUT_LEN)),
        ('LSTM (Primary)', lambda: SeaIceLSTM(input_size=1, hidden_size=config.HIDDEN_SIZE,
                                     num_layers=config.NUM_LAYERS, output_len=config.OUTPUT_LEN,
                                     dropout=config.DROPOUT)),
        ('SimpleRNN (Validation)', lambda: SimpleRNNModel(input_size=1, hidden_size=64, num_layers=1,
                                              output_len=config.OUTPUT_LEN, dropout=0.2)),
    ]
    
    metrics_dict = {}
    all_preds = {}
    
    for model_name, model_fn in models_config:
        model = model_fn().to(device)
        metrics, y_pred, y_true, train_losses, val_losses, best_epoch, train_time_s = train_and_evaluate(
            model, model_name, train_loader, val_loader, X_test, y_test, scaler, config, device
        )
        metrics_dict[model_name] = metrics
        all_preds[model_name] = y_pred
    
    print("\n" + "=" * 70)
    print("Model Comparison Results - Role-Based Analysis")
    print("=" * 70)

    print(f"\n{'Model Role':<25} {'RMSE':<12} {'MAE':<12} {'MAPE':<10} {'R2':<10} {'Params':<10}")
    print("-" * 84)

    # Print baseline model first
    baseline_metrics = metrics_dict['LinearRegression (Baseline)']
    print(f"{'📊 Baseline Model':<25} {baseline_metrics['rmse']:<12.4f} {baseline_metrics['mae']:<12.4f} {baseline_metrics['mape']:<10.2f} {baseline_metrics['r2']:<10.4f} {baseline_metrics['params']:<10,}")

    # Print primary model with emphasis
    primary_metrics = metrics_dict['LSTM (Primary)']
    print(f"{'🎯 Primary Model':<25} {primary_metrics['rmse']:<12.4f} {primary_metrics['mae']:<12.4f} {primary_metrics['mape']:<10.2f} {primary_metrics['r2']:<10.4f} {primary_metrics['params']:<10,}")

    # Print validation model
    validation_metrics = metrics_dict['SimpleRNN (Validation)']
    print(f"{'🔍 Validation Model':<25} {validation_metrics['rmse']:<12.4f} {validation_metrics['mae']:<12.4f} {validation_metrics['mape']:<10.2f} {validation_metrics['r2']:<10.4f} {validation_metrics['params']:<10,}")
    
    # Performance comparison analysis
    print("\n" + "=" * 70)
    print("Performance Analysis vs Baseline")
    print("=" * 70)

    baseline_rmse = baseline_metrics['rmse']
    primary_rmse = primary_metrics['rmse']
    validation_rmse = validation_metrics['rmse']

    primary_improvement = ((baseline_rmse - primary_rmse) / baseline_rmse) * 100
    validation_improvement = ((baseline_rmse - validation_rmse) / baseline_rmse) * 100

    print(f"\nBaseline Model (LinearRegression): RMSE = {baseline_rmse:.4f}")
    print(f"Primary Model (LSTM) vs Baseline: RMSE = {primary_rmse:.4f} ({primary_improvement:+.1f}%)")
    print(f"Validation Model (SimpleRNN) vs Baseline: RMSE = {validation_rmse:.4f} ({validation_improvement:+.1f}%)")

    if primary_improvement > 0:
        print(f"\n✅ LSTM outperforms baseline by {primary_improvement:.1f}%")
    else:
        print(f"\n⚠️  LSTM underperforms baseline by {abs(primary_improvement):.1f}%")

    # Three key metrics comparison
    print("\n" + "=" * 80)
    print("Three Key Metrics Detailed Comparison")
    print("=" * 80)

    print(f"\n{'Metric':<15} {'Baseline':<15} {'Primary (LSTM)':<15} {'Validation':<15} {'Best':<10}")
    print("-" * 75)

    # RMSE comparison
    rmse_values = [baseline_metrics['rmse'], primary_metrics['rmse'], validation_metrics['rmse']]
    best_rmse_idx = rmse_values.index(min(rmse_values))
    rmse_labels = ['Baseline', 'Primary', 'Validation']

    print(f"{'RMSE':<15} {baseline_metrics['rmse']:<15.4f} {primary_metrics['rmse']:<15.4f} {validation_metrics['rmse']:<15.4f} {rmse_labels[best_rmse_idx]:<10}")

    # MAE comparison
    mae_values = [baseline_metrics['mae'], primary_metrics['mae'], validation_metrics['mae']]
    best_mae_idx = mae_values.index(min(mae_values))

    print(f"{'MAE':<15} {baseline_metrics['mae']:<15.4f} {primary_metrics['mae']:<15.4f} {validation_metrics['mae']:<15.4f} {rmse_labels[best_mae_idx]:<10}")

    # R² comparison
    r2_values = [baseline_metrics['r2'], primary_metrics['r2'], validation_metrics['r2']]
    best_r2_idx = r2_values.index(max(r2_values))

    print(f"{'R² Score':<15} {baseline_metrics['r2']:<15.4f} {primary_metrics['r2']:<15.4f} {validation_metrics['r2']:<15.4f} {rmse_labels[best_r2_idx]:<10}")

    print("\n" + "=" * 60)
    print("Monthly RMSE by Model")
    print("=" * 60)

    print(f"\n{'Month':<8}", end="")
    for name in ['Baseline', 'Primary', 'Validation']:
        print(f"{name:<18}", end="")
    print()
    print("-" * 60)

    for m in range(12):
        print(f"Month {m+1:<3}", end="")
        print(f"{baseline_metrics['monthly_rmse'][m]:<18.4f}", end="")
        print(f"{primary_metrics['monthly_rmse'][m]:<18.4f}", end="")
        print(f"{validation_metrics['monthly_rmse'][m]:<18.4f}", end="")
        print()
    
    # Prediction vs Actual value comparison
    print("\n" + "=" * 80)
    print("Prediction vs Actual Value Statistics")
    print("=" * 80)

    # Calculate correlation coefficients
    baseline_corr = np.corrcoef(y_true.flatten(), all_preds['LinearRegression (Baseline)'].flatten())[0, 1]
    primary_corr = np.corrcoef(y_true.flatten(), all_preds['LSTM (Primary)'].flatten())[0, 1]
    validation_corr = np.corrcoef(y_true.flatten(), all_preds['SimpleRNN (Validation)'].flatten())[0, 1]

    # Calculate bias (mean prediction error)
    baseline_bias = np.mean(all_preds['LinearRegression (Baseline)'] - y_true)
    primary_bias = np.mean(all_preds['LSTM (Primary)'] - y_true)
    validation_bias = np.mean(all_preds['SimpleRNN (Validation)'] - y_true)

    print(f"\n{'Model':<20} {'Correlation':<15} {'Bias (Pred-Act)':<20} {'Std Error':<15}")
    print("-" * 70)

    baseline_std_error = np.std(all_preds['LinearRegression (Baseline)'] - y_true)
    primary_std_error = np.std(all_preds['LSTM (Primary)'] - y_true)
    validation_std_error = np.std(all_preds['SimpleRNN (Validation)'] - y_true)

    print(f"{'Baseline':<20} {baseline_corr:<15.4f} {baseline_bias:<20.4f} {baseline_std_error:<15.4f}")
    print(f"{'Primary (LSTM)':<20} {primary_corr:<15.4f} {primary_bias:<20.4f} {primary_std_error:<15.4f}")
    print(f"{'Validation':<20} {validation_corr:<15.4f} {validation_bias:<20.4f} {validation_std_error:<15.4f}")

    # Prediction accuracy analysis
    print("\n" + "=" * 80)
    print("Prediction Accuracy Analysis")
    print("=" * 80)

    # Calculate percentage of predictions within different error thresholds
    thresholds = [0.1, 0.2, 0.5, 1.0]  # million km²

    print(f"\n{'Error Threshold':<15} {'Baseline':<15} {'Primary':<15} {'Validation':<15}")
    print("-" * 60)

    for threshold in thresholds:
        baseline_acc = np.mean(np.abs(all_preds['LinearRegression (Baseline)'] - y_true) <= threshold) * 100
        primary_acc = np.mean(np.abs(all_preds['LSTM (Primary)'] - y_true) <= threshold) * 100
        validation_acc = np.mean(np.abs(all_preds['SimpleRNN (Validation)'] - y_true) <= threshold) * 100

        print(f"{threshold:<15.1f} {baseline_acc:<15.1f}% {primary_acc:<15.1f}% {validation_acc:<15.1f}%")

    plot_comparison_bar(metrics_dict, f"{config.PLOTS_DIR}/model_comparison_metrics.png")
    plot_predictions_comparison(all_preds, y_true, metrics_dict.keys(),
                                f"{config.PLOTS_DIR}/model_comparison_predictions.png")
    plot_monthly_rmse_comparison(metrics_dict, f"{config.PLOTS_DIR}/model_comparison_monthly_rmse.png")

    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  {config.PLOTS_DIR}/model_comparison_metrics.png")
    print(f"  {config.PLOTS_DIR}/model_comparison_predictions.png")
    print(f"  {config.PLOTS_DIR}/model_comparison_monthly_rmse.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
