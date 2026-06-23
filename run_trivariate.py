"""⭐ PRIMARY: Three-way comparison — univariate vs bivariate vs trivariate LSTM."""

1. Optuna-tunes the trivariate model for the current output length
   (saves to outputs/results/trivariate_best_params.json)
2. Compares univariate vs bivariate (tuned) vs trivariate (tuned)
   with same year-based split.

Original univariate config.py params remain unchanged.
"""
import sys
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import optuna
import config
from src.data_preprocessing import (
    load_trivariate_data, load_multivariate_data,
    create_sequences, train_val_test_split_by_year, train_val_test_split
)
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM, count_parameters
from src.train import train_model, predict
from src.utils import set_seed, calculate_metrics

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


AO_CSV = os.path.join(config.BASE_DIR, "data", "ao_monthly.csv")
SST_CSV = os.path.join(config.BASE_DIR, "data", "arctic_sst_monthly.csv")
BI_PARAMS_PATH = os.path.join(config.RESULTS_DIR, "bivariate_best_params.json")
TRI_PARAMS_PATH = os.path.join(config.RESULTS_DIR, "trivariate_best_params.json")


# ==================== Parameter loading ====================

def load_tuned_params(json_path, output_len, label):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        all_p = json.load(f)
    key = str(output_len)
    if key not in all_p:
        return None
    params = all_p[key]
    print(f"Loaded {label} tuned params (out={output_len}): {params}")
    return params


def make_config_override(cfg, params_override):
    override = SimpleNamespace()
    for attr in dir(cfg):
        if not attr.startswith('_'):
            setattr(override, attr, getattr(cfg, attr))
    if params_override:
        key_map = {
            'hidden_size': 'HIDDEN_SIZE', 'num_layers': 'NUM_LAYERS',
            'dropout': 'DROPOUT', 'learning_rate': 'LEARNING_RATE',
            'batch_size': 'BATCH_SIZE', 'weight_decay': 'WEIGHT_DECAY',
        }
        for json_key, cfg_key in key_map.items():
            if json_key in params_override:
                setattr(override, cfg_key, params_override[json_key])
    return override


# ==================== Trivariate Optuna tuning ====================

def tune_trivariate(output_len, n_trials=30):
    """Run Optuna tuning for input_size=3, save best params to JSON."""

    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

        set_seed(42)

        df, sc_ice, sc_ao, sc_sst = load_trivariate_data(
            config.DATA_DIR, AO_CSV, SST_CSV, config.TARGET_COLUMN
        )
        data_input = df[[f'{config.TARGET_COLUMN}_scaled', 'ao_scaled', 'sst_scaled']].values
        data_target = df[f'{config.TARGET_COLUMN}_scaled'].values
        X, y = create_sequences(data_input, config.INPUT_LEN, output_len,
                                target_data=data_target)

        (X_train, y_train), (X_val, y_val), _ = train_val_test_split(X, y, 0.7, 0.15)

        train_ds = SeaIceDataset(X_train, y_train)
        val_ds = SeaIceDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SeaIceLSTM(
            input_size=3, hidden_size=hidden_size, num_layers=num_layers,
            output_len=output_len, dropout=dropout,
        ).to(device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(50):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    val_loss += criterion(model(batch_X), batch_y).item()
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 10:
                break
        return best_val_loss

    print(f"\n{'='*60}")
    print(f"  Trivariate Optuna Tuning: output_len = {output_len}")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction='minimize',
        study_name=f'trivariate_lstm_out{output_len}',
        storage=f'sqlite:///optuna_trivariate_out{output_len}.db',
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    best["best_val_loss"] = study.best_value

    # Save to JSON
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    # Load existing or create new
    if os.path.exists(TRI_PARAMS_PATH):
        with open(TRI_PARAMS_PATH, 'r') as f:
            all_best = json.load(f)
    else:
        all_best = {}
    all_best[str(output_len)] = best
    with open(TRI_PARAMS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_best, f, indent=2, ensure_ascii=False)

    print(f"  Best trivariate (out={output_len}): val_loss={study.best_value:.6f}")
    for k, v in best.items():
        if k != 'best_val_loss':
            print(f"    {k}: {v}")

    return best


# ==================== Model training ====================

def train_single_model(X_train, y_train, X_val, y_val, X_test, y_test,
                       scaler_ice, input_size, device, cfg_ov, label):
    train_dataset = SeaIceDataset(X_train, y_train)
    val_dataset = SeaIceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=cfg_ov.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg_ov.BATCH_SIZE)

    model = SeaIceLSTM(
        input_size=input_size,
        hidden_size=cfg_ov.HIDDEN_SIZE,
        num_layers=cfg_ov.NUM_LAYERS,
        output_len=cfg_ov.OUTPUT_LEN,
        dropout=cfg_ov.DROPOUT,
    ).to(device)

    print(f"\n{'='*50}")
    print(f"Training {label} (input_size={input_size}, params={count_parameters(model):,})")
    print(f"  hidden={cfg_ov.HIDDEN_SIZE}, layers={cfg_ov.NUM_LAYERS}, "
          f"dropout={cfg_ov.DROPOUT}, lr={cfg_ov.LEARNING_RATE:.6f}")
    print(f"{'='*50}")

    train_losses, val_losses, best_model_path, best_epoch, train_time_s = train_model(
        model, train_loader, val_loader, cfg_ov, device
    )

    model.load_state_dict(torch.load(best_model_path))
    y_pred = predict(model, X_test, device, scaler_ice)
    y_true = scaler_ice.inverse_transform(y_test)

    metrics = calculate_metrics(y_true, y_pred)
    monthly_rmse = []
    for m in range(cfg_ov.OUTPUT_LEN):
        rmse_m = np.sqrt(np.mean((y_pred[:, m] - y_true[:, m]) ** 2))
        monthly_rmse.append(rmse_m)
    metrics['monthly_rmse'] = monthly_rmse
    metrics['params'] = count_parameters(model)
    metrics['train_time_s'] = train_time_s
    metrics['best_epoch'] = best_epoch
    metrics['y_pred'] = y_pred
    metrics['y_true'] = y_true
    return metrics


# ==================== Plotting ====================

def plot_three_way(metrics_list, names, y_true, save_dir):
    """Three-way comparison plots: univariate vs bivariate vs trivariate."""
    os.makedirs(save_dir, exist_ok=True)

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    # Metrics bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_keys = ['rmse', 'mae', 'r2']
    metric_labels = ['RMSE (lower better)', 'MAE (lower better)', 'R² (higher better)']

    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        values = [m[mkey] for m in metrics_list]
        bars = ax.bar(names, values, color=colors, edgecolor='black')
        ax.set_title(mlabel, fontsize=12)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Monthly RMSE
    plt.figure(figsize=(12, 6))
    months = range(1, len(metrics_list[0]['monthly_rmse']) + 1)
    for m, name, color in zip(metrics_list, names, colors):
        plt.plot(months, m['monthly_rmse'], 'o-', color=color, linewidth=2,
                 markersize=6, label=name)
    plt.xlabel('Forecast Month', fontsize=12)
    plt.ylabel('RMSE (million km²)', fontsize=12)
    plt.title('Monthly RMSE: Three Models', fontsize=14)
    plt.xticks(months)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'monthly_rmse_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Scatter (each model uses its own y_true from metrics, since test sizes may differ)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, m, name, color in zip(axes, metrics_list, names, colors):
        yt = m['y_true'].flatten()
        yp = m['y_pred'].flatten()
        corr = np.corrcoef(yt, yp)[0, 1]
        ax.scatter(yt, yp, alpha=0.5, s=10, color=color)
        ax.plot([yt.min(), yt.max()], [yt.min(), yt.max()], 'k--', linewidth=1.5)
        ax.set_xlabel('Actual (M km²)', fontsize=11)
        ax.set_ylabel('Predicted (M km²)', fontsize=11)
        ax.set_title(f'{name}\nR={corr:.4f}', fontsize=12)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {save_dir}")


# ==================== Main ====================

def main():
    set_seed(config.RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    target_name = "Sea Ice Area" if config.TARGET_COLUMN == "area" else "Sea Ice Extent"
    output_len = config.OUTPUT_LEN

    # ==================== Step 1: Tune trivariate (if needed) ====================
    print("\n" + "=" * 60)
    print("Step 1: Tune Trivariate LSTM")
    print("=" * 60)

    tri_params = load_tuned_params(TRI_PARAMS_PATH, output_len, "trivariate")
    if tri_params is None:
        print(f"No trivariate params found for output_len={output_len}, running Optuna...")
        tri_params = tune_trivariate(output_len, n_trials=20)
    else:
        print(f"Using existing trivariate tuned params.")

    # ==================== Step 2: Load all data ====================
    print("\n" + "=" * 60)
    print("Step 2: Loading Data")
    print("=" * 60)

    # Trivariate data (smallest overlap: sea ice + AO + SST)
    df3, scaler_ice3, scaler_ao3, scaler_sst3 = load_trivariate_data(
        config.DATA_DIR, AO_CSV, SST_CSV, config.TARGET_COLUMN,
        start_year=config.START_YEAR, end_year=config.END_YEAR,
    )

    # Bivariate data (same overlap as trivariate for fair comparison)
    df2, scaler_ice2, scaler_ao2 = load_multivariate_data(
        config.DATA_DIR, AO_CSV, config.TARGET_COLUMN,
        start_year=config.START_YEAR, end_year=config.END_YEAR,
    )
    # Trim to trivariate time range for fair comparison
    df2 = df2[(df2['year'] >= df3['year'].min()) & (df2['year'] <= df3['year'].max())]

    # ==================== Step 3: Create sequences ====================
    print("\n" + "=" * 60)
    print("Step 3: Creating Sequences")
    print("=" * 60)

    # Univariate
    data_uni = df2[f'{config.TARGET_COLUMN}_scaled'].values
    X_uni, y_uni = create_sequences(data_uni, config.INPUT_LEN, output_len)

    # Bivariate
    data_bi_in = df2[[f'{config.TARGET_COLUMN}_scaled', 'ao_scaled']].values
    data_bi_tgt = df2[f'{config.TARGET_COLUMN}_scaled'].values
    X_bi, y_bi = create_sequences(data_bi_in, config.INPUT_LEN, output_len,
                                  target_data=data_bi_tgt)

    # Trivariate
    data_tri_in = df3[[f'{config.TARGET_COLUMN}_scaled', 'ao_scaled', 'sst_scaled']].values
    data_tri_tgt = df3[f'{config.TARGET_COLUMN}_scaled'].values
    X_tri, y_tri = create_sequences(data_tri_in, config.INPUT_LEN, output_len,
                                    target_data=data_tri_tgt)

    # ==================== Step 4: Split ====================
    print("\n" + "=" * 60)
    print("Step 4: Splitting (year-based)")
    print("=" * 60)

    (X_tr_u, y_tr_u), (X_v_u, y_v_u), (X_te_u, y_te_u) = \
        train_val_test_split_by_year(df2, X_uni, y_uni, config.TARGET_COLUMN)
    (X_tr_b, y_tr_b), (X_v_b, y_v_b), (X_te_b, y_te_b) = \
        train_val_test_split_by_year(df2, X_bi, y_bi, config.TARGET_COLUMN)
    (X_tr_t, y_tr_t), (X_v_t, y_v_t), (X_te_t, y_te_t) = \
        train_val_test_split_by_year(df3, X_tri, y_tri, config.TARGET_COLUMN)

    # ==================== Step 5: Train ====================
    print("\n" + "=" * 60)
    print("Step 5: Training")
    print("=" * 60)

    cfg_uni = make_config_override(config, None)
    bi_params = load_tuned_params(BI_PARAMS_PATH, output_len, "bivariate")
    cfg_bi = make_config_override(config, bi_params)
    cfg_tri = make_config_override(config, tri_params)

    print("\n[1/3] Univariate (original config)")
    m_uni = train_single_model(X_tr_u, y_tr_u, X_v_u, y_v_u, X_te_u, y_te_u,
                               scaler_ice2, 1, device, cfg_uni, "Univariate")

    print("\n[2/3] Bivariate (tuned)")
    m_bi = train_single_model(X_tr_b, y_tr_b, X_v_b, y_v_b, X_te_b, y_te_b,
                              scaler_ice2, 2, device, cfg_bi, "Bivariate")

    print("\n[3/3] Trivariate (tuned)")
    m_tri = train_single_model(X_tr_t, y_tr_t, X_v_t, y_v_t, X_te_t, y_te_t,
                               scaler_ice3, 3, device, cfg_tri, "Trivariate")

    # ==================== Step 6: Compare ====================
    print("\n" + "=" * 80)
    print("Step 6: Three-Way Comparison")
    print("=" * 80)

    all_metrics = [m_uni, m_bi, m_tri]
    labels = ['Univariate', 'Bivariate', 'Trivariate']

    # Overall metrics table
    print(f"\n{'Metric':<15} {'Univariate':<15} {'Bivariate':<15} {'Trivariate':<15}")
    print("-" * 60)
    for mkey, mname in [('rmse', 'RMSE (M km²)'), ('mae', 'MAE (M km²)'),
                         ('mape', 'MAPE (%)'), ('r2', 'R² Score')]:
        vals = [m[mkey] for m in all_metrics]
        print(f"{mname:<15} {vals[0]:<15.4f} {vals[1]:<15.4f} {vals[2]:<15.4f}")

    # Monthly RMSE table
    print(f"\n{'Month':<8} {'Univariate':<15} {'Bivariate':<15} {'Trivariate':<15}")
    print("-" * 53)
    for mo in range(output_len):
        vals = [m['monthly_rmse'][mo] for m in all_metrics]
        print(f"{mo+1:<8} {vals[0]:<15.4f} {vals[1]:<15.4f} {vals[2]:<15.4f}")

    # Info table
    print(f"\n{'':<15} {'Univariate':<15} {'Bivariate':<15} {'Trivariate':<15}")
    print("-" * 60)
    print(f"{'Params':<15} {m_uni['params']:<15,} {m_bi['params']:<15,} {m_tri['params']:<15,}")
    print(f"{'Best Epoch':<15} {m_uni['best_epoch']:<15} {m_bi['best_epoch']:<15} {m_tri['best_epoch']:<15}")
    print(f"{'Train Time':<15} {m_uni['train_time_s']:<14.1f}s {m_bi['train_time_s']:<14.1f}s {m_tri['train_time_s']:<14.1f}s")

    # Save table
    results_path = os.path.join(config.RESULTS_DIR, "three_way_comparison.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("Three-Way LSTM Comparison\n")
        f.write(f"Target: {target_name}, Output Length: {output_len} months\n")
        f.write(f"Data period: {df3['year'].min()}-{df3['year'].max()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Metric':<15} {'Univariate':<15} {'Bivariate':<15} {'Trivariate':<15}\n")
        f.write("-" * 60 + "\n")
        for mkey, mname in [('rmse', 'RMSE (M km²)'), ('mae', 'MAE (M km²)'),
                             ('mape', 'MAPE (%)'), ('r2', 'R² Score')]:
            vals = [m[mkey] for m in all_metrics]
            f.write(f"{mname:<15} {vals[0]:<15.4f} {vals[1]:<15.4f} {vals[2]:<15.4f}\n")
        f.write(f"\nMonthly RMSE:\n")
        f.write(f"{'Month':<8} {'Univariate':<15} {'Bivariate':<15} {'Trivariate':<15}\n")
        for mo in range(output_len):
            vals = [m['monthly_rmse'][mo] for m in all_metrics]
            f.write(f"{mo+1:<8} {vals[0]:<15.4f} {vals[1]:<15.4f} {vals[2]:<15.4f}\n")
    print(f"\nResults saved to: {results_path}")

    # ==================== Plots ====================
    print("\n" + "=" * 60)
    print("Step 7: Plotting")
    print("=" * 60)

    y_true_shared = scaler_ice2.inverse_transform(y_te_u)  # all have same test targets
    plot_three_way(all_metrics, labels, y_true_shared,
                   os.path.join(config.PLOTS_DIR, "three_way_comparison"))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
