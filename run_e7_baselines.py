"""⭐ PRIMARY: E7 Dual-Encoder vs LinearRegression vs SimpleRNN on same data split."""
  - LinearRegression (baseline)
  - SimpleRNN (validation)
  - Dual-Encoder LSTM (E7, primary)

Same year-based split and same data range (1979-2025) for all models.
"""
import sys, os, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_preprocessing import (
    load_dual_encoder_data, create_dual_targets,
    load_and_merge_data, create_sequences
)
from src.dataset import SeaIceDataset
from src.model import (
    SeaIceLSTM, LinearRegressionModel, SimpleRNNModel,
    SeaIceDualEncoderLSTM, count_parameters
)
from src.train import train_model, predict
from src.utils import set_seed, calculate_metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

LAGGED_CSV = os.path.join(config.BASE_DIR, "data", "lagged_features.csv")
DE_PARAMS = os.path.join(config.RESULTS_DIR, "dual_encoder_best_params.json")


# ==================== Dataset & split helpers ====================

class DualEncoderDataset(Dataset):
    def __init__(self, X_main, X_aux, y):
        self.X_main = torch.tensor(X_main, dtype=torch.float32)
        self.X_aux = torch.tensor(X_aux, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_main)

    def __getitem__(self, idx):
        return self.X_main[idx], self.X_aux[idx], self.y[idx]


def year_split(X, y, years):
    train_m = (years >= 1979) & (years <= 2010)
    val_m = (years >= 2011) & (years <= 2015)
    test_m = (years >= 2016) & (years <= 2025)
    return ((X[train_m], y[train_m]),
            (X[val_m], y[val_m]),
            (X[test_m], y[test_m]))


def year_split_dual(Xm, Xa, y, years):
    train_m = (years >= 1979) & (years <= 2010)
    val_m = (years >= 2011) & (years <= 2015)
    test_m = (years >= 2016) & (years <= 2025)
    return ((Xm[train_m], Xa[train_m], y[train_m]),
            (Xm[val_m], Xa[val_m], y[val_m]),
            (Xm[test_m], Xa[test_m], y[test_m]))


# ==================== Train standard models ====================

def train_standard(model, name, X_tr, y_tr, X_v, y_v, X_te, y_te, scaler, device):
    train_ds = SeaIceDataset(X_tr, y_tr)
    val_ds = SeaIceDataset(X_v, y_v)
    train_ldr = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_ldr = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    n_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"Training {name} (params={n_params:,})")
    print(f"{'='*50}")

    t0 = time.time()
    train_losses, val_losses, best_path, best_ep, _ = train_model(
        model, train_ldr, val_ldr, config, device)
    train_time = time.time() - t0

    model.load_state_dict(torch.load(best_path))
    y_pred = predict(model, X_te, device, scaler)
    y_true = scaler.inverse_transform(y_te)

    metrics = calculate_metrics(y_true, y_pred)
    metrics['monthly_rmse'] = [np.sqrt(np.mean((y_pred[:, m] - y_true[:, m]) ** 2))
                               for m in range(config.OUTPUT_LEN)]
    metrics['params'] = n_params
    metrics['best_epoch'] = best_ep
    metrics['train_time_s'] = train_time
    metrics['y_pred'] = y_pred
    metrics['y_true'] = y_true
    return metrics


# ==================== Train dual-encoder ====================

def train_dual_encoder(Xm_tr, Xa_tr, y_tr, Xm_v, Xa_v, y_v,
                       Xm_te, Xa_te, y_te, scaler, device, cfg, label):
    train_ds = DualEncoderDataset(Xm_tr, Xa_tr, y_tr)
    val_ds = DualEncoderDataset(Xm_v, Xa_v, y_v)
    train_ldr = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_ldr = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)

    aux_hidden = getattr(cfg, 'AUX_HIDDEN_SIZE', 32)
    aux_dropout = getattr(cfg, 'AUX_DROPOUT', 0.6)

    model = SeaIceDualEncoderLSTM(
        input_size=1, main_hidden=cfg.HIDDEN_SIZE, aux_hidden=aux_hidden,
        aux_input_size=7, aux_seq_len=3, num_layers=cfg.NUM_LAYERS,
        output_len=cfg.OUTPUT_LEN, dropout=cfg.DROPOUT, aux_dropout=aux_dropout,
    ).to(device)

    n_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"Training {label} (params={n_params:,})")
    print(f"  main_hidden={cfg.HIDDEN_SIZE}, aux_hidden={aux_hidden}, aux_dropout={aux_dropout}")
    print(f"{'='*50}")

    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                           weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=cfg.REDUCE_LR_FACTOR, patience=cfg.REDUCE_LR_PATIENCE)

    tl, vl = [], []
    best_vl = float('inf')
    patience = 0
    best_path = os.path.join(config.MODELS_DIR, "best_model_de_compare.pth")
    best_ep = None
    t0 = time.time()

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        train_loss = 0
        for Xm_b, Xa_b, yb in train_ldr:
            Xm_b, Xa_b, yb = Xm_b.to(device), Xa_b.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xm_b, Xa_b), yb)
            loss.backward()
            if cfg.GRAD_CLIP_NORM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            opt.step()
            train_loss += loss.item()
        tl.append(train_loss / len(train_ldr))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xm_b, Xa_b, yb in val_ldr:
                Xm_b, Xa_b, yb = Xm_b.to(device), Xa_b.to(device), yb.to(device)
                val_loss += crit(model(Xm_b, Xa_b), yb).item()
        val_loss /= len(val_ldr)
        vl.append(val_loss)
        sched.step(val_loss)

        if val_loss < best_vl:
            best_vl = val_loss
            patience = 0
            torch.save(model.state_dict(), best_path)
            best_ep = epoch + 1
        else:
            patience += 1

        if (epoch + 1) % 10 == 0 or epoch < 5:
            lr = opt.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{cfg.NUM_EPOCHS}] TL={tl[-1]:.6f} VL={vl[-1]:.6f} LR={lr:.6f}")

        if patience >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at Epoch {epoch+1}")
            break

    train_time = time.time() - t0
    model.load_state_dict(torch.load(best_path))
    model.eval()
    with torch.no_grad():
        Xm_t = torch.tensor(Xm_te, dtype=torch.float32).to(device)
        Xa_t = torch.tensor(Xa_te, dtype=torch.float32).to(device)
        yp = model(Xm_t, Xa_t).cpu().numpy()
    y_pred = scaler.inverse_transform(yp)
    y_true = scaler.inverse_transform(y_te)

    metrics = calculate_metrics(y_true, y_pred)
    metrics['monthly_rmse'] = [np.sqrt(np.mean((y_pred[:, m] - y_true[:, m]) ** 2))
                               for m in range(cfg.OUTPUT_LEN)]
    metrics['params'] = n_params
    metrics['best_epoch'] = best_ep
    metrics['train_time_s'] = train_time
    metrics['y_pred'] = y_pred
    metrics['y_true'] = y_true
    return metrics


# ==================== Main ====================

def main():
    set_seed(config.RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    ol = config.OUTPUT_LEN
    target_name = "Sea Ice Area" if config.TARGET_COLUMN == "area" else "Sea Ice Extent"

    # ===== Step 1: Load dual-encoder data =====
    print("\n" + "=" * 60)
    print("Step 1: Loading Data")
    print("=" * 60)

    X_main, X_aux, df_de, scaler_de = load_dual_encoder_data(
        LAGGED_CSV, config.TARGET_COLUMN)
    y_de, years_de = create_dual_targets(df_de, 12, ol, 3, config.TARGET_COLUMN)
    n = min(len(X_main), len(y_de))
    X_main, X_aux, y_de, years_de = X_main[:n], X_aux[:n], y_de[:n], years_de[:n]

    # Univariate data for LR and RNN (same time range)
    df_uni, scaler_uni = load_and_merge_data(config.DATA_DIR, config.TARGET_COLUMN)
    df_uni = df_uni[df_uni['year'] >= df_de['year'].min()]
    data_uni = df_uni[f'{config.TARGET_COLUMN}_scaled'].values
    X_uni, y_uni = create_sequences(data_uni, 12, ol)
    years_uni = df_uni['year'].values[12:len(df_uni) - ol + 1]
    n_common = min(len(X_uni), len(X_main), len(years_uni))
    X_uni, y_uni = X_uni[:n_common], y_uni[:n_common]
    years_uni = years_uni[:n_common]
    X_main, X_aux, y_de = X_main[:n_common], X_aux[:n_common], y_de[:n_common]
    years_de = years_de[:n_common]

    # ===== Step 2: Split (same for all models) =====
    print("\n" + "=" * 60)
    print("Step 2: Splitting (year-based)")
    print("=" * 60)

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = year_split(X_uni, y_uni, years_uni)
    (Xm_tr, Xa_tr, y_tr_de), (Xm_v, Xa_v, y_v_de), (Xm_te, Xa_te, y_te_de) = \
        year_split_dual(X_main, X_aux, y_de, years_de)

    print(f"Train: {len(X_tr)}, Val: {len(X_v)}, Test: {len(X_te)}")

    # ===== Step 3: Train =====
    print("\n" + "=" * 60)
    print("Step 3: Training")
    print("=" * 60)

    # Load dual-encoder tuned params
    cfg_de = SimpleNamespace()
    for attr in dir(config):
        if not attr.startswith('_'):
            setattr(cfg_de, attr, getattr(config, attr))
    if os.path.exists(DE_PARAMS):
        with open(DE_PARAMS, 'r') as f:
            de_p = json.load(f)
        p = de_p.get(str(ol), {})
        cfg_de.HIDDEN_SIZE = p.get('main_hidden', cfg_de.HIDDEN_SIZE)
        cfg_de.AUX_HIDDEN_SIZE = p.get('aux_hidden', 32)
        cfg_de.DROPOUT = p.get('dropout', cfg_de.DROPOUT)
        cfg_de.AUX_DROPOUT = p.get('aux_dropout', 0.6)
        cfg_de.LEARNING_RATE = p.get('learning_rate', cfg_de.LEARNING_RATE)
        cfg_de.BATCH_SIZE = p.get('batch_size', cfg_de.BATCH_SIZE)
        cfg_de.WEIGHT_DECAY = p.get('weight_decay', cfg_de.WEIGHT_DECAY)

    # [1/3] LinearRegression
    print("\n[1/3] LinearRegression (Baseline)")
    lr_model = LinearRegressionModel(12, ol).to(device)
    m_lr = train_standard(lr_model, "LinearRegression", X_tr, y_tr, X_v, y_v,
                          X_te, y_te, scaler_uni, device)

    # [2/3] SimpleRNN
    print("\n[2/3] SimpleRNN (Validation)")
    rnn_model = SimpleRNNModel(
        input_size=1, hidden_size=64, num_layers=1,
        output_len=ol, dropout=0.2).to(device)
    m_rnn = train_standard(rnn_model, "SimpleRNN", X_tr, y_tr, X_v, y_v,
                           X_te, y_te, scaler_uni, device)

    # [3/3] Dual-Encoder
    print("\n[3/3] Dual-Encoder LSTM (E7)")
    m_de = train_dual_encoder(Xm_tr, Xa_tr, y_tr_de, Xm_v, Xa_v, y_v_de,
                              Xm_te, Xa_te, y_te_de, scaler_de, device, cfg_de,
                              "Dual-Encoder LSTM (E7)")

    # ===== Step 4: Compare =====
    print("\n" + "=" * 80)
    print("Step 4: Three-Model Comparison")
    print("=" * 80)

    labels = ['LinearRegression\n(Baseline)', 'SimpleRNN\n(Validation)',
              'Dual-Encoder LSTM\n(E7, Primary)']
    all_m = [m_lr, m_rnn, m_de]

    # Overall table
    print(f"\n{'Metric':<15} {labels[0].split(chr(10))[0]:<18} "
          f"{labels[1].split(chr(10))[0]:<18} {labels[2].split(chr(10))[0]:<18}")
    print("-" * 69)
    for mkey, mname in [('rmse','RMSE'), ('mae','MAE'), ('mape','MAPE'), ('r2','R2')]:
        vals = [f"{m[mkey]:.4f}" for m in all_m]
        print(f"{mname:<15} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}")

    # Monthly table
    print(f"\n{'Month':<6}", end="")
    for l in labels:
        print(f" {l.split(chr(10))[0]:<16}", end="")
    print(f"\n{'-'*54}")
    for mo in range(ol):
        print(f"{mo+1:<6}", end="")
        for m in all_m:
            print(f" {m['monthly_rmse'][mo]:<16.4f}", end="")
        print()

    # Info
    print(f"\n{'':<15} {labels[0].split(chr(10))[0]:<18} "
          f"{labels[1].split(chr(10))[0]:<18} {labels[2].split(chr(10))[0]:<18}")
    print("-" * 69)
    print(f"{'Params':<15} {m_lr['params']:<18,} {m_rnn['params']:<18,} {m_de['params']:<18,}")
    print(f"{'Best Epoch':<15} {m_lr['best_epoch']:<18} {m_rnn['best_epoch']:<18} {m_de['best_epoch']:<18}")
    print(f"{'Train Time':<15} {m_lr['train_time_s']:<17.1f}s {m_rnn['train_time_s']:<17.1f}s {m_de['train_time_s']:<17.1f}s")

    # Save
    rp = os.path.join(config.RESULTS_DIR, "e7_vs_baselines.txt")
    with open(rp, 'w', encoding='utf-8') as f:
        f.write("E7 Dual-Encoder LSTM vs Baselines\n")
        f.write(f"Target: {target_name}, Output: {ol} months\n")
        f.write("Data: 1979-2025 (SST masked pre-1981)\n")
        f.write("=" * 60 + "\n\n")

        short_names = ['LinearRegression', 'SimpleRNN', 'Dual-Encoder LSTM']
        f.write(f"{'Metric':<15} {'LinearReg':<15} {'SimpleRNN':<15} {'Dual-Enc':<15}\n")
        f.write("-" * 60 + "\n")
        for mkey, mname in [('rmse','RMSE'),('mae','MAE'),('mape','MAPE'),('r2','R2')]:
            vals = [f"{m[mkey]:.4f}" for m in all_m]
            f.write(f"{mname:<15} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15}\n")

        f.write(f"\nMonthly RMSE:\n")
        f.write(f"{'Month':<6} {'LinearReg':<15} {'SimpleRNN':<15} {'Dual-Enc':<15}\n")
        for mo in range(ol):
            f.write(f"{mo+1:<6} {m_lr['monthly_rmse'][mo]:<15.4f} "
                    f"{m_rnn['monthly_rmse'][mo]:<15.4f} {m_de['monthly_rmse'][mo]:<15.4f}\n")

    print(f"\nResults saved to: {rp}")

    # ===== Plots =====
    save_dir = os.path.join(config.PLOTS_DIR, "e7_vs_baselines")
    os.makedirs(save_dir, exist_ok=True)

    colors = ['#3498db', '#2ecc71', '#e74c3c']

    # Monthly RMSE plot
    fig, ax = plt.subplots(figsize=(12, 6))
    months = range(1, ol + 1)
    for m, name, c in zip(all_m, short_names, colors):
        ax.plot(months, m['monthly_rmse'], 'o-', color=c, linewidth=2,
                markersize=6, label=name)
    ax.set_xlabel('Forecast Month', fontsize=12)
    ax.set_ylabel('RMSE (million km²)', fontsize=12)
    ax.set_title('Monthly RMSE: E7 vs Baselines', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'monthly_rmse.png'), dpi=150)
    plt.close()

    # Metrics bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, mkey, mlabel in zip(axes, ['rmse', 'mae', 'r2'],
                                ['RMSE (lower better)', 'MAE (lower better)', 'R² (higher better)']):
        vals = [m[mkey] for m in all_m]
        bars = ax.bar(short_names, vals, color=colors, edgecolor='black')
        ax.set_title(mlabel, fontsize=12)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics.png'), dpi=150)
    plt.close()

    print(f"Plots saved to: {save_dir}")
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
