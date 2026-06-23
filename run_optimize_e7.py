
"""⭐ PRIMARY: Optimize E7 — 50-trial Optuna + 5-seed ensemble + baseline comparison."""
import sys, os, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import optuna
import config
from src.data_preprocessing import (
    load_dual_encoder_data, create_dual_targets,
    load_and_merge_data, create_sequences
)
from src.dataset import SeaIceDataset
from src.model import (
    LinearRegressionModel, SimpleRNNModel,
    SeaIceDualEncoderLSTM, count_parameters
)
from src.train import train_model, predict
from src.utils import set_seed, calculate_metrics

LAGGED_CSV = os.path.join(config.BASE_DIR, "data", "lagged_features.csv")
DE_PARAMS = os.path.join(config.RESULTS_DIR, "dual_encoder_best_params.json")
N_ENSEMBLE = 5
TUNING_TRIALS = 50
ol = config.OUTPUT_LEN
target = config.TARGET_COLUMN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(config.RANDOM_SEED)

# ============ Data ============

print("=" * 60)
print("Loading Data")
print("=" * 60)

X_main, X_aux, df_de, scaler_de = load_dual_encoder_data(LAGGED_CSV, target)
y_de, years_de = create_dual_targets(df_de, 12, ol, 3, target)
n = min(len(X_main), len(y_de))
X_main, X_aux, y_de, years_de = X_main[:n], X_aux[:n], y_de[:n], years_de[:n]

df_uni, scaler_uni = load_and_merge_data(config.DATA_DIR, target)
df_uni = df_uni[df_uni['year'] >= df_de['year'].min()]
data_uni = df_uni[f'{target}_scaled'].values
X_uni, y_uni = create_sequences(data_uni, 12, ol)
years_uni = df_uni['year'].values[12:len(df_uni) - ol + 1]
nc = min(len(X_uni), len(X_main), len(years_uni))
X_uni, y_uni, years_uni = X_uni[:nc], y_uni[:nc], years_uni[:nc]
X_main, X_aux, y_de, years_de = X_main[:nc], X_aux[:nc], y_de[:nc], years_de[:nc]

# Year masks
u_tr_m = (years_uni >= 1979) & (years_uni <= 2010)
u_v_m  = (years_uni >= 2011) & (years_uni <= 2015)
u_te_m = (years_uni >= 2016) & (years_uni <= 2025)

X_tr_u, y_tr_u = X_uni[u_tr_m], y_uni[u_tr_m]
X_v_u, y_v_u   = X_uni[u_v_m],  y_uni[u_v_m]
X_te_u, y_te_u = X_uni[u_te_m], y_uni[u_te_m]
Xm_tr, Xa_tr, y_tr_de = X_main[u_tr_m], X_aux[u_tr_m], y_de[u_tr_m]
Xm_v,  Xa_v,  y_v_de  = X_main[u_v_m],  X_aux[u_v_m],  y_de[u_v_m]
Xm_te, Xa_te, y_te_de = X_main[u_te_m], X_aux[u_te_m], y_de[u_te_m]

print(f"Train: {X_tr_u.shape[0]}, Val: {X_v_u.shape[0]}, Test: {X_te_u.shape[0]}")

# ============ Dataset ============

class DEDataset(Dataset):
    def __init__(self, Xm, Xa, y):
        self.Xm = torch.tensor(Xm, dtype=torch.float32)
        self.Xa = torch.tensor(Xa, dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.float32)
    def __len__(self): return len(self.Xm)
    def __getitem__(self, idx): return self.Xm[idx], self.Xa[idx], self.y[idx]

# ============ Optuna Tuning ============

def tune_dual_encoder():
    def objective(trial):
        main_hidden = trial.suggest_categorical('main_hidden', [64, 128, 256])
        aux_hidden = trial.suggest_categorical('aux_hidden', [16, 32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.4, step=0.1)
        aux_dropout = trial.suggest_float('aux_dropout', 0.2, 0.7, step=0.1)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        bs = trial.suggest_categorical('batch_size', [8, 16, 32])
        wd = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        aux_seq_len = trial.suggest_categorical('aux_seq_len', [3, 6])

        set_seed(42)
        train_ds = DEDataset(Xm_tr, Xa_tr[:, -aux_seq_len:, :], y_tr_de)
        val_ds   = DEDataset(Xm_v,  Xa_v[:,  -aux_seq_len:, :], y_v_de)
        train_ldr = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_ldr   = DataLoader(val_ds,   batch_size=bs)

        model = SeaIceDualEncoderLSTM(
            input_size=1, main_hidden=main_hidden, aux_hidden=aux_hidden,
            aux_input_size=7, aux_seq_len=aux_seq_len, num_layers=1,
            output_len=ol, dropout=dropout, aux_dropout=aux_dropout,
        ).to(device)

        crit = nn.MSELoss()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        best_vl = float('inf'); patience = 0
        for _ in range(50):
            model.train()
            for Xm_b, Xa_b, yb in train_ldr:
                Xm_b, Xa_b, yb = Xm_b.to(device), Xa_b.to(device), yb.to(device)
                opt.zero_grad()
                crit(model(Xm_b, Xa_b), yb).backward()
                opt.step()
            model.eval(); vl = 0
            with torch.no_grad():
                for Xm_b, Xa_b, yb in val_ldr:
                    Xm_b, Xa_b, yb = Xm_b.to(device), Xa_b.to(device), yb.to(device)
                    vl += crit(model(Xm_b, Xa_b), yb).item()
            vl /= len(val_ldr)
            if vl < best_vl: best_vl = vl; patience = 0
            else: patience += 1
            if patience >= 10: break
        return best_vl

    print(f"\n{'='*60}")
    print(f"  Optuna Tuning: {TUNING_TRIALS} trials")
    print(f"{'='*60}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        study_name='de_optimized_out12',
        storage='sqlite:///optuna_de_optimized.db',
    )
    study.optimize(objective, n_trials=TUNING_TRIALS, show_progress_bar=True)
    best = study.best_params
    best["best_val_loss"] = study.best_value
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(DE_PARAMS, 'w') as f:
        json.dump({"12": best}, f, indent=2)
    print(f"  Best: val_loss={study.best_value:.6f}")
    for k, v in best.items():
        if k != 'best_val_loss': print(f"    {k}: {v}")
    return best

if os.path.exists(DE_PARAMS):
    with open(DE_PARAMS, 'r') as f:
        bp = json.load(f).get("12")
    print(f"Using existing tuned params: {bp}")
else:
    bp = tune_dual_encoder()

# Build config override
cfg_de = SimpleNamespace()
for attr in dir(config):
    if not attr.startswith('_'): setattr(cfg_de, attr, getattr(config, attr))
cfg_de.HIDDEN_SIZE = bp.get('main_hidden', 128)
cfg_de.AUX_HIDDEN_SIZE = bp.get('aux_hidden', 32)
cfg_de.DROPOUT = bp.get('dropout', 0.1)
cfg_de.AUX_DROPOUT = bp.get('aux_dropout', 0.6)
cfg_de.LEARNING_RATE = bp.get('lr', 0.0026)
cfg_de.BATCH_SIZE = bp.get('batch_size', 8)
cfg_de.WEIGHT_DECAY = bp.get('weight_decay', 5e-6)
aux_sl = bp.get('aux_seq_len', 3)

# ============ Ensemble Training ============

print(f"\n{'='*60}")
print(f"Training Ensemble ({N_ENSEMBLE} seeds)")
print(f"{'='*60}")

def train_one_de(seed, Xm_tr_i, Xa_tr_i, y_tr_i, Xm_v_i, Xa_v_i, y_v_i, Xm_te_i, Xa_te_i):
    set_seed(seed)
    train_ds = DEDataset(Xm_tr_i, Xa_tr_i[:, -aux_sl:, :], y_tr_i)
    val_ds   = DEDataset(Xm_v_i,  Xa_v_i[:,  -aux_sl:, :], y_v_i)
    train_ldr = DataLoader(train_ds, batch_size=cfg_de.BATCH_SIZE, shuffle=True)
    val_ldr   = DataLoader(val_ds,   batch_size=cfg_de.BATCH_SIZE)

    model = SeaIceDualEncoderLSTM(
        input_size=1, main_hidden=cfg_de.HIDDEN_SIZE,
        aux_hidden=cfg_de.AUX_HIDDEN_SIZE, aux_input_size=7,
        aux_seq_len=aux_sl, num_layers=cfg_de.NUM_LAYERS,
        output_len=ol, dropout=cfg_de.DROPOUT,
        aux_dropout=cfg_de.AUX_DROPOUT,
    ).to(device)

    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_de.LEARNING_RATE,
                            weight_decay=cfg_de.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=cfg_de.REDUCE_LR_FACTOR,
        patience=cfg_de.REDUCE_LR_PATIENCE)

    best_vl = float('inf'); patience = 0; best_ep = None
    best_path = os.path.join(config.MODELS_DIR, f'best_model_de_s{seed}.pth')
    t0 = time.time()

    for epoch in range(cfg_de.NUM_EPOCHS):
        model.train(); tl = 0
        for Xm_b, Xa_b, yb in train_ldr:
            Xm_b, Xa_b, yb = Xm_b.to(device), Xa_b.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(Xm_b, Xa_b), yb)
            loss.backward()
            if cfg_de.GRAD_CLIP_NORM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_de.GRAD_CLIP_NORM)
            opt.step(); tl += loss.item()
        tl /= len(train_ldr)

        model.eval(); vl = 0
        with torch.no_grad():
            for Xm_b, Xa_b, yb in val_ldr:
                Xm_b, Xa_b, yb = Xm_b.to(device), Xa_b.to(device), yb.to(device)
                vl += crit(model(Xm_b, Xa_b), yb).item()
        vl /= len(val_ldr); sched.step(vl)

        if vl < best_vl:
            best_vl = vl; patience = 0
            torch.save(model.state_dict(), best_path); best_ep = epoch + 1
        else: patience += 1

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"  S{seed} E{epoch+1:3d}: TL={tl:.6f} VL={vl:.6f}")

        if patience >= cfg_de.EARLY_STOPPING_PATIENCE:
            print(f"  S{seed} early stop at E{epoch+1}")
            break

    train_time = time.time() - t0
    model.load_state_dict(torch.load(best_path))
    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(Xm_te_i, dtype=torch.float32).to(device),
                   torch.tensor(Xa_te_i[:, -aux_sl:, :], dtype=torch.float32).to(device)).cpu().numpy()
    y_pred = scaler_de.inverse_transform(yp)
    return y_pred, best_ep, train_time

all_preds = []; best_eps = []; train_times = []
for s in range(N_ENSEMBLE):
    yp, be, tt = train_one_de(42 + s, Xm_tr, Xa_tr, y_tr_de, Xm_v, Xa_v, y_v_de, Xm_te, Xa_te)
    all_preds.append(yp); best_eps.append(be); train_times.append(tt)

ensemble_pred = np.mean(all_preds, axis=0)
y_true_de = scaler_de.inverse_transform(y_te_de)
m_de_ens = calculate_metrics(y_true_de, ensemble_pred)
m_de_ens['monthly_rmse'] = [np.sqrt(np.mean((ensemble_pred[:, m] - y_true_de[:, m]) ** 2)) for m in range(ol)]
m_de_ens['params'] = count_parameters(SeaIceDualEncoderLSTM(
    input_size=1, main_hidden=cfg_de.HIDDEN_SIZE, aux_hidden=cfg_de.AUX_HIDDEN_SIZE,
    aux_input_size=7, aux_seq_len=aux_sl, num_layers=1,
    output_len=ol, dropout=cfg_de.DROPOUT, aux_dropout=cfg_de.AUX_DROPOUT))

best_idx = min(range(N_ENSEMBLE), key=lambda i: np.sqrt(np.mean((all_preds[i] - y_true_de) ** 2)))
m_de_single = calculate_metrics(y_true_de, all_preds[best_idx])
m_de_single['monthly_rmse'] = [np.sqrt(np.mean((all_preds[best_idx][:, m] - y_true_de[:, m]) ** 2)) for m in range(ol)]

# ============ Baselines ============

print(f"\n{'='*60}")
print("Training Baselines")
print(f"{'='*60}")

def train_baseline(model_cls, name, **kwargs):
    set_seed(config.RANDOM_SEED)
    model = model_cls(**kwargs).to(device)
    train_ds = SeaIceDataset(X_tr_u, y_tr_u)
    val_ds   = SeaIceDataset(X_v_u, y_v_u)
    train_ldr = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_ldr   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE)
    print(f"\n{name}: {count_parameters(model):,} params")
    t0 = time.time()
    _, _, best_path, best_ep, _ = train_model(model, train_ldr, val_ldr, config, device)
    tt = time.time() - t0
    model.load_state_dict(torch.load(best_path))
    y_pred = predict(model, X_te_u, device, scaler_uni)
    y_true = scaler_uni.inverse_transform(y_te_u)
    metrics = calculate_metrics(y_true, y_pred)
    metrics['monthly_rmse'] = [np.sqrt(np.mean((y_pred[:, m] - y_true[:, m]) ** 2)) for m in range(ol)]
    metrics['params'] = count_parameters(model)
    metrics['best_epoch'] = best_ep
    metrics['train_time_s'] = tt
    metrics['y_pred'] = y_pred
    metrics['y_true'] = y_true
    return metrics

m_lr  = train_baseline(LinearRegressionModel, "LinearRegression", input_len=12, output_len=ol)
m_rnn = train_baseline(SimpleRNNModel, "SimpleRNN",
                       input_size=1, hidden_size=64, num_layers=1, output_len=ol, dropout=0.2)

# ============ Comparison ============

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")

labels = ['LinearRegression', 'SimpleRNN', 'Dual-Enc(best)', 'Dual-Enc(ens5)']
all_m  = [m_lr, m_rnn, m_de_single, m_de_ens]

print(f"\n{'Metric':<15} {labels[0]:<18} {labels[1]:<18} {labels[2]:<18} {labels[3]:<18}")
print("-" * 87)
for mk, mn in [('rmse','RMSE'),('mae','MAE'),('mape','MAPE'),('r2','R2')]:
    vals = [f'{m[mk]:.4f}' for m in all_m]
    print(f"{mn:<15} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18} {vals[3]:<18}")

print(f"\n{'Month':<6} {'LR':<9} {'RNN':<9} {'DE-best':<9} {'DE-ens':<9}")
print("-" * 42)
for mo in range(ol):
    print(f"{mo+1:<6} {m_lr['monthly_rmse'][mo]:<9.4f} {m_rnn['monthly_rmse'][mo]:<9.4f} "
          f"{m_de_single['monthly_rmse'][mo]:<9.4f} {m_de_ens['monthly_rmse'][mo]:<9.4f}")

print(f"\n{'':<15} {labels[0]:<18} {labels[1]:<18} {labels[2]:<18} {labels[3]:<18}")
de_params_count = count_parameters(SeaIceDualEncoderLSTM(
    input_size=1, main_hidden=cfg_de.HIDDEN_SIZE, aux_hidden=cfg_de.AUX_HIDDEN_SIZE,
    aux_input_size=7, aux_seq_len=aux_sl, num_layers=1,
    output_len=ol, dropout=cfg_de.DROPOUT, aux_dropout=cfg_de.AUX_DROPOUT))
print(f"{'Params':<15} {m_lr['params']:<18,} {m_rnn['params']:<18,} {de_params_count:<18,} {'5x'+str(de_params_count):<18}")
print(f"{'Epoch':<15} {m_lr['best_epoch']:<18} {m_rnn['best_epoch']:<18} {best_eps[best_idx]:<18} {sum(best_eps)/len(best_eps):<18.0f}")
print(f"{'Time':<15} {m_lr['train_time_s']:<17.1f}s {m_rnn['train_time_s']:<17.1f}s {train_times[best_idx]:<17.1f}s {sum(train_times):<17.1f}s")

rp = os.path.join(config.RESULTS_DIR, "e7_optimized_comparison.txt")
with open(rp, 'w') as f:
    f.write("E7 Optimized vs Baselines\n")
    f.write(f"E7 params: {bp}\n")
    f.write(f"Ensemble: {N_ENSEMBLE} seeds\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"{'Metric':<15} {'LR':<15} {'RNN':<15} {'DE-best':<15} {'DE-ens5':<15}\n")
    for mk, mn in [('rmse','RMSE'),('mae','MAE'),('mape','MAPE'),('r2','R2')]:
        vals = [f'{m[mk]:.4f}' for m in all_m]
        f.write(f"{mn:<15} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15} {vals[3]:<15}\n")
    f.write(f"\nMonthly RMSE:\n")
    f.write(f"{'Mo':<4} {'LR':<9} {'RNN':<9} {'DE-best':<9} {'DE-ens5':<9}\n")
    for mo in range(ol):
        f.write(f"{mo+1:<4} {m_lr['monthly_rmse'][mo]:<9.4f} {m_rnn['monthly_rmse'][mo]:<9.4f} "
                f"{m_de_single['monthly_rmse'][mo]:<9.4f} {m_de_ens['monthly_rmse'][mo]:<9.4f}\n")
print(f"Saved: {rp}")
print("\nDone!")
