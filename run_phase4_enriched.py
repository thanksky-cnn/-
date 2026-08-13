# run_phase4_enriched.py
# Phase 4 enriched: Per-region 4-index full scan with self-attention dual-encoder.
# Design: 7 regions × 5 configs (baseline + AO + NAO + PNA + Nino3.4) × 5 seeds
# Total: 175 training runs (one per config/seed)
import sys, os, json, argparse, time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import warnings; warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import DualEncoderDataset
from src.model import SeaIceDualEncoderLSTM, count_parameters
from src.train import train_dual_encoder, predict_dual_encoder
from src.utils import set_seed, calculate_metrics

BASE_DIR = config.BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "results", "phase4_enriched")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hypermeters (unified, kept for Phase 1-3 comparison) ──
HP = {
    "main_hidden": 256, "aux_hidden": 64, "aux_seq_len": 6,
    "num_layers": 1, "dropout": 0.1, "aux_dropout": 0.6,
    "batch_size": 32, "learning_rate": 0.000635, "weight_decay": 2.59e-06,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
INPUT_LEN = 12; OUTPUT_LEN = 12
N_SEEDS = 5
SEEDS = [42, 52, 62, 72, 82]

# ── Climate indices (without SST) ──
ALL_INDICES = ["ao", "nao", "pna", "nino34"]
INDEX_NAMES = {"ao": "AO", "nao": "NAO", "pna": "PNA", "nino34": "Nino3.4"}

# ── 7 Regions ──
REGIONS = [
    {"key": "bering",         "name": "白令海",       "sector": "Pacific"},
    {"key": "chukchi",        "name": "楚科奇海",     "sector": "Pacific"},
    {"key": "barents",        "name": "巴伦支海",     "sector": "Atlantic"},
    {"key": "kara",           "name": "喀拉海",       "sector": "Atlantic"},
    {"key": "laptev",         "name": "拉普捷夫海",   "sector": "Atlantic"},
    {"key": "greenland",      "name": "格陵兰海",     "sector": "Atlantic"},
    {"key": "central_arctic", "name": "中北冰洋",     "sector": "Core"},
]

# ── Experiment matrix ──
# Per region: baseline (no aux) + 4 single-variable configs
def build_experiments():
    """Build list of all experiments."""
    exps = []
    eid = 0
    for region in REGIONS:
        rk = region["key"]
        # Baseline (no aux)
        eid += 1
        exps.append({
            "exp_id": f"E{eid:03d}",
            "region": rk, "region_name": region["name"],
            "aux_vars": None, "aux_label": "纯冰基线",
            "type": "baseline",
        })
        # Single-variable
        for idx in ALL_INDICES:
            eid += 1
            exps.append({
                "exp_id": f"E{eid:03d}",
                "region": rk, "region_name": region["name"],
                "aux_vars": [idx], "aux_label": f"+{INDEX_NAMES[idx]}",
                "type": "single",
            })
    return exps

EXPERIMENTS = build_experiments()
print(f"Total experiments: {len(EXPERIMENTS)} (5 configs × 7 regions)")
print(f"Total training runs: {len(EXPERIMENTS) * N_SEEDS} ({N_SEEDS} seeds each)")


def load_regional_data(region_key, aux_vars):
    """Load regional ice area + climate indices, build sequences."""
    ice_csv = os.path.join(DATA_DIR, f"{region_key}_monthly.csv")
    df_ice = pd.read_csv(ice_csv)
    # Filter valid data
    df_ice = df_ice[df_ice['area'] > -0.005].copy()
    ice_raw = df_ice['area'].values.reshape(-1, 1)

    scaler_ice = MinMaxScaler(feature_range=(0, 1))
    ice_scaled = scaler_ice.fit_transform(ice_raw).flatten()

    # Build X_main (12-month ice sequences)
    n = len(ice_scaled)
    X_main_list, year_list = [], []
    for i in range(n - INPUT_LEN - OUTPUT_LEN + 1):
        X_main_list.append(ice_scaled[i:i + INPUT_LEN])
        year_list.append(df_ice['year'].values[i + INPUT_LEN])
    X_main = np.array(X_main_list).reshape(-1, INPUT_LEN, 1)
    years = np.array(year_list)

    if aux_vars is None:
        # Baseline
        X_aux = np.zeros((len(X_main), HP["aux_seq_len"], 1), dtype=np.float32)
        n_aux_ch = 1
    else:
        lagged_csv = os.path.join(DATA_DIR, "lagged_features_v2.csv")
        df_lag = pd.read_csv(lagged_csv)

        aux_channels = []
        for var in aux_vars:
            if var == "nino34":
                aux_channels.extend(["nino34", "nino34_lag1", "nino34_lag2", "nino34_mask"])
            else:
                aux_channels.extend([var, f"{var}_lag1", f"{var}_lag2"])

        available = [c for c in aux_channels if c in df_lag.columns]
        aux_raw = df_lag[available].values

        scaler_aux = MinMaxScaler(feature_range=(0, 1))
        aux_norm = scaler_aux.fit_transform(aux_raw)

        aux_seq = HP["aux_seq_len"]
        X_aux_list = []
        for i in range(len(aux_norm) - INPUT_LEN - OUTPUT_LEN + 1):
            idx = i + INPUT_LEN
            X_aux_list.append(aux_norm[idx - aux_seq:idx])
        X_aux = np.array(X_aux_list)
        n_aux_ch = X_aux.shape[2]

        n_samples = min(len(X_main), len(X_aux))
        X_main = X_main[:n_samples]; X_aux = X_aux[:n_samples]; years = years[:n_samples]

    # Year-based split: train 1979-2010, val 2011-2015, test 2016-2025
    tr_mask = (years >= 1979) & (years <= 2010)
    v_mask  = (years >= 2011) & (years <= 2015)
    te_mask = (years >= 2016) & (years <= 2025)

    y_list = []
    for i in range(len(ice_scaled) - INPUT_LEN - OUTPUT_LEN + 1):
        y_list.append(ice_scaled[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN])
    y_all = np.array(y_list)[:len(years)]

    return {
        "train": (X_main[tr_mask], X_aux[tr_mask], y_all[tr_mask]),
        "val":   (X_main[v_mask],  X_aux[v_mask],  y_all[v_mask]),
        "test":  (X_main[te_mask], X_aux[te_mask], y_all[te_mask]),
        "test_years": years[te_mask],
        "df_months": df_ice['month'].values,
    }, scaler_ice, n_aux_ch


def run_single_seed(exp_id, data_dict, scaler_ice, n_aux_ch, seed):
    """Train one seed of one experiment config."""
    set_seed(seed)
    hp = HP.copy()

    tr_ds = DualEncoderDataset(*data_dict["train"])
    v_ds  = DualEncoderDataset(*data_dict["val"])
    tr_ldr = DataLoader(tr_ds, batch_size=hp["batch_size"], shuffle=True)
    v_ldr  = DataLoader(v_ds,  batch_size=hp["batch_size"])

    model = SeaIceDualEncoderLSTM(
        input_size=1, main_hidden=hp["main_hidden"],
        aux_hidden=hp["aux_hidden"], aux_input_size=n_aux_ch,
        aux_seq_len=hp["aux_seq_len"], num_layers=hp["num_layers"],
        output_len=OUTPUT_LEN, dropout=hp["dropout"], aux_dropout=hp["aux_dropout"],
    ).to(device)

    train_losses, val_losses, best_state, best_epoch, train_time = train_dual_encoder(
        model, tr_ldr, v_ldr, hp, device, num_epochs=1000,
        early_stopping_patience=30, verbose=False)

    model.load_state_dict(best_state)
    y_pred = predict_dual_encoder(
        model, data_dict["test"][0], data_dict["test"][1], device, scaler=scaler_ice)
    y_true = scaler_ice.inverse_transform(
        data_dict["test"][2].reshape(-1, 1)).reshape(data_dict["test"][2].shape)

    metrics = calculate_metrics(y_true, y_pred)

    # Extract attention weights (mean over test batch)
    attn_weights = None
    try:
        aw = model.get_attention_weights()
        if aw is not None:
            attn_weights = aw.cpu().numpy().mean(axis=0).tolist()
    except:
        pass

    return {
        "seed": seed, "rmse": float(metrics['rmse']), "mae": float(metrics['mae']), "mape": float(metrics['mape']),
        "train_time_s": train_time, "best_epoch": best_epoch,
        "y_pred": y_pred.tolist(), "y_true": y_true.tolist(),
        "attn_weights": attn_weights,
    }


def run_experiment(exp, fast_mode=False):
    """Run one experiment config with ensemble."""
    exp_id = exp["exp_id"]
    region_key = exp["region"]
    aux_vars = exp["aux_vars"]

    print(f"\n{'='*60}")
    print(f"  {exp_id}: {exp['region_name']} ({region_key}) — {exp['aux_label']}")
    print(f"{'='*60}")

    data_dict, scaler_ice, n_aux_ch = load_regional_data(region_key, aux_vars)
    print(f"  Train samples: {len(data_dict['train'][0])}, "
          f"Val: {len(data_dict['val'][0])}, Test: {len(data_dict['test'][0])}")
    print(f"  Aux channels: {n_aux_ch}")

    seeds_to_run = SEEDS[:1] if fast_mode else SEEDS
    seed_results = []
    for seed in seeds_to_run:
        t0 = time.time()
        result = run_single_seed(exp_id, data_dict, scaler_ice, n_aux_ch, seed)
        elapsed = time.time() - t0
        print(f"  seed={seed:2d}  rmse={result['rmse']:.6f}  "
              f"epoch={result['best_epoch']:3d}  time={elapsed:.0f}s")
        seed_results.append(result)

    # Ensemble prediction
    all_preds = np.array([r["y_pred"] for r in seed_results])  # (S, N, 12)
    ensemble_pred = all_preds.mean(axis=0)  # (N, 12)
    y_true = np.array(seed_results[0]["y_true"])

    ensemble_rmse = float(np.sqrt(np.mean((y_true - ensemble_pred) ** 2)))
    ensemble_mae  = float(np.mean(np.abs(y_true - ensemble_pred)))

    rmse_list = [r["rmse"] for r in seed_results]
    rmse_mean = float(np.mean(rmse_list))
    rmse_std  = float(np.std(rmse_list, ddof=1)) if len(rmse_list) > 1 else 0.0

    # Ensemble attention weights
    all_attn = [r["attn_weights"] for r in seed_results if r["attn_weights"] is not None]
    ensemble_attn = None
    if all_attn and len(all_attn) == len(seed_results):
        ensemble_attn = np.mean(all_attn, axis=0).tolist()

    summary = {
        "exp_id": exp_id,
        "region": region_key, "region_name": exp["region_name"],
        "aux_vars": aux_vars, "aux_label": exp["aux_label"], "type": exp["type"],
        "n_aux_ch": n_aux_ch, "n_seeds": len(seed_results),
        "rmse_mean": rmse_mean, "rmse_std": rmse_std,
        "ensemble_rmse": ensemble_rmse, "ensemble_mae": ensemble_mae,
        "ens_attn_weights": ensemble_attn,
        "hp": HP,
    }

    # Save
    out_path = os.path.join(RESULTS_DIR, f"{exp_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  → {out_path}")
    print(f"  ensemble_rmse={ensemble_rmse:.6f}  rmse_mean={rmse_mean:.6f}±{rmse_std:.6f}")

    return summary


def run_phase4_enriched(args):
    """Run all Phase 4 enriched experiments."""
    all_summaries = []

    exps = EXPERIMENTS
    skip_until = getattr(args, 'resume_from', None)

    if args.exp_id:
        exps = [e for e in exps if e["exp_id"] == args.exp_id]

    if args.region:
        exps = [e for e in exps if e["region"] == args.region]

    resume = skip_until is not None
    for exp in exps:
        if resume:
            if exp["exp_id"] == skip_until:
                resume = False
            else:
                print(f"  SKIP {exp['exp_id']} (resume from {skip_until})")
                continue

        summary = run_experiment(exp, fast_mode=args.fast)
        all_summaries.append(summary)

    # Save combined summary
    if not args.exp_id:
        summary_path = os.path.join(RESULTS_DIR, "phase4_enriched_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"  All {len(all_summaries)} experiments complete.")
        print(f"  Summary: {summary_path}")

        # Print ranking
        print(f"\n  Top-10 by ensemble RMSE:")
        for i, s in enumerate(sorted(all_summaries, key=lambda x: x["ensemble_rmse"])[:10]):
            print(f"  {i+1:2d}. {s['exp_id']} {s['region_name']:6s} {s['aux_label']:12s}  "
                  f"RMSE={s['ensemble_rmse']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Enriched: Per-region 4-index full scan")
    parser.add_argument("--exp-id", type=str, default=None, help="Run single experiment (e.g. E001)")
    parser.add_argument("--region", type=str, default=None, help="Run all experiments for one region")
    parser.add_argument("--fast", action="store_true", help="1-seed fast mode")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from experiment ID")
    args = parser.parse_args()

    if args.exp_id:
        print(f"Single experiment mode: {args.exp_id}")
    elif args.region:
        print(f"Region mode: {args.region}")
    else:
        print(f"Full Phase 4 enriched: {len(EXPERIMENTS)} experiments × {N_SEEDS} seeds")

    run_phase4_enriched(args)
