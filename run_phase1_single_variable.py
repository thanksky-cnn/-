# run_phase1_single_variable.py
# Phase 1: Single-variable incremental experiments
# Each experiment adds ONE climate variable via dual-encoder.
# 5-seed ensemble per experiment. Uniform hyperparameters from E7v1.
import sys, os, json, time, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import warnings; warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_preprocessing import (
    load_dual_encoder_data_v2, create_dual_targets_v2,
    load_and_merge_data, create_sequences
)
from src.dataset import SeaIceDataset, DualEncoderDataset
from src.model import SeaIceDualEncoderLSTM, SeaIceLSTM, count_parameters
from src.train import train_dual_encoder, predict_dual_encoder
from src.utils import set_seed, calculate_metrics

LAGGED_CSV = os.path.join(config.BASE_DIR, "data", "lagged_features_v2.csv")
RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "results", "phase1")
os.makedirs(RESULTS_DIR, exist_ok=True)

E7V1_HP = {
    "main_hidden": 256, "aux_hidden": 64, "aux_seq_len": 6,
    "num_layers": 1, "dropout": 0.1, "aux_dropout": 0.6,
    "batch_size": 32, "learning_rate": 0.000635, "weight_decay": 2.59e-06,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ol = config.OUTPUT_LEN; target = config.TARGET_COLUMN
N_ENSEMBLE = 5

EXPERIMENTS = {
    "E8":  {"aux_vars": ["nao"],       "desc": "DE (ice + NAO)"},
    "E9":  {"aux_vars": ["nino34"],    "desc": "DE (ice + Nino3.4)"},
    "E10": {"aux_vars": ["pdo"],       "desc": "DE (ice + PDO)"},
    "E11": {"aux_vars": ["t2m"],       "desc": "DE (ice + T2M)"},
    "E12": {"aux_vars": ["slp"],       "desc": "DE (ice + SLP)"},
    "E13": {"aux_vars": ["lag12_ice"], "desc": "DE (ice + Lag-12 ice)"},
    "E14": {"aux_vars": ["ao"],        "desc": "DE (ice + AO only)"},
    "E7v1_ref": {"aux_vars": ["ao", "sst"], "desc": "DE (ice + AO/SST) ref"},
}

print(f"Device: {device}, Output len: {ol}, Target: {target}")


def load_and_prepare_data(aux_vars):
    """Load v2 data, build sequences, split by year. Returns train/val/test sets."""
    X_main, X_aux, df, scaler_ice, ch_names, var_cfg = load_dual_encoder_data_v2(
        LAGGED_CSV, aux_vars=aux_vars, target_column=target,
        aux_seq_len=E7V1_HP["aux_seq_len"]
    )
    y, years, months = create_dual_targets_v2(
        df, 12, ol, E7V1_HP["aux_seq_len"], target
    )
    n = min(len(X_main), len(y))
    X_main, X_aux, y, years = X_main[:n], X_aux[:n], y[:n], years[:n]
    months = months[:n]

    # Year-based split
    tr_mask = (years >= 1979) & (years <= 2010)
    v_mask  = (years >= 2011) & (years <= 2015)
    te_mask = (years >= 2016) & (years <= 2025)

    n_aux_ch = X_aux.shape[2]
    aux_seq = E7V1_HP["aux_seq_len"]
    return {
        "train": (X_main[tr_mask], X_aux[tr_mask][:, -aux_seq:, :], y[tr_mask]),
        "val":   (X_main[v_mask],  X_aux[v_mask][:, -aux_seq:, :],  y[v_mask]),
        "test":  (X_main[te_mask], X_aux[te_mask][:, -aux_seq:, :], y[te_mask]),
        "test_years": years[te_mask],
        "test_months": months[te_mask],
    }, scaler_ice, n_aux_ch


def run_single_seed(exp_id, aux_vars, data_dict, scaler_ice, n_aux_ch, seed):
    """Train one seed of a dual-encoder experiment. Returns (model, preds, true, metrics)."""
    set_seed(seed)
    hp = E7V1_HP.copy()

    # Create datasets
    tr_ds = DualEncoderDataset(*data_dict["train"])
    v_ds  = DualEncoderDataset(*data_dict["val"])
    te_ds = DualEncoderDataset(*data_dict["test"])

    tr_ldr = DataLoader(tr_ds, batch_size=hp["batch_size"], shuffle=True)
    v_ldr  = DataLoader(v_ds,  batch_size=hp["batch_size"])

    # Build model
    model = SeaIceDualEncoderLSTM(
        input_size=1, main_hidden=hp["main_hidden"],
        aux_hidden=hp["aux_hidden"], aux_input_size=n_aux_ch,
        aux_seq_len=hp["aux_seq_len"], num_layers=hp["num_layers"],
        output_len=ol, dropout=hp["dropout"], aux_dropout=hp["aux_dropout"],
    ).to(device)

    print(f"  [{exp_id}] seed={seed}, params={count_parameters(model):,}, aux_ch={n_aux_ch}")

    # Train
    train_losses, val_losses, best_state, best_epoch, train_time = train_dual_encoder(
        model, tr_ldr, v_ldr, hp, device,
        num_epochs=1000, early_stopping_patience=30, verbose=False
    )

    # Predict
    y_pred = predict_dual_encoder(model, *data_dict["test"][:2], device, scaler_ice)
    y_true = scaler_ice.inverse_transform(
        data_dict["test"][2].reshape(-1, 1)
    ).reshape(-1, ol)

    metrics = calculate_metrics(y_true, y_pred)
    print(f"  [{exp_id}] seed={seed}: RMSE={metrics['rmse']:.4f}, "
          f"MAE={metrics['mae']:.4f}, best_ep={best_epoch}, time={train_time:.0f}s")

    return {
        "seed": seed, "best_epoch": best_epoch, "train_time_s": train_time,
        "y_pred": y_pred, "y_true": y_true, "metrics": metrics,
        "train_losses": train_losses, "val_losses": val_losses,
    }


def run_experiment(exp_id, cfg, fast_mode=False):
    """Run one experiment with 5-seed ensemble (or 1-seed in fast mode)."""
    n_seeds = 1 if fast_mode else N_ENSEMBLE
    print(f"\n{'='*60}")
    print(f"  {exp_id}: {cfg['desc']}")
    print(f"  Aux vars: {cfg['aux_vars']}, Seeds: {n_seeds}")
    print(f"{'='*60}")

    data_dict, scaler_ice, n_aux_ch = load_and_prepare_data(cfg["aux_vars"])

    seeds = [42 + i * 10 for i in range(n_seeds)]
    results = []
    for seed in seeds:
        r = run_single_seed(exp_id, cfg["aux_vars"], data_dict, scaler_ice, n_aux_ch, seed)
        results.append(r)

    # Ensemble
    all_preds = np.stack([r["y_pred"] for r in results], axis=0)  # (n_seeds, N, ol)
    ensemble_pred = all_preds.mean(axis=0)
    y_true = results[0]["y_true"]
    ensemble_metrics = calculate_metrics(y_true, ensemble_pred)

    # Individual seed stats
    seed_rmses = [r["metrics"]["rmse"] for r in results]
    rmse_mean = np.mean(seed_rmses)
    rmse_std = np.std(seed_rmses)

    print(f"  {exp_id} seeds: RMSE={seed_rmses}")
    print(f"  {exp_id} ensemble: RMSE={ensemble_metrics['rmse']:.4f}, "
          f"MAE={ensemble_metrics['mae']:.4f}")
    print(f"  {exp_id} seed stats: mean={rmse_mean:.4f}, std={rmse_std:.4f}")

    # Save
    result = {
        "exp_id": exp_id, "desc": cfg["desc"], "aux_vars": cfg["aux_vars"],
        "n_aux_ch": n_aux_ch, "n_seeds": n_seeds,
        "seed_rmses": seed_rmses, "rmse_mean": rmse_mean, "rmse_std": rmse_std,
        "ensemble_rmse": ensemble_metrics["rmse"],
        "ensemble_mae": ensemble_metrics["mae"],
        "ensemble_mape": ensemble_metrics.get("mape", None),
        "test_years": data_dict["test_years"].tolist(),
        "test_months": data_dict["test_months"].tolist() if hasattr(data_dict["test_months"], 'tolist') else [],
    }
    result_path = os.path.join(RESULTS_DIR, f"{exp_id}.json")
    # Strip non-serializable fields
    save_result = {k: v for k, v in result.items()
                   if not k.startswith("y_")}
    # Convert numpy arrays
    for k in save_result:
        if isinstance(save_result[k], np.ndarray):
            save_result[k] = save_result[k].tolist()
        elif isinstance(save_result[k], list) and save_result[k] and isinstance(save_result[k][0], np.floating):
            save_result[k] = [float(x) for x in save_result[k]]
    with open(result_path, "w") as f:
        json.dump(save_result, f, indent=2)
    print(f"  Saved: {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Single-variable experiments")
    parser.add_argument("--exp", type=str, default="all",
                        help="Comma-separated experiment IDs (e.g., E8,E9)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: single seed, 100 epochs")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments and exit")
    args = parser.parse_args()

    if args.list:
        for eid, cfg in EXPERIMENTS.items():
            print(f"  {eid}: {cfg['desc']} [{', '.join(cfg['aux_vars'])}]")
        return

    if args.exp == "all":
        to_run = list(EXPERIMENTS.keys())
    else:
        to_run = [e.strip() for e in args.exp.split(",")]
        for e in to_run:
            if e not in EXPERIMENTS:
                print(f"Unknown experiment: {e}, skipping")

    print(f"Running {len(to_run)} experiments: {to_run}")
    if args.fast:
        print("FAST MODE: single seed only")

    all_results = {}
    for exp_id in to_run:
        if exp_id not in EXPERIMENTS:
            continue
        r = run_experiment(exp_id, EXPERIMENTS[exp_id], fast_mode=args.fast)
        all_results[exp_id] = {
            "desc": r["desc"], "rmse_mean": r["rmse_mean"],
            "rmse_std": r["rmse_std"], "ensemble_rmse": r["ensemble_rmse"],
        }

    # Summary table
    print(f"\n{'='*70}")
    print("  Phase 1 Results Summary")
    print(f"{'='*70}")
    print(f"  {'Exp':<10s} {'Description':<35s} {'RMSE':>8s} {'Std':>8s}")
    print(f"  {'-'*60}")
    for eid, r in all_results.items():
        print(f"  {eid:<10s} {r['desc']:<35s} {r['rmse_mean']:8.4f} {r['rmse_std']:8.4f}")
    print(f"  {'='*70}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "phase1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
