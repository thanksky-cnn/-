# run_phase2_combinations.py
# Phase 2: Variable combination experiments
# Tests pairs and triples of the best-performing variables from Phase 1.
# Same dual-encoder architecture and hyperparameters as Phase 1.
import sys, os, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings; warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_preprocessing import (
    load_dual_encoder_data_v2, create_dual_targets_v2
)
from src.dataset import DualEncoderDataset
from src.model import SeaIceDualEncoderLSTM, count_parameters
from src.train import train_dual_encoder, predict_dual_encoder
from src.utils import set_seed, calculate_metrics

LAGGED_CSV = os.path.join(config.BASE_DIR, "data", "lagged_features_v2.csv")
RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "results", "phase2")
os.makedirs(RESULTS_DIR, exist_ok=True)

E7V1_HP = {
    "main_hidden": 256, "aux_hidden": 64, "aux_seq_len": 6,
    "num_layers": 1, "dropout": 0.1, "aux_dropout": 0.6,
    "batch_size": 32, "learning_rate": 0.000635, "weight_decay": 2.59e-06,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ol = config.OUTPUT_LEN; target = config.TARGET_COLUMN
N_ENSEMBLE = 5

# Phase 2: Combination experiments
# These should be updated based on Phase 1 results
# Format: aux_vars -> list of variable names to use together
COMBINATIONS = {
    # Two-variable combos
    "E15": {"aux_vars": ["ao", "t2m"],       "desc": "AO + T2M"},
    "E16": {"aux_vars": ["ao", "nao"],       "desc": "AO + NAO"},
    "E17": {"aux_vars": ["ao", "nino34"],    "desc": "AO + Nino3.4"},
    "E18": {"aux_vars": ["t2m", "slp"],      "desc": "T2M + SLP"},
    # Three-variable combos
    "E19": {"aux_vars": ["ao", "sst", "t2m"],     "desc": "AO + SST + T2M"},
    "E20": {"aux_vars": ["ao", "sst", "nao", "t2m"], "desc": "AO + SST + NAO + T2M"},
    # Top-3 from Phase 1 (update after Phase 1 results are in)
    "E21": {"aux_vars": ["ao", "sst"],        "desc": "Phase1-Top-N (placeholder)"},
    # Full ensemble
    "E22": {"aux_vars": ["ao", "sst", "nao", "nino34", "pdo", "t2m", "slp"],
            "desc": "ALL variables"},
}


def load_and_prepare_data(aux_vars):
    X_main, X_aux, df, scaler_ice, ch_names, var_cfg = load_dual_encoder_data_v2(
        LAGGED_CSV, aux_vars=aux_vars, target_column=target,
        aux_seq_len=E7V1_HP["aux_seq_len"]
    )
    y, years, months = create_dual_targets_v2(df, 12, ol, E7V1_HP["aux_seq_len"], target)
    n = min(len(X_main), len(y))
    X_main, X_aux, y, years = X_main[:n], X_aux[:n], y[:n], years[:n]
    months = months[:n]

    tr_mask = (years >= 1979) & (years <= 2010)
    v_mask  = (years >= 2011) & (years <= 2015)
    te_mask = (years >= 2016) & (years <= 2025)

    aux_seq = E7V1_HP["aux_seq_len"]
    return {
        "train": (X_main[tr_mask], X_aux[tr_mask][:, -aux_seq:, :], y[tr_mask]),
        "val":   (X_main[v_mask],  X_aux[v_mask][:, -aux_seq:, :],  y[v_mask]),
        "test":  (X_main[te_mask], X_aux[te_mask][:, -aux_seq:, :], y[te_mask]),
        "test_years": years[te_mask],
        "test_months": months[te_mask],
    }, scaler_ice, X_aux.shape[2]


def run_single_seed(exp_id, aux_vars, data_dict, scaler_ice, n_aux_ch, seed):
    set_seed(seed)
    hp = E7V1_HP.copy()

    tr_ds = DualEncoderDataset(*data_dict["train"])
    v_ds  = DualEncoderDataset(*data_dict["val"])
    tr_ldr = DataLoader(tr_ds, batch_size=hp["batch_size"], shuffle=True)
    v_ldr  = DataLoader(v_ds,  batch_size=hp["batch_size"])

    model = SeaIceDualEncoderLSTM(
        input_size=1, main_hidden=hp["main_hidden"],
        aux_hidden=hp["aux_hidden"], aux_input_size=n_aux_ch,
        aux_seq_len=hp["aux_seq_len"], num_layers=hp["num_layers"],
        output_len=ol, dropout=hp["dropout"], aux_dropout=hp["aux_dropout"],
    ).to(device)

    train_losses, val_losses, best_state, best_epoch, train_time = train_dual_encoder(
        model, tr_ldr, v_ldr, hp, device, num_epochs=1000,
        early_stopping_patience=30, verbose=False
    )

    y_pred = predict_dual_encoder(model, *data_dict["test"][:2], device, scaler_ice)
    y_true = scaler_ice.inverse_transform(
        data_dict["test"][2].reshape(-1, 1)
    ).reshape(-1, ol)

    metrics = calculate_metrics(y_true, y_pred)
    print(f"  [{exp_id}] seed={seed}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    return {
        "seed": seed, "best_epoch": best_epoch, "train_time_s": train_time,
        "y_pred": y_pred, "y_true": y_true, "metrics": metrics,
    }


def run_combination(exp_id, cfg, fast_mode=False):
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

    all_preds = np.stack([r["y_pred"] for r in results], axis=0)
    ensemble_pred = all_preds.mean(axis=0)
    ensemble_metrics = calculate_metrics(results[0]["y_true"], ensemble_pred)

    seed_rmses = [r["metrics"]["rmse"] for r in results]
    print(f"  {exp_id}: ensemble RMSE={ensemble_metrics['rmse']:.4f}")
    print(f"  Seeds: mean={np.mean(seed_rmses):.4f}, std={np.std(seed_rmses):.4f}")

    result = {
        "exp_id": exp_id, "desc": cfg["desc"], "aux_vars": cfg["aux_vars"],
        "n_aux_ch": n_aux_ch, "n_seeds": n_seeds,
        "seed_rmses": seed_rmses,
        "rmse_mean": float(np.mean(seed_rmses)),
        "rmse_std": float(np.std(seed_rmses)),
        "ensemble_rmse": float(ensemble_metrics["rmse"]),
        "ensemble_mae": float(ensemble_metrics["mae"]),
    }
    result_path = os.path.join(RESULTS_DIR, f"{exp_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Variable combination experiments")
    parser.add_argument("--exp", type=str, default="all")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for eid, cfg in COMBINATIONS.items():
            print(f"  {eid}: {cfg['desc']} [{', '.join(cfg['aux_vars'])}]")
        return

    to_run = list(COMBINATIONS.keys()) if args.exp == "all" else [e.strip() for e in args.exp.split(",")]

    all_results = {}
    for exp_id in to_run:
        if exp_id not in COMBINATIONS:
            continue
        r = run_combination(exp_id, COMBINATIONS[exp_id], fast_mode=args.fast)
        all_results[exp_id] = {
            "desc": r["desc"], "rmse_mean": r["rmse_mean"],
            "rmse_std": r["rmse_std"], "ensemble_rmse": r["ensemble_rmse"],
            "n_aux_ch": r["n_aux_ch"],
        }

    print(f"\n{'='*70}")
    print("  Phase 2 Results Summary")
    print(f"{'='*70}")
    for eid, r in all_results.items():
        print(f"  {eid}: {r['desc']:<35s}  RMSE={r['rmse_mean']:.4f}+/-{r['rmse_std']:.4f}  ch={r['n_aux_ch']}")

    summary_path = os.path.join(RESULTS_DIR, "phase2_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
