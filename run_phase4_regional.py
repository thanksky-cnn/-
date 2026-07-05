# run_phase4_regional.py
# Phase 4: Spatial heterogeneity experiments
# Tests matched vs mismatched climate indices for regional sea ice prediction.
import sys, os, json, argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import warnings; warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import DualEncoderDataset
from src.model import SeaIceDualEncoderLSTM, SeaIceLSTM, count_parameters
from src.train import train_dual_encoder, predict_dual_encoder
from src.utils import set_seed, calculate_metrics

BASE_DIR = config.BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "results", "phase4")
os.makedirs(RESULTS_DIR, exist_ok=True)

HP = {
    "main_hidden": 256, "aux_hidden": 64, "aux_seq_len": 6,
    "num_layers": 1, "dropout": 0.1, "aux_dropout": 0.6,
    "batch_size": 32, "learning_rate": 0.000635, "weight_decay": 2.59e-06,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_LEN = 12; OUTPUT_LEN = 12; N_ENSEMBLE = 5

# Experiments: region -> {baseline, matched_index, mismatched_index}
REGIONAL_EXPERIMENTS = {
    # --- 已有: 太平洋扇区 ---
    "E26":  {"region": "bering",  "aux": None,    "desc": "Bering Sea baseline"},
    "E26a": {"region": "bering",  "aux": ["pna"], "desc": "Bering + PNA (matched)"},
    "E26b": {"region": "bering",  "aux": ["nao"], "desc": "Bering + NAO (mismatched)"},
    # --- 已有: 大西洋扇区 ---
    "E27":  {"region": "barents", "aux": None,    "desc": "Barents Sea baseline"},
    "E27a": {"region": "barents", "aux": ["nao"], "desc": "Barents + NAO (matched)"},
    "E27b": {"region": "barents", "aux": ["pna"], "desc": "Barents + PNA (mismatched)"},
    # --- 已有: 局地核心区 ---
    "E28":  {"region": "central_arctic", "aux": None,    "desc": "Central Arctic baseline"},
    "E28a": {"region": "central_arctic", "aux": ["ao"],  "desc": "Central Arctic + AO"},
    "E28b": {"region": "central_arctic", "aux": ["sst"], "desc": "Central Arctic + SST"},
    # --- 新增: 楚科奇海 (太平洋扇区, PNA影响) ---
    "E29":  {"region": "chukchi",  "aux": None,    "desc": "Chukchi Sea baseline"},
    "E29a": {"region": "chukchi",  "aux": ["pna"], "desc": "Chukchi + PNA (matched)"},
    "E29b": {"region": "chukchi",  "aux": ["nao"], "desc": "Chukchi + NAO (mismatched)"},
    # --- 新增: 喀拉海 (大西洋扇区, NAO影响) ---
    "E30":  {"region": "kara",     "aux": None,    "desc": "Kara Sea baseline"},
    "E30a": {"region": "kara",     "aux": ["nao"], "desc": "Kara + NAO (matched)"},
    "E30b": {"region": "kara",     "aux": ["pna"], "desc": "Kara + PNA (mismatched)"},
    # --- 新增: 拉普捷夫海 (大西洋扇区, NAO/AO影响) ---
    "E31":  {"region": "laptev",   "aux": None,    "desc": "Laptev Sea baseline"},
    "E31a": {"region": "laptev",   "aux": ["nao"], "desc": "Laptev + NAO (matched)"},
    "E31b": {"region": "laptev",   "aux": ["pna"], "desc": "Laptev + PNA (mismatched)"},
    # --- 新增: 格陵兰海 (大西洋扇区, NAO控制Fram Strait) ---
    "E32":  {"region": "greenland","aux": None,    "desc": "Greenland Sea baseline"},
    "E32a": {"region": "greenland","aux": ["nao"], "desc": "Greenland + NAO (matched)"},
    "E32b": {"region": "greenland","aux": ["pna"], "desc": "Greenland + PNA (mismatched)"},
}

# Matched index per region (for summary labeling)
REGION_MATCHED = {
    "bering": "pna",
    "chukchi": "pna",
    "barents": "nao",
    "kara": "nao",
    "laptev": "nao",
    "greenland": "nao",
    "central_arctic": "ao",  # AO/SST for central Arctic
}


def load_regional_data(region_key, aux_vars):
    """Load regional ice area + climate indices, build sequences."""
    # Load regional ice area
    ice_csv = os.path.join(DATA_DIR, f"{region_key}_monthly.csv")
    df_ice = pd.read_csv(ice_csv)
    df_ice = df_ice[df_ice['area'] > 0].copy()  # filter -9999
    ice_raw = df_ice['area'].values.reshape(-1, 1)

    # Normalize ice
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
        # Univariate baseline: no aux encoder
        X_aux = np.zeros((len(X_main), HP["aux_seq_len"], 1), dtype=np.float32)
        n_aux_ch = 1
    else:
        # Load climate indices from lagged_features_v2.csv
        lagged_csv = os.path.join(DATA_DIR, "lagged_features_v2.csv")
        df_lag = pd.read_csv(lagged_csv)

        # Align time periods
        aux_channels = []
        for var in aux_vars:
            if var == "sst":
                aux_channels.extend(["sst", "sst_lag1", "sst_lag2", "sst_mask"])
            elif var == "nino34":
                aux_channels.extend(["nino34", "nino34_lag1", "nino34_lag2", "nino34_mask"])
            else:
                aux_channels.extend([var, f"{var}_lag1", f"{var}_lag2"])

        # Check columns exist
        available = [c for c in aux_channels if c in df_lag.columns]
        aux_raw = df_lag[available].values  # (N_lag, n_ch)

        # Normalize aux features independently (fit on all data including zeros)
        scaler_aux = MinMaxScaler(feature_range=(0, 1))
        aux_norm = scaler_aux.fit_transform(aux_raw)

        # Build aux sequences (last aux_seq_len months)
        n_total = len(aux_norm)
        aux_seq = HP["aux_seq_len"]
        X_aux_list = []
        for i in range(n_total - INPUT_LEN - OUTPUT_LEN + 1):
            idx = i + INPUT_LEN
            X_aux_list.append(aux_norm[idx - aux_seq:idx])

        X_aux = np.array(X_aux_list)
        n_aux_ch = X_aux.shape[2]

        # Trim to match ice data length
        n_samples = min(len(X_main), len(X_aux))
        X_main = X_main[:n_samples]
        X_aux = X_aux[:n_samples]
        years = years[:n_samples]

    # Year-based split
    tr_mask = (years >= 1979) & (years <= 2010)
    v_mask  = (years >= 2011) & (years <= 2015)
    te_mask = (years >= 2016) & (years <= 2025)

    # Build y (target: 12-month future ice area)
    y_list = []
    for i in range(len(ice_scaled) - INPUT_LEN - OUTPUT_LEN + 1):
        y_list.append(ice_scaled[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN])
    y_all = np.array(y_list)[:len(years)]

    data_dict = {
        "train": (X_main[tr_mask], X_aux[tr_mask], y_all[tr_mask]),
        "val":   (X_main[v_mask],  X_aux[v_mask],  y_all[v_mask]),
        "test":  (X_main[te_mask], X_aux[te_mask], y_all[te_mask]),
        "test_years": years[te_mask],
    }
    return data_dict, scaler_ice, n_aux_ch


def run_single_seed(exp_id, data_dict, scaler_ice, n_aux_ch, seed):
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
        early_stopping_patience=30, verbose=False
    )

    y_pred = predict_dual_encoder(model, *data_dict["test"][:2], device, scaler_ice)
    y_true = scaler_ice.inverse_transform(
        data_dict["test"][2].reshape(-1, 1)
    ).reshape(-1, OUTPUT_LEN)

    metrics = calculate_metrics(y_true, y_pred)
    print(f"  [{exp_id}] seed={seed}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    return {"seed": seed, "best_epoch": best_epoch, "train_time_s": train_time,
            "y_pred": y_pred, "y_true": y_true, "metrics": metrics}


def run_experiment(exp_id, cfg, fast_mode=False):
    n_seeds = 1 if fast_mode else N_ENSEMBLE
    aux_vars = cfg["aux"]
    aux_label = "none" if aux_vars is None else "+".join(aux_vars)
    print(f"\n{'='*60}")
    print(f"  {exp_id}: {cfg['desc']}")
    print(f"  Region: {cfg['region']}, Aux: {aux_label}, Seeds: {n_seeds}")
    print(f"{'='*60}")

    data_dict, scaler_ice, n_aux_ch = load_regional_data(cfg["region"], aux_vars)

    seeds = [42 + i * 10 for i in range(n_seeds)]
    results = []
    for seed in seeds:
        r = run_single_seed(exp_id, data_dict, scaler_ice, n_aux_ch, seed)
        results.append(r)

    # Ensemble
    all_preds = np.stack([r["y_pred"] for r in results], axis=0)
    ensemble_pred = all_preds.mean(axis=0)
    y_true = results[0]["y_true"]
    ensemble_metrics = calculate_metrics(y_true, ensemble_pred)

    seed_rmses = [r["metrics"]["rmse"] for r in results]
    rmse_mean = float(np.mean(seed_rmses))
    rmse_std = float(np.std(seed_rmses))

    print(f"  {exp_id} seeds: RMSE={[f'{r:.4f}' for r in seed_rmses]}")
    print(f"  {exp_id} ensemble: RMSE={ensemble_metrics['rmse']:.4f}, MAE={ensemble_metrics['mae']:.4f}")

    result = {
        "exp_id": exp_id, "desc": cfg["desc"], "region": cfg["region"],
        "aux_vars": aux_vars, "n_aux_ch": n_aux_ch, "n_seeds": n_seeds,
        "seed_rmses": [float(r) for r in seed_rmses],
        "rmse_mean": rmse_mean, "rmse_std": rmse_std,
        "ensemble_rmse": float(ensemble_metrics["rmse"]),
        "ensemble_mae": float(ensemble_metrics["mae"]),
    }
    result_path = os.path.join(RESULTS_DIR, f"{exp_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Regional spatial heterogeneity")
    parser.add_argument("--exp", type=str, default="all")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for eid, cfg in REGIONAL_EXPERIMENTS.items():
            aux = cfg["aux"] if cfg["aux"] else "(none)"
            print(f"  {eid}: {cfg['desc']} [{cfg['region']}, aux={aux}]")
        return

    to_run = list(REGIONAL_EXPERIMENTS.keys()) if args.exp == "all" \
             else [e.strip() for e in args.exp.split(",")]

    print(f"Device: {device}, Output len: {OUTPUT_LEN}")
    print(f"Running {len(to_run)} experiments: {to_run}")
    if args.fast:
        print("FAST MODE: single seed only")

    all_results = {}
    for exp_id in to_run:
        if exp_id not in REGIONAL_EXPERIMENTS:
            continue
        r = run_experiment(exp_id, REGIONAL_EXPERIMENTS[exp_id], fast_mode=args.fast)
        all_results[exp_id] = r

    # Summary
    print(f"\n{'='*80}")
    print("  Phase 4: Regional Spatial Heterogeneity Results")
    print(f"{'='*80}")

    # Group by region for matched vs mismatched comparison
    for region in ["bering", "chukchi", "barents", "kara", "laptev", "greenland", "central_arctic"]:
        region_exps = {eid: r for eid, r in all_results.items()
                       if r["region"] == region}
        if not region_exps:
            continue
        print(f"\n  [{region}]")
        base_rmse = None
        for eid, r in region_exps.items():
            aux_lbl = "+".join(r["aux_vars"]) if r["aux_vars"] else "none"
            line = f"    {eid}: RMSE={r['ensemble_rmse']:.6f}  (aux={aux_lbl})"
            if r["aux_vars"] is None:
                base_rmse = r["ensemble_rmse"]
            elif base_rmse:
                delta = r["ensemble_rmse"] - base_rmse
                matched_idx = REGION_MATCHED.get(region, "")
                is_matched = any(v == matched_idx for v in r["aux_vars"])
                match_type = "matched" if is_matched else "mismatched"
                line += f"  Δ={delta:+.6f} [{match_type}]"
            print(line)

    # Key hypothesis test
    print(f"\n{'='*80}")
    print("  Spatial Heterogeneity Hypothesis Test")
    print(f"{'='*80}")

    # Test all matched-vs-mismatched pairs
    hypothesis_tests = {
        "Pacific": [
            ("E26a", "E26b", "Bering: PNA(matched) vs NAO(mismatched)"),
            ("E29a", "E29b", "Chukchi: PNA(matched) vs NAO(mismatched)"),
        ],
        "Atlantic": [
            ("E27a", "E27b", "Barents: NAO(matched) vs PNA(mismatched)"),
            ("E30a", "E30b", "Kara: NAO(matched) vs PNA(mismatched)"),
            ("E31a", "E31b", "Laptev: NAO(matched) vs PNA(mismatched)"),
            ("E32a", "E32b", "Greenland: NAO(matched) vs PNA(mismatched)"),
        ],
    }

    pacific_support = 0; pacific_total = 0
    atlantic_support = 0; atlantic_total = 0

    for sector, tests in hypothesis_tests.items():
        for e_matched, e_mismatched, label in tests:
            if e_matched in all_results and e_mismatched in all_results:
                rmse_matched = all_results[e_matched]["ensemble_rmse"]
                rmse_mismatched = all_results[e_mismatched]["ensemble_rmse"]
                # Gain = baseline - rmse, larger is better
                # For matched to win: rmse_matched < rmse_mismatched
                better = "matched" if rmse_matched < rmse_mismatched else "mismatched"
                print(f"  {label}: matched={rmse_matched:.6f}, mismatched={rmse_mismatched:.6f} → {better} wins")
                if sector == "Pacific":
                    pacific_total += 1
                    if rmse_matched < rmse_mismatched:
                        pacific_support += 1
                else:
                    atlantic_total += 1
                    if rmse_matched < rmse_mismatched:
                        atlantic_support += 1

    print(f"\n  Pacific sector: {pacific_support}/{pacific_total} regions support matched>mismatched")
    print(f"  Atlantic sector: {atlantic_support}/{atlantic_total} regions support matched>mismatched")
    overall = pacific_support + atlantic_support
    overall_total = pacific_total + atlantic_total
    print(f"  Overall: {overall}/{overall_total} regions support spatial heterogeneity hypothesis")
    if overall >= overall_total * 0.6:
        print(f"  ✅ Spatial heterogeneity hypothesis broadly supported")
    else:
        print(f"  ⚠ Spatial heterogeneity hypothesis not consistently supported — see regional discussion")

    # Save summary
    summary = {eid: {"desc": r["desc"], "region": r["region"],
                     "ensemble_rmse": r["ensemble_rmse"],
                     "rmse_mean": r["rmse_mean"]}
               for eid, r in all_results.items()}
    summary_path = os.path.join(RESULTS_DIR, "phase4_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
