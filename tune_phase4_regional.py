# tune_phase4_regional.py
# Phase 4 补充: 对7个海域各自最优aux配置进行Optuna调参
# 验证统一超参数是否压制了区域气候指数的真实潜力
import sys, os, json, argparse
import numpy as np
import pandas as pd
import torch
import optuna
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
RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "results", "phase4")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 统一超参数 (E7v1 Optuna最优, 对照组)
E7V1_HP = {
    "main_hidden": 256, "aux_hidden": 64, "aux_seq_len": 6,
    "num_layers": 1, "dropout": 0.1, "aux_dropout": 0.6,
    "batch_size": 32, "learning_rate": 0.000635, "weight_decay": 2.59e-06,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_LEN = 12; OUTPUT_LEN = 12; N_ENSEMBLE = 5

# 7个海域各自的最优aux (基于Phase 4实验结果, 选择RMSE更低的那个aux)
REGION_TUNE_CONFIGS = {
    "bering":         {"aux": ["nao"],  "desc": "Bering + NAO",       "baseline_rmse": 0.101373, "best_aux_rmse": 0.099568},
    "chukchi":        {"aux": ["pna"],  "desc": "Chukchi + PNA",      "baseline_rmse": 0.080291, "best_aux_rmse": 0.080671},
    "barents":        {"aux": ["nao"],  "desc": "Barents + NAO",      "baseline_rmse": 0.100027, "best_aux_rmse": 0.098312},
    "kara":           {"aux": ["pna"],  "desc": "Kara + PNA",         "baseline_rmse": 0.117149, "best_aux_rmse": 0.117487},
    "laptev":         {"aux": ["pna"],  "desc": "Laptev + PNA",       "baseline_rmse": 0.115744, "best_aux_rmse": 0.114748},
    "greenland":      {"aux": ["pna"],  "desc": "Greenland + PNA",    "baseline_rmse": 0.070881, "best_aux_rmse": 0.071998},
    "central_arctic": {"aux": ["ao"],   "desc": "Central Arctic + AO","baseline_rmse": 0.143547, "best_aux_rmse": 0.143515},
}


def load_regional_data(region_key, aux_vars):
    """与 run_phase4_regional.py 完全相同的加载逻辑"""
    ice_csv = os.path.join(DATA_DIR, f"{region_key}_monthly.csv")
    df_ice = pd.read_csv(ice_csv)
    df_ice = df_ice[df_ice['area'] > 0].copy()
    ice_raw = df_ice['area'].values.reshape(-1, 1)

    scaler_ice = MinMaxScaler(feature_range=(0, 1))
    ice_scaled = scaler_ice.fit_transform(ice_raw).flatten()

    n = len(ice_scaled)
    X_main_list, year_list = [], []
    for i in range(n - INPUT_LEN - OUTPUT_LEN + 1):
        X_main_list.append(ice_scaled[i:i + INPUT_LEN])
        year_list.append(df_ice['year'].values[i + INPUT_LEN])
    X_main = np.array(X_main_list).reshape(-1, INPUT_LEN, 1)
    years = np.array(year_list)

    # Load climate indices
    lagged_csv = os.path.join(DATA_DIR, "lagged_features_v2.csv")
    df_lag = pd.read_csv(lagged_csv)

    aux_channels = []
    for var in aux_vars:
        if var == "sst":
            aux_channels.extend(["sst", "sst_lag1", "sst_lag2", "sst_mask"])
        elif var == "nino34":
            aux_channels.extend(["nino34", "nino34_lag1", "nino34_lag2", "nino34_mask"])
        else:
            aux_channels.extend([var, f"{var}_lag1", f"{var}_lag2"])

    available = [c for c in aux_channels if c in df_lag.columns]
    aux_raw = df_lag[available].values

    scaler_aux = MinMaxScaler(feature_range=(0, 1))
    aux_norm = scaler_aux.fit_transform(aux_raw)

    aux_seq = E7V1_HP["aux_seq_len"]
    n_total = len(aux_norm)
    X_aux_list = []
    for i in range(n_total - INPUT_LEN - OUTPUT_LEN + 1):
        idx = i + INPUT_LEN
        X_aux_list.append(aux_norm[idx - aux_seq:idx])
    X_aux = np.array(X_aux_list)

    n_samples = min(len(X_main), len(X_aux))
    X_main = X_main[:n_samples]
    X_aux = X_aux[:n_samples]
    years = years[:n_samples]

    tr_mask = (years >= 1979) & (years <= 2010)
    v_mask  = (years >= 2011) & (years <= 2015)
    te_mask = (years >= 2016) & (years <= 2025)

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
    return data_dict, scaler_ice, X_aux.shape[2]


def objective(trial, data_dict, scaler_ice, n_aux_ch, n_epochs=200):
    """Optuna objective: minimize validation loss."""
    hp = {
        "main_hidden": 256,
        "aux_hidden": trial.suggest_int("aux_hidden", 16, 128, step=16),
        "aux_seq_len": trial.suggest_int("aux_seq_len", 2, 12, step=2),
        "num_layers": 1,
        "dropout": 0.1,
        "aux_dropout": trial.suggest_float("aux_dropout", 0.1, 0.8, step=0.1),
        "batch_size": 32,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
    }

    set_seed(42)
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
        model, tr_ldr, v_ldr, hp, device,
        num_epochs=n_epochs, early_stopping_patience=20, verbose=False
    )

    return min(val_losses) if val_losses else float("inf")


def tune_region(region_key, cfg, n_trials=20):
    """对一个区域+最优aux运行Optuna调参。"""
    aux_vars = cfg["aux"]
    aux_label = "+".join(aux_vars)
    print(f"\n{'='*60}")
    print(f"  Tuning: {cfg['desc']}")
    print(f"  Region: {region_key}, Aux: {aux_label}, Trials: {n_trials}")
    print(f"  Baseline RMSE: {cfg['baseline_rmse']:.6f}")
    print(f"  Best aux (E7v1 HP): {cfg['best_aux_rmse']:.6f} (Δ={cfg['best_aux_rmse']-cfg['baseline_rmse']:+.6f})")
    print(f"{'='*60}")

    data_dict, scaler_ice, n_aux_ch = load_regional_data(region_key, aux_vars)

    # 创建 Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: objective(trial, data_dict, scaler_ice, n_aux_ch, n_epochs=200),
        n_trials=n_trials, show_progress_bar=True
    )

    best_params = study.best_params
    print(f"\n  Best params: {best_params}")
    print(f"  Best val loss: {study.best_value:.6f}")

    # === 用最优参数跑5-seed ensemble ===
    hp_opt = E7V1_HP.copy()
    hp_opt.update({
        "aux_hidden": best_params["aux_hidden"],
        "aux_seq_len": best_params["aux_seq_len"],
        "aux_dropout": best_params["aux_dropout"],
        "learning_rate": best_params["learning_rate"],
        "weight_decay": best_params["weight_decay"],
    })
    print(f"\n  Running 5-seed ensemble with tuned params...")
    print(f"  aux_hidden={hp_opt['aux_hidden']}, aux_seq_len={hp_opt['aux_seq_len']}, "
          f"aux_dropout={hp_opt['aux_dropout']}, lr={hp_opt['learning_rate']:.2e}, "
          f"wd={hp_opt['weight_decay']:.2e}")

    seeds = [42, 52, 62, 72, 82]
    seed_results = []
    for seed in seeds:
        set_seed(seed)
        tr_ds = DualEncoderDataset(*data_dict["train"])
        v_ds  = DualEncoderDataset(*data_dict["val"])
        tr_ldr = DataLoader(tr_ds, batch_size=hp_opt["batch_size"], shuffle=True)
        v_ldr  = DataLoader(v_ds,  batch_size=hp_opt["batch_size"])

        model = SeaIceDualEncoderLSTM(
            input_size=1, main_hidden=hp_opt["main_hidden"],
            aux_hidden=hp_opt["aux_hidden"], aux_input_size=n_aux_ch,
            aux_seq_len=hp_opt["aux_seq_len"], num_layers=hp_opt["num_layers"],
            output_len=OUTPUT_LEN, dropout=hp_opt["dropout"],
            aux_dropout=hp_opt["aux_dropout"],
        ).to(device)

        train_losses, val_losses, best_state, best_epoch, train_time = train_dual_encoder(
            model, tr_ldr, v_ldr, hp_opt, device,
            num_epochs=1000, early_stopping_patience=30, verbose=False
        )

        y_pred = predict_dual_encoder(model, *data_dict["test"][:2], device, scaler_ice)
        y_true = scaler_ice.inverse_transform(
            data_dict["test"][2].reshape(-1, 1)
        ).reshape(-1, OUTPUT_LEN)

        metrics = calculate_metrics(y_true, y_pred)
        print(f"    seed={seed}: RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}")
        seed_results.append({
            "seed": seed, "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]), "y_pred": y_pred, "y_true": y_true
        })

    # Ensemble
    all_preds = np.stack([r["y_pred"] for r in seed_results], axis=0)
    ensemble_pred = all_preds.mean(axis=0)
    ensemble_metrics = calculate_metrics(seed_results[0]["y_true"], ensemble_pred)
    seed_rmses = [r["rmse"] for r in seed_results]

    tuned_rmse = float(ensemble_metrics["rmse"])
    delta_vs_baseline = tuned_rmse - cfg["baseline_rmse"]
    delta_vs_e7v1 = tuned_rmse - cfg["best_aux_rmse"]

    print(f"\n  {'='*50}")
    print(f"  {cfg['desc']} - Tuned Results")
    print(f"  {'='*50}")
    print(f"  Baseline (no aux):      {cfg['baseline_rmse']:.6f}")
    print(f"  Best aux (E7v1 HP):     {cfg['best_aux_rmse']:.6f}")
    print(f"  Tuned aux:              {tuned_rmse:.6f}")
    print(f"  Δ vs Baseline:          {delta_vs_baseline:+.6f}")
    print(f"  Δ vs E7v1 (gain from tuning): {delta_vs_e7v1:+.6f}")
    print(f"  Seed RMSEs: {[f'{r:.6f}' for r in seed_rmses]}")
    print(f"  Tuned params: aux_hidden={hp_opt['aux_hidden']}, "
          f"aux_seq_len={hp_opt['aux_seq_len']}, aux_dropout={hp_opt['aux_dropout']}, "
          f"lr={hp_opt['learning_rate']:.2e}, wd={hp_opt['weight_decay']:.2e}")

    # 判定
    if delta_vs_baseline < -0.001:
        verdict = "✅ 调参后有实质改善"
    elif abs(delta_vs_baseline) < 0.001:
        verdict = "≈ 调参后接近基线，气候指数无增量"
    else:
        verdict = "✗ 调参后仍劣于基线"

    result = {
        "region": region_key, "desc": cfg["desc"], "aux_vars": aux_vars,
        "baseline_rmse": cfg["baseline_rmse"],
        "e7v1_rmse": cfg["best_aux_rmse"],
        "tuned_rmse": tuned_rmse,
        "delta_vs_baseline": delta_vs_baseline,
        "delta_vs_e7v1": delta_vs_e7v1,
        "best_params": best_params,
        "best_val_loss": study.best_value,
        "seed_rmses": seed_rmses,
        "verdict": verdict,
    }

    result_path = os.path.join(RESULTS_DIR, f"tuned_{region_key}.json")
    with open(result_path, "w") as f:
        # Convert numpy stuff
        save_result = {k: v for k, v in result.items()}
        json.dump(save_result, f, indent=2, default=str)
    print(f"  Saved: {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Tune Phase 4 regional experiments")
    parser.add_argument("--region", type=str, default="all",
                        help="Region to tune (bering, chukchi, barents, kara, laptev, greenland, central_arctic, all)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of Optuna trials per region")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for k, v in REGION_TUNE_CONFIGS.items():
            print(f"  {k:<18s}: {v['desc']:<25s} baseline={v['baseline_rmse']:.6f}  best_aux={v['best_aux_rmse']:.6f}")
        return

    regions = list(REGION_TUNE_CONFIGS.keys()) if args.region == "all" else [args.region]
    print(f"Tuning {len(regions)} regions with {args.trials} trials each")
    print(f"Device: {device}")

    all_results = {}
    for region in regions:
        if region not in REGION_TUNE_CONFIGS:
            print(f"Unknown region: {region}")
            continue
        r = tune_region(region, REGION_TUNE_CONFIGS[region], n_trials=args.trials)
        all_results[region] = r

    # ==== Final Summary ====
    print(f"\n{'='*80}")
    print(f"  Phase 4 Regional Tuning: Final Summary")
    print(f"{'='*80}")
    hdr = f"  {'Region':<18s} {'Aux':<12s} {'Baseline':>10s} {'E7v1 HP':>10s} {'Tuned':>10s} {'ΔvsBase':>10s} {'ΔvsE7v1':>10s} {'Verdict'}"
    print(hdr)
    print(f"  {'-'*78}")

    n_improved = 0; n_total = 0
    for region, r in all_results.items():
        aux = "+".join(r["aux_vars"])
        n_total += 1
        improved = r["delta_vs_baseline"] < 0
        if improved: n_improved += 1
        print(f"  {region:<18s} {aux:<12s} {r['baseline_rmse']:>10.6f} {r['e7v1_rmse']:>10.6f} "
              f"{r['tuned_rmse']:>10.6f} {r['delta_vs_baseline']:>+10.6f} {r['delta_vs_e7v1']:>+10.6f} "
              f"{'✓ better' if improved else '✗ worse'}")

    print(f"  {'='*80}")
    print(f"  Improved over baseline after tuning: {n_improved}/{n_total}")

    # Save master summary
    summary = {region: {
        "desc": r["desc"], "baseline_rmse": r["baseline_rmse"],
        "tuned_rmse": r["tuned_rmse"], "delta_vs_baseline": r["delta_vs_baseline"],
        "verdict": r["verdict"], "best_params": r["best_params"],
    } for region, r in all_results.items()}
    summary_path = os.path.join(RESULTS_DIR, "phase4_tuned_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
