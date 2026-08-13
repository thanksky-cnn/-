# tune_phase4_enriched.py
# 对7个海域各自最优aux配置进行 Optuna 调参（基于 phase4_enriched 结果）
# 使用新的自注意力双编码器，不含 SST
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
RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "results", "phase4_enriched")
TUNED_DIR = os.path.join(RESULTS_DIR, "tuned")
os.makedirs(TUNED_DIR, exist_ok=True)

# 统一超参数（对照组）
UNIFIED_HP = {
    "main_hidden": 256, "aux_hidden": 64, "aux_seq_len": 6,
    "num_layers": 1, "dropout": 0.1, "aux_dropout": 0.6,
    "batch_size": 32, "learning_rate": 0.000635, "weight_decay": 2.59e-06,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
INPUT_LEN = 12; OUTPUT_LEN = 12

# 7 海域 + 最优 aux（从 phase4_enriched 结果读取，或手动覆盖）
# 格式：region -> (aux_vars, 中文名)
DEFAULT_BEST_AUX = {
    "bering":         (["ao"],     "白令海 + AO"),
    "chukchi":        (["pna"],    "楚科奇海 + PNA"),
    "barents":        (["nao"],    "巴伦支海 + NAO"),
    "kara":           (["nino34"], "喀拉海 + Nino3.4"),
    "laptev":         (["pna"],    "拉普捷夫海 + PNA"),
    "greenland":      (["nino34"], "格陵兰海 + Nino3.4"),
    "central_arctic": (["nao"],    "中北冰洋 + NAO"),
}

REGION_NAMES = {
    "bering": "白令海", "chukchi": "楚科奇海", "barents": "巴伦支海",
    "kara": "喀拉海", "laptev": "拉普捷夫海", "greenland": "格陵兰海",
    "central_arctic": "中北冰洋",
}


def load_best_aux_from_results():
    """从 phase4_enriched 结果自动确定每海域最优单变量指数。"""
    best_aux = {}
    for region_key in REGION_NAMES:
        # 读取该海域的所有实验结果
        region_exps = []
        for fname in os.listdir(RESULTS_DIR):
            if not fname.startswith("E") or not fname.endswith(".json"):
                continue
            if "summary" in fname or "tuned" in fname:
                continue
            with open(os.path.join(RESULTS_DIR, fname), "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("region") == region_key:
                region_exps.append(d)

        if not region_exps:
            continue

        baseline = next((e for e in region_exps if e["type"] == "baseline"), None)
        singles = [e for e in region_exps if e["type"] == "single"]
        if baseline and singles:
            # 最优单变量 = RMSE 最低的 single
            best_single = min(singles, key=lambda x: x["ensemble_rmse"])
            best_aux[region_key] = (
                best_single["aux_vars"],
                f"{REGION_NAMES[region_key]} + {best_single['aux_label'].lstrip('+')}",
                baseline["ensemble_rmse"],
                best_single["ensemble_rmse"],
            )
    return best_aux


def load_regional_data(region_key, aux_vars, aux_seq_len=6):
    """加载区域数据（与 run_phase4_enriched.py 相同逻辑）。"""
    ice_csv = os.path.join(DATA_DIR, f"{region_key}_monthly.csv")
    df_ice = pd.read_csv(ice_csv)
    df_ice = df_ice[df_ice['area'] > -0.005].copy()
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

    n_total = len(aux_norm)
    X_aux_list = []
    for i in range(n_total - INPUT_LEN - OUTPUT_LEN + 1):
        idx = i + INPUT_LEN
        X_aux_list.append(aux_norm[idx - aux_seq_len:idx])
    X_aux = np.array(X_aux_list)

    n_samples = min(len(X_main), len(X_aux))
    X_main = X_main[:n_samples]; X_aux = X_aux[:n_samples]; years = years[:n_samples]

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
    }, scaler_ice, X_aux.shape[2]


def objective(trial, region_key, aux_vars, n_epochs=200):
    """Optuna 目标函数：最小化验证损失。

    每次 trial 根据搜索到的 aux_seq_len 重新构建数据，
    确保 aux_seq_len 维度真正参与搜索（而非固定 6 步）。
    """
    hp = {
        "main_hidden": 256,
        "aux_hidden": trial.suggest_int("aux_hidden", 16, 128, step=16),
        "aux_seq_len": trial.suggest_int("aux_seq_len", 2, 12, step=2),
        "num_layers": 1,
        "dropout": 0.1,
        "aux_dropout": trial.suggest_float("aux_dropout", 0.0, 0.8, step=0.1),
        "batch_size": 32,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
    }

    # 根据 trial 的 aux_seq_len 重建数据（修复 seq_len 搜索失效 bug）
    data_dict, scaler_ice, n_aux_ch = load_regional_data(
        region_key, aux_vars, aux_seq_len=hp["aux_seq_len"])

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


def tune_region(region_key, aux_vars, desc, baseline_rmse, unified_rmse, n_trials=20):
    """对一个海域+最优aux运行 Optuna 调参。"""
    print(f"\n{'='*60}")
    print(f"  Tuning: {desc}")
    print(f"  Region: {region_key}, Trials: {n_trials}")
    print(f"  Baseline RMSE: {baseline_rmse:.6f}")
    print(f"  Unified HP RMSE: {unified_rmse:.6f}")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: objective(trial, region_key, aux_vars, n_epochs=200),
        n_trials=n_trials, show_progress_bar=False
    )

    best_params = study.best_params
    print(f"\n  Best params: {best_params}")
    print(f"  Best val loss: {study.best_value:.6f}")

    # 用最优参数跑 5-seed ensemble
    hp_opt = UNIFIED_HP.copy()
    hp_opt.update({
        "aux_hidden": best_params["aux_hidden"],
        "aux_seq_len": best_params["aux_seq_len"],
        "aux_dropout": best_params["aux_dropout"],
        "learning_rate": best_params["learning_rate"],
        "weight_decay": best_params["weight_decay"],
    })

    # 用调参后的 aux_seq_len 重新加载数据
    data_dict, scaler_ice, n_aux_ch = load_regional_data(region_key, aux_vars, aux_seq_len=hp_opt["aux_seq_len"])

    print(f"\n  Running 5-seed ensemble with tuned params...")
    seeds = [42, 52, 62, 72, 82]
    seed_rmses = []
    all_preds = []
    y_true_ens = None
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
            data_dict["test"][2].reshape(-1, 1)).reshape(-1, OUTPUT_LEN)

        metrics = calculate_metrics(y_true, y_pred)
        seed_rmses.append(float(metrics["rmse"]))
        all_preds.append(y_pred)
        if y_true_ens is None:
            y_true_ens = y_true
        print(f"    seed={seed}: RMSE={metrics['rmse']:.6f}")

    # 真正的 ensemble RMSE（种子平均预测后算 RMSE，与 unified_rmse 可比）
    ensemble_pred = np.mean(np.array(all_preds), axis=0)
    tuned_rmse = float(np.sqrt(np.mean((y_true_ens - ensemble_pred) ** 2)))

    delta_vs_baseline = tuned_rmse - baseline_rmse
    delta_vs_unified = tuned_rmse - unified_rmse

    print(f"\n  {'='*50}")
    print(f"  {desc} — Tuned Results")
    print(f"  {'='*50}")
    print(f"  Baseline:   {baseline_rmse:.6f}")
    print(f"  Unified HP: {unified_rmse:.6f}")
    print(f"  Tuned:      {tuned_rmse:.6f}")
    print(f"  Δ vs Base:  {delta_vs_baseline:+.6f} ({delta_vs_baseline/baseline_rmse*100:+.1f}%)")
    print(f"  Δ vs Unified: {delta_vs_unified:+.6f}")

    if delta_vs_baseline < -0.001:
        verdict = "✅ 实质改善"
    elif abs(delta_vs_baseline) < 0.001:
        verdict = "≈ 无增量"
    else:
        verdict = "✗ 劣于基线"

    result = {
        "region": region_key, "desc": desc, "aux_vars": aux_vars,
        "baseline_rmse": baseline_rmse, "unified_rmse": unified_rmse,
        "tuned_rmse": tuned_rmse,
        "delta_vs_baseline": delta_vs_baseline,
        "delta_vs_unified": delta_vs_unified,
        "delta_pct_vs_baseline": round(delta_vs_baseline / baseline_rmse * 100, 2),
        "best_params": best_params,
        "best_val_loss": study.best_value,
        "seed_rmses": seed_rmses,
        "verdict": verdict,
    }

    result_path = os.path.join(TUNED_DIR, f"tuned_{region_key}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Tune Phase 4 enriched regional experiments")
    parser.add_argument("--region", type=str, default="all")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    # 自动从结果确定最优 aux
    auto_best = load_best_aux_from_results()
    print(f"从 phase4_enriched 结果确定最优 aux:")
    best_configs = {}
    for region_key, (aux_vars, desc, baseline, unified) in auto_best.items():
        best_configs[region_key] = (aux_vars, desc, baseline, unified)
        idx_name = "+".join(aux_vars)
        print(f"  {REGION_NAMES[region_key]:6s} -> {idx_name:10s} "
              f"(baseline={baseline:.6f}, unified={unified:.6f}, Δ={unified-baseline:+.6f})")

    if args.list:
        return

    regions = list(best_configs.keys()) if args.region == "all" else [args.region]
    print(f"\nTuning {len(regions)} regions with {args.trials} trials each")

    all_results = {}
    for region_key in regions:
        if region_key not in best_configs:
            print(f"Unknown region: {region_key}")
            continue
        aux_vars, desc, baseline, unified = best_configs[region_key]
        r = tune_region(region_key, aux_vars, desc, baseline, unified, n_trials=args.trials)
        all_results[region_key] = r

    # 汇总
    print(f"\n{'='*80}")
    print(f"  Phase 4 Enriched Tuning: Final Summary")
    print(f"{'='*80}")
    print(f"  {'Region':<16s} {'Baseline':>10s} {'Unified':>10s} {'Tuned':>10s} "
          f"{'ΔvsBase':>9s} {'Δ%':>8s} {'Verdict'}")
    print(f"  {'-'*76}")
    for region_key, r in all_results.items():
        print(f"  {REGION_NAMES[region_key]:<16s} {r['baseline_rmse']:>10.6f} "
              f"{r['unified_rmse']:>10.6f} {r['tuned_rmse']:>10.6f} "
              f"{r['delta_vs_baseline']:>+9.6f} {r['delta_pct_vs_baseline']:>+7.1f}% "
              f"{r['verdict']}")

    summary_path = os.path.join(TUNED_DIR, "tuned_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
