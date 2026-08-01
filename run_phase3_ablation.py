# run_phase3_ablation.py
# Phase 3: 消融验证实验 (Ablation Verification)
# 从全变量中逐个移除，反向验证每个变量的必要性。
# 5-seed ensemble per experiment. 统一超参数 (E7v1 Optuna 最优).
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
RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "results", "phase3")
os.makedirs(RESULTS_DIR, exist_ok=True)

E7V1_HP = {
    "main_hidden": 256, "aux_hidden": 64, "aux_seq_len": 6,
    "num_layers": 1, "dropout": 0.1, "aux_dropout": 0.6,
    "batch_size": 32, "learning_rate": 0.000635, "weight_decay": 2.59e-06,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ol = config.OUTPUT_LEN; target = config.TARGET_COLUMN
N_ENSEMBLE = 5

# 全变量集合
ALL_VARS = ["ao", "sst", "nao", "pna", "nino34"]

# Phase 3 消融实验定义 (提纲 2.2.2):
# 从全变量(ALL_VARS)中逐个移除，反向验证必要性
ABLATIONS = {
    "E20": {"remove": "nao",     "desc": "All − NAO (缺大西洋扇区)"},
    "E21": {"remove": "nino34",  "desc": "All − Nino3.4 (缺热带信号)"},
    "E22": {"remove": "sst",     "desc": "All − SST (缺海洋热力)"},
    "E25": {"remove": "pna",     "desc": "All − PNA (缺太平洋扇区)"},
    "E33": {"remove": "ao",      "desc": "All − AO (缺北极全域信号)"},
}

# E19 (全变量) 从 Phase 2 复用作为全变量参照


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

    print(f"  [{exp_id}] seed={seed}, params={count_parameters(model):,}, aux_ch={n_aux_ch}")

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


def run_ablation(exp_id, cfg, fast_mode=False):
    n_seeds = 1 if fast_mode else N_ENSEMBLE
    aux_vars = [v for v in ALL_VARS if v != cfg["remove"]]

    print(f"\n{'='*60}")
    print(f"  {exp_id}: {cfg['desc']}")
    print(f"  Aux vars: {aux_vars} (removed: {cfg['remove']}), Seeds: {n_seeds}")
    print(f"{'='*60}")

    data_dict, scaler_ice, n_aux_ch = load_and_prepare_data(aux_vars)

    seeds = [42 + i * 10 for i in range(n_seeds)]
    results = []
    for seed in seeds:
        r = run_single_seed(exp_id, aux_vars, data_dict, scaler_ice, n_aux_ch, seed)
        results.append(r)

    # Ensemble
    all_preds = np.stack([r["y_pred"] for r in results], axis=0)
    ensemble_pred = all_preds.mean(axis=0)
    ensemble_metrics = calculate_metrics(results[0]["y_true"], ensemble_pred)

    seed_rmses = [r["metrics"]["rmse"] for r in results]
    rmse_mean = float(np.mean(seed_rmses))
    rmse_std = float(np.std(seed_rmses))

    print(f"  {exp_id} seeds: RMSE={[f'{r:.4f}' for r in seed_rmses]}")
    print(f"  {exp_id} ensemble: RMSE={ensemble_metrics['rmse']:.4f}, "
          f"MAE={ensemble_metrics['mae']:.4f}")

    result = {
        "exp_id": exp_id, "desc": cfg["desc"], "removed_var": cfg["remove"],
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
    parser = argparse.ArgumentParser(description="Phase 3: Ablation verification experiments")
    parser.add_argument("--exp", type=str, default="all")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for eid, cfg in ABLATIONS.items():
            aux_vars = [v for v in ALL_VARS if v != cfg["remove"]]
            print(f"  {eid}: {cfg['desc']}")
            print(f"       Aux vars: {aux_vars} (removed: {cfg['remove']})")
        return

    to_run = list(ABLATIONS.keys()) if args.exp == "all" \
             else [e.strip() for e in args.exp.split(",")]

    print(f"Running {len(to_run)} experiments: {to_run}")
    print(f"Full variable set: {ALL_VARS}")
    if args.fast:
        print("FAST MODE: single seed only")

    # 尝试加载 E19 (全变量) 和 E1 (单变量) 结果作为参照
    e19_rmse = None
    e19_path = os.path.join(config.OUTPUT_DIR, "results", "phase2", "E19.json")
    if os.path.exists(e19_path):
        with open(e19_path) as f:
            e19_rmse = json.load(f).get("ensemble_rmse")
            print(f"E19 (ALL 5) reference RMSE (from Phase 2): {e19_rmse:.4f}")

    e1_rmse = None
    e1_path = os.path.join(config.OUTPUT_DIR, "results", "phase1", "E1.json")
    if os.path.exists(e1_path):
        with open(e1_path) as f:
            e1_rmse = json.load(f).get("ensemble_rmse")
            print(f"E1 baseline RMSE (from Phase 1): {e1_rmse:.4f}")

    all_results = {}
    for exp_id in to_run:
        if exp_id not in ABLATIONS:
            print(f"Unknown experiment: {exp_id}, skipping")
            continue
        r = run_ablation(exp_id, ABLATIONS[exp_id], fast_mode=args.fast)
        all_results[exp_id] = r

    # Summary table
    print(f"\n{'='*80}")
    print("  Phase 3 Ablation Results Summary")
    print(f"{'='*80}")
    header = f"  {'Exp':<8s} {'Description':<35s} {'EnsRMSE':>8s}"
    if e19_rmse:
        header += f" {'ΔvsALL':>8s} {'Impact':>10s}"
    if e1_rmse:
        header += f" {'ΔvsE1':>8s}"
    print(header)
    print(f"  {'-'*70}")

    for eid in ABLATIONS:
        if eid not in all_results:
            continue
        r = all_results[eid]
        removed = ABLATIONS[eid]["remove"]
        line = f"  {eid:<8s} {r['desc']:<35s} {r['ensemble_rmse']:8.4f}"
        if e19_rmse:
            delta = r["ensemble_rmse"] - e19_rmse
            impact = "关键变量⚠" if delta > 0.01 else ("冗余变量✓" if abs(delta) < 0.003 else "边际变量")
            line += f" {delta:+8.4f} {impact:>10s}"
        if e1_rmse:
            line += f" {r['ensemble_rmse'] - e1_rmse:+8.4f}"
        print(line)

    # 变量重要性排名
    if e19_rmse:
        print(f"\n{'='*80}")
        print("  变量必要性排名 (ΔRMSE = 移除后RMSE − 全变量RMSE)")
        print(f"  (正值越大 = 移除该变量后性能下降越多 = 该变量越必要)")
        print(f"{'='*80}")
        ranked = sorted(all_results.items(),
                        key=lambda x: x[1]["ensemble_rmse"] - e19_rmse, reverse=True)
        for rank, (eid, r) in enumerate(ranked, 1):
            removed = ABLATIONS[eid]["remove"]
            delta = r["ensemble_rmse"] - e19_rmse
            bar = "█" * max(1, int(abs(delta) * 200))
            print(f"  {rank}. {removed:8s}  ΔRMSE={delta:+.4f}  {bar}")

    print(f"  {'='*80}")

    # Save summary
    summary = {eid: {"desc": r["desc"], "removed_var": r["removed_var"],
                     "aux_vars": r["aux_vars"],
                     "ensemble_rmse": r["ensemble_rmse"],
                     "rmse_mean": r["rmse_mean"], "rmse_std": r["rmse_std"]}
               for eid, r in all_results.items()}
    summary_path = os.path.join(RESULTS_DIR, "phase3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
