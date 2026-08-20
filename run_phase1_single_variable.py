# run_phase1_single_variable.py
# Phase 1: 单变量增量实验 (Single-variable Increment)
# 度量每个气候指数独立加入双编码器辅编码器后的预测性能变化。
# 5-seed ensemble per experiment. 统一超参数 (E7v1 Optuna 最优).
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
MODELS_DIR = os.path.join(config.OUTPUT_DIR, "models", "phase1")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 统一超参数策略 (E7v1 Optuna 最优, 提纲1.3节)
E7V1_HP = {
    "main_hidden": 256, "aux_hidden": 64, "aux_seq_len": 6,
    "num_layers": 1, "dropout": 0.1, "aux_dropout": 0.6,
    "batch_size": 32, "learning_rate": 0.000635, "weight_decay": 2.59e-06,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ol = config.OUTPUT_LEN; target = config.TARGET_COLUMN
N_ENSEMBLE = 5

# Phase 1 实验定义 (提纲 2.1):
# E1=单变量基线, E7v1=已有最佳多变量, E14=AO, E8=NAO, E10=PNA, E9=Nino3.4
EXPERIMENTS = {
    "E1":   {"aux_vars": None,        "desc": "Univariate LSTM baseline"},
    "E7v1": {"aux_vars": ["ao","sst"],"desc": "DE (AO + SST)"},
    "E14":  {"aux_vars": ["ao"],      "desc": "DE (AO only)"},
    "E8":   {"aux_vars": ["nao"],     "desc": "DE (NAO only)"},
    "E10":  {"aux_vars": ["pna"],     "desc": "DE (PNA only)"},
    "E9":   {"aux_vars": ["nino34"],  "desc": "DE (Nino3.4 only)"},
}

print(f"Device: {device}, Output len: {ol}, Target: {target}")


# ============================================================
# E1: 单变量基线 (SeaIceLSTM, 无辅助编码器)
# ============================================================
def run_e1_univariate(fast_mode=False):
    """训练单变量LSTM基线 (12月冰面积 → 12月预测)。"""
    n_seeds = 1 if fast_mode else N_ENSEMBLE
    hp = E7V1_HP
    exp_id = "E1"
    print(f"\n{'='*60}")
    print(f"  {exp_id}: {EXPERIMENTS[exp_id]['desc']}")
    print(f"  Seeds: {n_seeds}")
    print(f"{'='*60}")

    # 加载冰数据 (使用 lagged_features_v2.csv 的 area 列)
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    df = pd.read_csv(LAGGED_CSV)
    target = config.TARGET_COLUMN
    ol = config.OUTPUT_LEN
    area = df[target].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    area_scaled = scaler.fit_transform(area).flatten()

    X, y = create_sequences(area_scaled, input_len=12, output_len=ol)
    # 年份数组需要与序列样本对齐：全数据564行 → 541个序列样本
    seq_start = 12
    seq_end = len(df) - ol + 1
    years = df['year'].values[seq_start:seq_end]

    tr_mask = (years >= 1979) & (years <= 2010)
    v_mask  = (years >= 2011) & (years <= 2015)
    te_mask = (years >= 2016) & (years <= 2025)

    X_train, y_train = X[tr_mask], y[tr_mask]
    X_val,   y_val   = X[v_mask],  y[v_mask]
    X_test,  y_test  = X[te_mask], y[te_mask]

    # 训练循环 (使用与双编码器一致的训练协议)
    seeds = [42 + i * 10 for i in range(n_seeds)]
    results = []
    for seed in seeds:
        set_seed(seed)
        model = SeaIceLSTM(
            input_size=1, hidden_size=hp["main_hidden"],
            num_layers=hp["num_layers"], output_len=ol,
            dropout=hp["dropout"],
        ).to(device)

        tr_ds = SeaIceDataset(X_train, y_train)
        v_ds  = SeaIceDataset(X_val, y_val)
        tr_ldr = DataLoader(tr_ds, batch_size=hp["batch_size"], shuffle=True)
        v_ldr  = DataLoader(v_ds,  batch_size=hp["batch_size"])

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=hp["learning_rate"],
            weight_decay=hp["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=15
        )

        best_val, best_state, best_ep, patience = float("inf"), None, 0, 0
        for epoch in range(1000):
            model.train()
            tr_loss = 0
            for Xb, yb in tr_ldr:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= len(tr_ldr)

            model.eval()
            v_loss = 0
            with torch.no_grad():
                for Xb, yb in v_ldr:
                    Xb, yb = Xb.to(device), yb.to(device)
                    v_loss += criterion(model(Xb), yb).item()
            v_loss /= len(v_ldr)
            scheduler.step(v_loss)

            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_ep = epoch + 1; patience = 0
            else:
                patience += 1
            if patience >= 30:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1, ol)
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, ol)
        metrics = calculate_metrics(y_true, y_pred)
        print(f"  [{exp_id}] seed={seed}: RMSE={metrics['rmse']:.4f}, "
              f"MAE={metrics['mae']:.4f}, best_ep={best_ep}")
        results.append({"seed": seed, "best_epoch": best_ep,
                        "y_pred": y_pred, "y_true": y_true, "metrics": metrics})

    return _ensemble_and_save(exp_id, EXPERIMENTS[exp_id], results, n_seeds,
                              None, years[te_mask])


# ============================================================
# 双编码器实验 (E7v1, E14, E8, E10, E9)
# ============================================================
def load_and_prepare_data(aux_vars):
    """加载v2数据，构建序列，按年分割。"""
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
    """训练单个种子。"""
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
        model, tr_ldr, v_ldr, hp, device,
        num_epochs=1000, early_stopping_patience=30, verbose=False
    )

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


def _ensemble_and_save(exp_id, cfg, results, n_seeds, aux_vars, test_years):
    """集成预测 + 保存结果。"""
    all_preds = np.stack([r["y_pred"] for r in results], axis=0)
    ensemble_pred = all_preds.mean(axis=0)
    y_true = results[0]["y_true"]
    ensemble_metrics = calculate_metrics(y_true, ensemble_pred)

    seed_rmses = [r["metrics"]["rmse"] for r in results]
    rmse_mean = np.mean(seed_rmses)
    rmse_std = np.std(seed_rmses)

    print(f"  {exp_id} seeds: RMSE={[f'{r:.4f}' for r in seed_rmses]}")
    print(f"  {exp_id} ensemble: RMSE={ensemble_metrics['rmse']:.4f}, "
          f"MAE={ensemble_metrics['mae']:.4f}")
    print(f"  {exp_id} seed stats: mean={rmse_mean:.4f}, std={rmse_std:.4f}")

    result = {
        "exp_id": exp_id, "desc": cfg["desc"], "aux_vars": aux_vars,
        "n_seeds": n_seeds,
        "seed_rmses": [float(r) for r in seed_rmses],
        "rmse_mean": float(rmse_mean), "rmse_std": float(rmse_std),
        "ensemble_rmse": float(ensemble_metrics["rmse"]),
        "ensemble_mae": float(ensemble_metrics["mae"]),
        "ensemble_mape": float(ensemble_metrics.get("mape", 0)),
    }
    if test_years is not None:
        result["test_years"] = test_years.tolist() if hasattr(test_years, 'tolist') else []

    result_path = os.path.join(RESULTS_DIR, f"{exp_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {result_path}")
    return result


def run_de_experiment(exp_id, cfg, fast_mode=False):
    """运行一个双编码器实验。"""
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

    return _ensemble_and_save(exp_id, cfg, results, n_seeds, cfg["aux_vars"],
                              data_dict["test_years"])


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Single-variable increment experiments")
    parser.add_argument("--exp", type=str, default="all",
                        help="Comma-separated experiment IDs (e.g., E1,E8,E10)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: single seed only")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments and exit")
    args = parser.parse_args()

    if args.list:
        for eid, cfg in EXPERIMENTS.items():
            aux = cfg["aux_vars"] if cfg["aux_vars"] else "(none — univariate)"
            print(f"  {eid}: {cfg['desc']} [{aux}]")
        return

    if args.exp == "all":
        to_run = list(EXPERIMENTS.keys())
    else:
        to_run = [e.strip() for e in args.exp.split(",")]

    print(f"Running {len(to_run)} experiments: {to_run}")
    if args.fast:
        print("FAST MODE: single seed only")

    all_results = {}
    for exp_id in to_run:
        if exp_id not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_id}, skipping")
            continue
        cfg = EXPERIMENTS[exp_id]

        if exp_id == "E1":
            r = run_e1_univariate(fast_mode=args.fast)
        else:
            r = run_de_experiment(exp_id, cfg, fast_mode=args.fast)
        all_results[exp_id] = r

    # Summary table
    print(f"\n{'='*70}")
    print("  Phase 1 Results Summary")
    print(f"{'='*70}")
    print(f"  {'Exp':<8s} {'Description':<30s} {'EnsRMSE':>8s} {'SeedMean':>8s} {'SeedStd':>8s}")
    print(f"  {'-'*65}")
    baseline_rmse = all_results.get("E1", {}).get("ensemble_rmse", None)
    for eid in EXPERIMENTS:
        if eid not in all_results:
            continue
        r = all_results[eid]
        delta = ""
        if baseline_rmse and eid != "E1":
            d = r["ensemble_rmse"] - baseline_rmse
            delta = f" {d:+.4f}"
        print(f"  {eid:<8s} {r['desc']:<30s} {r['ensemble_rmse']:8.4f} "
              f"{r['rmse_mean']:8.4f} {r['rmse_std']:8.4f}{delta}")

    print(f"  {'='*70}")

    # Save summary
    summary = {eid: {"desc": r["desc"], "ensemble_rmse": r["ensemble_rmse"],
                     "rmse_mean": r["rmse_mean"], "rmse_std": r["rmse_std"]}
               for eid, r in all_results.items()}
    summary_path = os.path.join(RESULTS_DIR, "phase1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
