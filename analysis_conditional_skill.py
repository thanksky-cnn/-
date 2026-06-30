import sys, os, json, argparse
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_preprocessing import (
    load_dual_encoder_data_v2, create_dual_targets_v2,
    load_and_merge_data, create_sequences
)
import torch
from src.model import SeaIceDualEncoderLSTM, SeaIceLSTM
from src.train import train_dual_encoder, predict_dual_encoder
from src.utils import set_seed, calculate_metrics

LAGGED_CSV = os.path.join(config.BASE_DIR, "data", "lagged_features_v2.csv")
RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "results", "conditional_skill")
os.makedirs(RESULTS_DIR, exist_ok=True)
E7V1_HP = {
    "main_hidden": 256, "aux_hidden": 64, "aux_seq_len": 6,
    "num_layers": 1, "dropout": 0.1, "aux_dropout": 0.6,
    "batch_size": 32, "learning_rate": 0.000635, "weight_decay": 2.59e-06,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ol = config.OUTPUT_LEN
target = config.TARGET_COLUMN
N_BOOTSTRAP = 10000


def rmse(y_true, y_pred):
    """Compute RMSE."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_monthly_rmse(y_true, y_pred, target_months):
    """Compute RMSE for each calendar month (1-12)."""
    monthly = {m: {"pred": [], "true": []} for m in range(1, 13)}
    for i in range(len(y_pred)):
        for k in range(y_pred.shape[1]):
            cal_month = int(target_months[i, k])
            monthly[cal_month]["pred"].append(y_pred[i, k])
            monthly[cal_month]["true"].append(y_true[i, k])
    rmse_by_month = {}
    for m in range(1, 13):
        if monthly[m]["pred"]:
            rmse_by_month[m] = rmse(
                np.array(monthly[m]["true"]),
                np.array(monthly[m]["pred"])
            )
    return rmse_by_month


def compute_leadtime_rmse(y_true, y_pred):
    """Compute RMSE for each forecast lead time (1..output_len)."""
    return {k+1: rmse(y_true[:, k], y_pred[:, k])
            for k in range(y_pred.shape[1])}


def bootstrap_rmse_diff(y_true, y_pred_a, y_pred_b, n_bootstrap=10000):
    """
    Bootstrap test: is model A significantly better than model B?
    Returns: mean_diff, ci_lower, ci_upper, p_value
    (diff = RMSE_A - RMSE_B, negative = A is better)
    """
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        rmse_a = rmse(y_true[idx], y_pred_a[idx])
        rmse_b = rmse(y_true[idx], y_pred_b[idx])
        diffs.append(rmse_a - rmse_b)
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    # p-value: fraction of bootstrap samples where diff >= 0 (i.e., A not better)
    p_value = np.mean(diffs >= 0)
    return float(np.mean(diffs)), float(ci_lower), float(ci_upper), float(p_value)


def get_target_months(df, start_idx, n_samples, ol, input_len=12):
    """Build target-month array from DataFrame."""
    all_months = df["month"].values
    target_months = np.zeros((n_samples, ol), dtype=int)
    for i in range(n_samples):
        base = start_idx + i
        for k in range(ol):
            if base + input_len + k < len(all_months):
                target_months[i, k] = int(all_months[base + input_len + k])
    return target_months


def compute_ao_phase_rmse(y_true, y_pred, target_months, df, test_mask):
    """Compute RMSE stratified by AO phase (positive/neutral/negative)."""
    years_test = df["year"].values[test_mask]
    months_test = df["month"].values[test_mask]

    # Get AO values for test samples
    ao_values = []
    for i in range(len(y_pred)):
        yr = years_test[i]
        mo = months_test[i]
        ao_row = df[(df["year"] == yr) & (df["month"] == mo)]
        ao_val = ao_row["ao"].values[0] if len(ao_row) > 0 and "ao" in df.columns else 0
        ao_values.append(ao_val)
    ao_values = np.array(ao_values)
    ao_std = np.std(ao_values[ao_values != 0]) if np.any(ao_values != 0) else 1.0

    results = {}
    for phase_name, mask in [
        ("AO+", ao_values > 0.5 * ao_std),
        ("AO neutral", (ao_values >= -0.5 * ao_std) & (ao_values <= 0.5 * ao_std)),
        ("AO-", ao_values < -0.5 * ao_std),
    ]:
        if mask.sum() > 5:
            r = rmse(y_true[mask], y_pred[mask])
            results[phase_name] = {"rmse": float(r), "n": int(mask.sum())}
    return results


def main():
    parser = argparse.ArgumentParser(description="Conditional skill analysis")
    parser.add_argument("--exp_a", type=str, default="E7v1_ref",
                        help="Multivariate experiment ID")
    parser.add_argument("--exp_b", type=str, default="E1",
                        help="Univariate baseline experiment ID")
    parser.add_argument("--exp_a_vars", type=str, default="ao,sst",
                        help="Aux vars for exp_a (comma-separated)")
    parser.add_argument("--load-results", type=str, default=None,
                        help="Load pre-computed results JSON instead of training")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    n_boot = args.bootstrap

    # Load data
    print("=" * 60)
    print("  Conditional Skill Analysis")
    print("=" * 60)
    print(f"  Model A (multivariate): {args.exp_a} ({args.exp_a_vars})")
    print(f"  Model B (univariate):   {args.exp_b} (ice only)")
    print(f"  Bootstrap: {n_boot} samples")

    if args.load_results:
        with open(args.load_results, "r") as f:
            results = json.load(f)
        y_true = np.array(results["y_true"])
        y_pred_a = np.array(results["y_pred_a"])
        y_pred_b = np.array(results["y_pred_b"])
        target_months = np.array(results["target_months"])
        df = pd.DataFrame(results["df"])
    else:
        # Load and train model A (multivariate)
        aux_vars = [v.strip() for v in args.exp_a_vars.split(",")]
        print(f"\n--- Training {args.exp_a} ---")
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
        from src.dataset import DualEncoderDataset
        from torch.utils.data import DataLoader

        set_seed(42)
        tr_ds = DualEncoderDataset(
            X_main[tr_mask], X_aux[tr_mask][:, -aux_seq:, :], y[tr_mask])
        v_ds = DualEncoderDataset(
            X_main[v_mask], X_aux[v_mask][:, -aux_seq:, :], y[v_mask])
        tr_ldr = DataLoader(tr_ds, batch_size=E7V1_HP["batch_size"], shuffle=True)
        v_ldr = DataLoader(v_ds, batch_size=E7V1_HP["batch_size"])

        model_a = SeaIceDualEncoderLSTM(
            input_size=1, main_hidden=E7V1_HP["main_hidden"],
            aux_hidden=E7V1_HP["aux_hidden"],
            aux_input_size=X_aux.shape[2],
            aux_seq_len=aux_seq, num_layers=1,
            output_len=ol, dropout=E7V1_HP["dropout"],
            aux_dropout=E7V1_HP["aux_dropout"],
        ).to(device)

        print(f"  Training {args.exp_a} ({count_parameters(model_a):,} params)...")
        train_dual_encoder(model_a, tr_ldr, v_ldr, E7V1_HP, device,
                          num_epochs=1000, early_stopping_patience=30, verbose=True)

        y_pred_a = predict_dual_encoder(
            model_a, X_main[te_mask], X_aux[te_mask][:, -aux_seq:, :],
            device, scaler_ice)
        y_true = scaler_ice.inverse_transform(
            y[te_mask].reshape(-1, 1)).reshape(-1, ol)

        # Load and train model B (univariate)
        print(f"\n--- Training {args.exp_b} (Univariate) ---")
        df_uni, scaler_uni = load_and_merge_data(config.DATA_DIR, target)
        df_uni = df_uni[df_uni["year"] >= df["year"].min()]
        data_uni = df_uni[f"{target}_scaled"].values
        X_uni, y_uni = create_sequences(data_uni, 12, ol)
        years_uni = df_uni["year"].values[12:len(df_uni) - ol + 1]

        nc = min(len(X_uni), len(y_uni), len(y))
        te_mask_u = (years_uni[:nc] >= 2016) & (years_uni[:nc] <= 2025)
        X_uni_te = X_uni[:nc][te_mask_u]

        set_seed(42)
        from src.model import SeaIceLSTM
        model_b = SeaIceLSTM(
            input_size=1, hidden_size=E7V1_HP["main_hidden"],
            num_layers=1, output_len=ol, dropout=E7V1_HP["dropout"],
        ).to(device)

        # Train B with simplified loop
        from src.dataset import SeaIceDataset
        tr_ds_b = SeaIceDataset(
            X_uni[:nc][(years_uni[:nc] >= 1979) & (years_uni[:nc] <= 2010)],
            y_uni[:nc][(years_uni[:nc] >= 1979) & (years_uni[:nc] <= 2010)])
        v_ds_b = SeaIceDataset(
            X_uni[:nc][(years_uni[:nc] >= 2011) & (years_uni[:nc] <= 2015)],
            y_uni[:nc][(years_uni[:nc] >= 2011) & (years_uni[:nc] <= 2015)])
        tr_ldr_b = DataLoader(tr_ds_b, batch_size=E7V1_HP["batch_size"], shuffle=True)
        v_ldr_b = DataLoader(v_ds_b, batch_size=E7V1_HP["batch_size"])

        from src.train import train_model
        train_losses_b, val_losses_b, best_path_b, best_ep_b, train_time_b = train_model(
            model_b, tr_ldr_b, v_ldr_b, config, device)

        # Predict B
        model_b.load_state_dict(torch.load(best_path_b))
        model_b.eval()
        with torch.no_grad():
            y_pred_b_scaled = model_b(
                torch.tensor(X_uni_te, dtype=torch.float32).to(device)
            ).cpu().numpy()
        y_pred_b = scaler_uni.inverse_transform(
            y_pred_b_scaled.reshape(-1, 1)).reshape(-1, ol)

        # Build target_months
        target_months = get_target_months(df, 0, len(y_true), ol)

    # ============================================================
    # 1. Overall comparison
    # ============================================================
    print(f"\n{'='*60}")
    print("  1. Overall RMSE Comparison")
    print(f"{'='*60}")
    rmse_a = rmse(y_true, y_pred_a)
    rmse_b = rmse(y_true, y_pred_b)
    mean_diff, ci_low, ci_up, p_val = bootstrap_rmse_diff(
        y_true, y_pred_a, y_pred_b, n_boot)
    print(f"  {args.exp_a} RMSE: {rmse_a:.4f}")
    print(f"  {args.exp_b} RMSE: {rmse_b:.4f}")
    print(f"  Delta (A - B): {rmse_a - rmse_b:.4f}")
    print(f"  Bootstrap 95% CI: [{ci_low:.4f}, {ci_up:.4f}]")
    print(f"  p-value (A not better): {p_val:.4f}")
    significant = (ci_up < 0)  # CI entirely negative = A significantly better
    print(f"  Significant (A < B): {'YES' if significant else 'no'}")

    # ============================================================
    # 2. Calendar-month decomposition
    # ============================================================
    print(f"\n{'='*60}")
    print("  2. Calendar-Month RMSE Decomposition")
    print(f"{'='*60}")
    monthly_a = compute_monthly_rmse(y_true, y_pred_a, target_months)
    monthly_b = compute_monthly_rmse(y_true, y_pred_b, target_months)
    print(f"  {'Month':>6s}  {'RMSE_A':>8s}  {'RMSE_B':>8s}  {'Delta':>8s}")
    for m in range(1, 13):
        if m in monthly_a and m in monthly_b:
            d = monthly_a[m] - monthly_b[m]
            print(f"  {m:>6d}  {monthly_a[m]:8.4f}  {monthly_b[m]:8.4f}  {d:+8.4f}")

    # ============================================================
    # 3. Lead-time decomposition
    # ============================================================
    print(f"\n{'='*60}")
    print("  3. Lead-Time RMSE Decomposition")
    print(f"{'='*60}")
    lt_a = compute_leadtime_rmse(y_true, y_pred_a)
    lt_b = compute_leadtime_rmse(y_true, y_pred_b)
    print(f"  {'Lead':>6s}  {'RMSE_A':>8s}  {'RMSE_B':>8s}  {'Delta':>8s}")
    for k in range(1, ol + 1):
        d = lt_a[k] - lt_b[k]
        print(f"  {k:>6d}  {lt_a[k]:8.4f}  {lt_b[k]:8.4f}  {d:+8.4f}")

    # ============================================================
    # 4. Save results
    # ============================================================
    output = {
        "overall": {
            "rmse_a": float(rmse_a), "rmse_b": float(rmse_b),
            "delta": float(rmse_a - rmse_b),
            "bootstrap_ci_low": ci_low, "bootstrap_ci_upper": ci_up,
            "p_value": p_val, "significant": bool(significant),
            "n_bootstrap": n_boot,
        },
        "monthly": {str(m): {"rmse_a": monthly_a.get(m), "rmse_b": monthly_b.get(m),
                              "delta": monthly_a.get(m, 0) - monthly_b.get(m, 0)}
                     for m in range(1, 13)},
        "leadtime": {str(k): {"rmse_a": lt_a[k], "rmse_b": lt_b[k],
                               "delta": lt_a[k] - lt_b[k]}
                      for k in range(1, ol + 1)},
    }

    out_path = os.path.join(RESULTS_DIR, f"conditional_{args.exp_a}_vs_{args.exp_b}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    return output


if __name__ == "__main__":
    main()
