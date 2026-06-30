# summary_v2.py
# Extended experiment summary including Phase 1-3 multivariate results.
# Generates comparison tables and a complete report.
import os, sys, json
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "summary")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Known baseline results (from existing summary.py / CLAUDE.md)
BASELINES = {
    "E1":   {"model": "Univariate LSTM",      "rmse": 0.4975, "params": 268000, "desc": "Baseline: ice only"},
    "E7v1": {"model": "Dual-Encoder AO+SST",  "rmse": 0.4974, "params": 74000,  "desc": "Best multivariate"},
    "LR":   {"model": "LinearRegression",     "rmse": 0.5317, "params": 156,     "desc": "156-parameter baseline"},
    "RNN":  {"model": "SimpleRNN",            "rmse": 0.5255, "params": 5000,    "desc": "Basic RNN baseline"},
}


def load_phase_results(results_dir, phase_name):
    """Load all JSON results from a phase directory."""
    results = {}
    if not os.path.exists(results_dir):
        return results

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        exp_id = fname.replace(".json", "")
        with open(os.path.join(results_dir, fname), "r") as f:
            data = json.load(f)
            results[exp_id] = {
                "rmse_mean": data.get("rmse_mean", data.get("ensemble_rmse")),
                "rmse_std": data.get("rmse_std", 0),
                "ensemble_rmse": data.get("ensemble_rmse"),
                "ensemble_mae": data.get("ensemble_mae"),
                "desc": data.get("desc", exp_id),
                "aux_vars": data.get("aux_vars", []),
                "n_aux_ch": data.get("n_aux_ch", 0),
                "n_seeds": data.get("n_seeds", 1),
            }
    return results


def generate_report():
    """Generate complete summary report with all experiments."""
    p1 = load_phase_results(config.RESULTS_PHASE1_DIR, "Phase 1")
    p2 = load_phase_results(config.RESULTS_PHASE2_DIR, "Phase 2")
    cond = load_phase_results(config.RESULTS_COND_DIR, "Conditional Skill")

    all_experiments = {}
    all_experiments.update(BASELINES)
    all_experiments.update({f"P1_{k}": {
        "model": v.get("desc", k), "rmse": v.get("rmse_mean", 0),
        "rmse_std": v.get("rmse_std", 0), "params": "varies",
        "desc": f"Phase1: {v.get('desc', k)}", "source": "phase1",
    } for k, v in p1.items()})
    all_experiments.update({f"P2_{k}": {
        "model": v.get("desc", k), "rmse": v.get("rmse_mean", 0),
        "rmse_std": v.get("rmse_std", 0), "params": "varies",
        "desc": f"Phase2: {v.get('desc', k)}", "source": "phase2",
    } for k, v in p2.items()})

    # Sort by RMSE
    sorted_items = sorted(all_experiments.items(), key=lambda x: x[1].get("rmse", 999))

    # Build report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("  Arctic Sea Ice Prediction: Extended Multivariate Experiments")
    report_lines.append("=" * 80)
    report_lines.append(f"  Generated: {__import__('datetime').datetime.now().isoformat()}")
    report_lines.append("")

    # Table 1: All experiments ranked by RMSE
    report_lines.append("-" * 80)
    report_lines.append("  Table 1: All Experiments Ranked by RMSE")
    report_lines.append("-" * 80)
    report_lines.append(f"  {'Rank':<6s} {'ID':<12s} {'Model':<35s} {'RMSE':>8s} {'Std':>8s} {'Params':>10s}")
    report_lines.append(f"  {'-'*76}")

    for rank, (eid, info) in enumerate(sorted_items, 1):
        rmse_val = info.get("rmse", 0)
        rmse_std = info.get("rmse_std", 0)
        params_str = str(info.get("params", "")) if info.get("params") else ""
        report_lines.append(
            f"  {rank:<6d} {eid:<12s} {info['model']:<35s} {rmse_val:8.4f} "
            f"{rmse_std:8.4f} {params_str:>10s}"
        )

    report_lines.append("")

    # Table 2: Phase 1 - Single variable impact
    if p1:
        report_lines.append("-" * 80)
        report_lines.append("  Table 2: Phase 1 - Single Variable Impact")
        report_lines.append("-" * 80)
        report_lines.append(f"  {'Exp':<10s} {'Variable':<25s} {'RMSE':>8s} {'Std':>8s} {'Channels':>10s} {'Seeds':>8s}")
        report_lines.append(f"  {'-'*68}")

        e1_rmse = BASELINES.get("E1", {}).get("rmse", 0.4975)
        p1_sorted = sorted(p1.items(), key=lambda x: x[1].get("rmse_mean", 999))

        for eid, info in p1_sorted:
            rmse_val = info.get("rmse_mean", 0)
            delta = rmse_val - e1_rmse
            sign = "+" if delta > 0 else ""
            report_lines.append(
                f"  {eid:<10s} {info['desc']:<25s} {rmse_val:8.4f} {info.get('rmse_std', 0):8.4f} "
                f"{info.get('n_aux_ch', 0):>10d} {info.get('n_seeds', 0):>8d}  (Delta E1: {sign}{delta:.4f})"
            )
        report_lines.append("")

    # Table 3: Phase 2 - Variable combinations
    if p2:
        report_lines.append("-" * 80)
        report_lines.append("  Table 3: Phase 2 - Variable Combinations")
        report_lines.append("-" * 80)
        report_lines.append(f"  {'Exp':<10s} {'Combination':<35s} {'RMSE':>8s} {'Std':>8s} {'Channels':>10s}")
        report_lines.append(f"  {'-'*72}")

        p2_sorted = sorted(p2.items(), key=lambda x: x[1].get("rmse_mean", 999))
        for eid, info in p2_sorted:
            vars_str = "+".join(info.get("aux_vars", []))
            report_lines.append(
                f"  {eid:<10s} {info['desc']:<35s} {info.get('rmse_mean', 0):8.4f} "
                f"{info.get('rmse_std', 0):8.4f} {info.get('n_aux_ch', 0):>10d}"
            )
        report_lines.append("")

    # Table 4: Conditional skill summary
    if cond:
        report_lines.append("-" * 80)
        report_lines.append("  Table 4: Conditional Skill Analysis")
        report_lines.append("-" * 80)
        for fname, data in cond.items():
            overall = data.get("overall", {})
            report_lines.append(f"  {fname}:")
            report_lines.append(f"    Overall RMSE delta: {overall.get('delta', 'N/A'):.4f}")
            report_lines.append(f"    95% CI: [{overall.get('bootstrap_ci_low', 'N/A')}, "
                               f"{overall.get('bootstrap_ci_upper', 'N/A')}]")
            report_lines.append(f"    p-value: {overall.get('p_value', 'N/A')}")
            report_lines.append(f"    Significant: {overall.get('significant', 'N/A')}")

            # Best/worst months
            monthly = data.get("monthly", {})
            if monthly:
                best_month = min(monthly.items(), key=lambda x: x[1].get("delta", 0))
                worst_month = max(monthly.items(), key=lambda x: x[1].get("delta", 0))
                report_lines.append(f"    Best month: {best_month[0]} (delta={best_month[1].get('delta', 0):.4f})")
                report_lines.append(f"    Worst month: {worst_month[0]} (delta={worst_month[1].get('delta', 0):.4f})")
        report_lines.append("")

    # Key findings
    report_lines.append("-" * 80)
    report_lines.append("  Key Findings")
    report_lines.append("-" * 80)

    findings = []

    # Finding 1: Best multivariate model
    if p1 or p2:
        best_multivariate = min(
            [(k, v) for k, v in all_experiments.items() if v.get("source") in ("phase1", "phase2")],
            key=lambda x: x[1].get("rmse", 999), default=(None, None)
        )
        if best_multivariate[0]:
            findings.append(
                f"1. Best multivariate model: {best_multivariate[0]} "
                f"(RMSE={best_multivariate[1].get('rmse', 0):.4f})"
            )

    # Finding 2: Variables that help
    if p1:
        helpful = [(eid, info) for eid, info in p1.items()
                    if info.get("rmse_mean", 999) < BASELINES.get("E1", {}).get("rmse", 0.4975)]
        if helpful:
            names = [f"{eid} ({h[1]['desc']})" for h in helpful]
            findings.append(f"2. Variables improving over univariate baseline: {', '.join(names)}")
        else:
            findings.append("2. No single variable significantly improved over univariate baseline.")

    # Finding 3: Conditional skill
    if cond:
        sig_results = [fname for fname, data in cond.items()
                       if data.get("overall", {}).get("significant", False)]
        if sig_results:
            findings.append(f"3. Significant conditional skill found in: {', '.join(sig_results)}")
        else:
            findings.append("3. No significant conditional skill at p < 0.05 in overall comparison.")
            findings.append("   Check per-month decomposition for subset-specific improvements.")

    # Finding 4: Diminishing returns
    if p2 and p1:
        best_single = min(p1.values(), key=lambda x: x.get("rmse_mean", 999))
        best_combo = min(p2.values(), key=lambda x: x.get("rmse_mean", 999))
        if best_combo.get("rmse_mean", 999) >= best_single.get("rmse_mean", 999):
            findings.append("4. Diminishing returns: best combination did NOT outperform best single variable.")
            findings.append("   This suggests information redundancy among climate variables.")

    for f_text in findings:
        report_lines.append(f"  {f_text}")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("  End of Report")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    # Save
    report_path = os.path.join(OUTPUT_DIR, "extended_multivariate_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved: {report_path}")
    return report_text


if __name__ == "__main__":
    generate_report()
