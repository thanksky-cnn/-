# gen_fig_ablation.py
# Fig 4.9: Ablation study waterfall chart.
# Shows RMSE change when removing each variable from the full model.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os, sys, json
fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "plots", "paper")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ABLATION_LABELS_CN = {
    "ao": "AO",
    "sst": "SST",
    "nao": "NAO",
    "nino34": "Nino3.4",
    "pdo": "PDO",
    "t2m": "T2M",
    "slp": "SLP",
    "lag12_ice": "Lag-12 Ice",
}


def plot_ablation_waterfall(baseline_rmse, ablation_results, save_path):
    """
    Waterfall chart: start from full model RMSE, show impact of removing each variable.

    Args:
        baseline_rmse: RMSE of full model (all variables)
        ablation_results: {var_name: rmse_without_var}
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort variables by impact (largest degradation when removed = most important)
    impacts = [(var, ablation_results[var] - baseline_rmse)
               for var in ablation_results]
    impacts.sort(key=lambda x: x[1], reverse=True)

    variables = [ABLATION_LABELS_CN.get(v, v) for v, _ in impacts]
    deltas = [d for _, d in impacts]

    # Bar chart
    colors = ['#d73027' if d > 0 else '#1b7837' for d in deltas]
    x = np.arange(len(variables))
    ax.bar(x, deltas, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Reference line
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-', alpha=0.8)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

    # Annotations
    for i, (var, delta) in enumerate(zip(variables, deltas)):
        sign = '+' if delta > 0 else ''
        y_pos = delta + (0.0005 if delta >= 0 else -0.0005)
        va = 'bottom' if delta >= 0 else 'top'
        ax.text(i, y_pos, f'{sign}{delta:.4f}', ha='center', va=va,
                fontsize=9, fontweight='bold',
                color='#d73027' if delta > 0 else '#1b7837')

    ax.set_xticks(x)
    ax.set_xticklabels(variables, fontsize=11, rotation=30, ha='right')
    ax.set_ylabel('RMSE Change when removed (百万平方公里)', fontsize=12)
    ax.set_xlabel('Removed Variable', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Title-like annotation
    ax.text(0.5, 1.02,
            f'Full model RMSE: {baseline_rmse:.4f} | '
            f'Positive = variable is helpful (RMSE increases when removed)',
            transform=ax.transAxes, fontsize=10, ha='center', style='italic')

    ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def load_ablation_results():
    """Try to load ablation results from Phase 2 or 3 directories."""
    results = {}
    for dir_path in [config.RESULTS_PHASE2_DIR]:
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if fname.endswith(".json") and not fname.startswith("phase"):
                with open(os.path.join(dir_path, fname), "r") as f:
                    d = json.load(f)
                    results[fname.replace(".json", "")] = d
    return results


if __name__ == "__main__":
    results = load_ablation_results()

    if results:
        # Find full model (all variables, max aux_vars)
        full_model = max(results.items(), key=lambda x: len(x[1].get("aux_vars", [])))
        full_id, full_data = full_model
        full_rmse = full_data.get("rmse_mean", full_data.get("ensemble_rmse", 0))
        full_vars = full_data.get("aux_vars", [])

        # Find ablation experiments (full model minus one variable)
        ablation = {}
        for eid, data in results.items():
            aux_vars = data.get("aux_vars", [])
            if len(aux_vars) == len(full_vars) - 1:
                missing = set(full_vars) - set(aux_vars)
                if len(missing) == 1:
                    var_name = list(missing)[0]
                    ablation[var_name] = data.get("rmse_mean", data.get("ensemble_rmse", 0))

        if ablation:
            out_path = os.path.join(OUTPUT_DIR, "fig_ablation_waterfall.png")
            plot_ablation_waterfall(full_rmse, ablation, out_path)
        else:
            print("No ablation experiments found (need full model minus one variable).")
            print("Run Phase 3 ablation experiments (E22-E25) first.")
    else:
        print("No results found. Run Phase 2/3 experiments first.")
