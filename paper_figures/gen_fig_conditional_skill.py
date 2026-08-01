# gen_fig_conditional_skill.py
# Fig 4.4-4.8: Conditional skill analysis.
# Monthly RMSE decomposition, lead-time skill, bootstrap forest plot.
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
import nature_figure_config  # nature-figure: 600 DPI + Arial + clean spines


OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "plots", "paper")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Month labels
MONTH_LABELS = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']


def load_conditional_results(results_dir):
    """Load conditional skill analysis JSON results."""
    results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname), "r") as f:
                data = json.load(f)
                results[fname] = data
    return results


def plot_monthly_rmse(data, save_path):
    """Fig 4.4: Calendar-month RMSE for multivariate vs univariate."""
    fig, ax = plt.subplots(figsize=(10, 5))

    monthly = data.get("monthly", {})
    months = list(range(1, 13))
    rmse_a = [monthly.get(str(m), {}).get("rmse_a", np.nan) for m in months]
    rmse_b = [monthly.get(str(m), {}).get("rmse_b", np.nan) for m in months]
    deltas = [monthly.get(str(m), {}).get("delta", np.nan) for m in months]

    # Main plot: RMSE curves
    ax.plot(months, rmse_a, 's--', color='#1b7837', linewidth=2, markersize=8,
            label='Multivariate (E7v1)')
    ax.plot(months, rmse_b, 'o-', color='#2166ac', linewidth=2, markersize=8,
            label='Univariate (E1)')
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS, fontsize=10)
    ax.set_ylabel('RMSE (百万平方公里)', fontsize=12)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=11)
    ax.grid(alpha=0.3)

    # Inset: delta bar chart
    ax_inset = ax.inset_axes([0.55, 0.55, 0.40, 0.40])
    colors = ['#d73027' if d > 0 else '#1b7837' for d in deltas]
    ax_inset.bar(months, deltas, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_inset.axhline(y=0, color='black', linewidth=0.5)
    ax_inset.set_xticks(months)
    ax_inset.set_xticklabels(MONTH_LABELS, fontsize=7, rotation=45)
    ax_inset.set_ylabel('Delta RMSE', fontsize=8)
    ax_inset.grid(axis='y', alpha=0.3)

    ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    plt.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_leadtime_skill(data, save_path):
    """Fig 4.8: Lead-time skill decay for multivariate vs univariate."""
    fig, ax = plt.subplots(figsize=(10, 5))

    lt_data = data.get("leadtime", {})
    leads = sorted([int(k) for k in lt_data.keys()])
    rmse_a = [lt_data.get(str(k), {}).get("rmse_a", np.nan) for k in leads]
    rmse_b = [lt_data.get(str(k), {}).get("rmse_b", np.nan) for k in leads]
    deltas = [lt_data.get(str(k), {}).get("delta", np.nan) for k in leads]

    ax.plot(leads, rmse_a, 's--', color='#1b7837', linewidth=2, markersize=8,
            label='Multivariate')
    ax.plot(leads, rmse_b, 'o-', color='#2166ac', linewidth=2, markersize=8,
            label='Univariate (E1)')
    ax.fill_between(leads, rmse_a, rmse_b, alpha=0.1, color='gray')

    ax.set_xlabel('Prediction lead time (months)', fontsize=12)
    ax.set_ylabel('RMSE (M km^2)', fontsize=12)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=11)
    ax.grid(alpha=0.3)

    # Annotate delta at each lead
    for k, d in zip(leads, deltas):
        if not np.isnan(d):
            sign = "+" if d > 0 else ""
            ax.annotate(f'{sign}{d:.3f}', (k, max(rmse_a[leads.index(k)], rmse_b[leads.index(k)])),
                       fontsize=8, ha='center', va='bottom', color='#d73027' if d > 0 else '#1b7837')

    ax.text(0.03, 0.97, '(b)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    plt.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_bootstrap_forest(data, save_path):
    """Fig 4.5: Bootstrap significance forest plot (overall RMSE diff + CI)."""
    fig, ax = plt.subplots(figsize=(8, 4))

    overall = data.get("overall", {})
    delta = overall.get("delta", 0)
    ci_low = overall.get("bootstrap_ci_low", delta - 0.01)
    ci_upper = overall.get("bootstrap_ci_upper", delta + 0.01)
    p_val = overall.get("p_value", 1.0)
    significant = overall.get("significant", False)

    # Plot the point estimate and CI
    y_pos = 0
    ax.errorbar(delta, y_pos, xerr=[[delta - ci_low], [ci_upper - delta]],
                fmt='o', color='#1b7837' if delta < 0 else '#d73027',
                markersize=10, capsize=8, linewidth=2)
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-', alpha=0.5)

    # Add text annotations
    ax.text(delta, y_pos + 0.3, f'Delta = {delta:.4f}\n95% CI: [{ci_low:.4f}, {ci_upper:.4f}]\np = {p_val:.4f}',
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    status = 'SIGNIFICANT (multivariate better)' if significant else 'NOT significant'
    ax.set_title(f'Bootstrap Test ({data.get("overall", {}).get("n_bootstrap", 0)} samples): {status}',
                 fontsize=12)

    ax.set_ylim(-1, 1.5)
    ax.set_yticks([])
    ax.set_xlabel('RMSE Difference (Multivariate - Univariate)', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    results_dir = config.RESULTS_COND_DIR
    if not os.path.exists(results_dir):
        print(f"No conditional skill results found: {results_dir}")
        print("Run analysis_conditional_skill.py first.")
    else:
        results = load_conditional_results(results_dir)
        if not results:
            print("No results JSON files found.")

        for fname, data in results.items():
            base = fname.replace(".json", "")

            out1 = os.path.join(OUTPUT_DIR, f"{base}_monthly.png")
            plot_monthly_rmse(data, out1)

            out2 = os.path.join(OUTPUT_DIR, f"{base}_leadtime.png")
            plot_leadtime_skill(data, out2)

            out3 = os.path.join(OUTPUT_DIR, f"{base}_bootstrap.png")
            plot_bootstrap_forest(data, out3)
