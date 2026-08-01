# gen_fig_bestworst_multi.py
# Fig 4.6: Best/worst year comparison — multivariate vs univariate.
# Calendar-year 12-month overlay for the year with best and worst RMSE.
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
MONTHS_CN = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']


def get_year_data(y_true, y_pred_a, y_pred_b, test_years, test_months, target_year):
    """Extract all 12 calendar months of predictions for a specific year."""
    pred_a, pred_b, true_v = [], [], []
    for i in range(len(y_pred_a)):
        for k in range(y_pred_a.shape[1]):
            yr = int(test_years[i]) if k < len(test_months[i]) else None
            # Check if this prediction sample's target month falls in target_year
            try:
                cal_yr = int(test_years[i])
                cal_mo = int(test_months[i, k]) if test_months.shape[1] > k else 0
                # Approximate: use the year from the first target month + step offset
                base_yr = int(test_years[i])
                base_mo = int(test_months[i, 0])
                actual_yr = base_yr + (base_mo + k - 1) // 12
                actual_mo = (base_mo + k - 1) % 12 + 1
                if actual_yr == target_year:
                    pred_a.append((actual_mo, y_pred_a[i, k]))
                    pred_b.append((actual_mo, y_pred_b[i, k]))
                    true_v.append((actual_mo, y_true[i, k]))
            except (IndexError, ValueError):
                continue

    if len(pred_a) >= 12:
        pred_a.sort(key=lambda x: x[0])
        pred_b.sort(key=lambda x: x[0])
        true_v.sort(key=lambda x: x[0])
        return (np.array([v for _, v in pred_a]),
                np.array([v for _, v in pred_b]),
                np.array([v for _, v in true_v]))
    return None, None, None


def find_best_worst_years(y_true, y_pred_a, test_years, test_months):
    """Find years with best and worst per-year RMSE."""
    years = sorted(set(int(y) for y in test_years))
    year_rmse = {}
    for yr in years:
        mask = np.array([int(y) == yr for y in test_years])
        if mask.sum() > 0:
            r = np.sqrt(np.mean((y_true[mask] - y_pred_a[mask])**2))
            year_rmse[yr] = r
    sorted_years = sorted(year_rmse.items(), key=lambda x: x[1])
    return sorted_years[0][0], sorted_years[-1][0]  # best, worst


def plot_bestworst(y_true, y_pred_a, y_pred_b, test_years, test_months, save_path):
    """Two-panel figure: best year and worst year, with multivariate vs univariate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    best_yr, worst_yr = find_best_worst_years(y_true, y_pred_a, test_years, test_months)
    year_pairs = [("Best", best_yr), ("Worst", worst_yr)]

    for ax, (label, yr) in zip(axes, year_pairs):
        pa, pb, tv = get_year_data(y_true, y_pred_a, y_pred_b,
                                   test_years, test_months, yr)
        if pa is not None and len(pa) == 12:
            months = range(1, 13)
            ax.plot(months, tv, 'o-', color='black', linewidth=2.5, markersize=8,
                    label='Observed')
            ax.plot(months, pa, 's--', color='#1b7837', linewidth=2, markersize=8,
                    label='Multivariate')
            ax.plot(months, pb, '^--', color='#2166ac', linewidth=2, markersize=8,
                    label='Univariate (E1)')

            # Compute RMSE for each model in this year
            rmse_a = np.sqrt(np.mean((tv - pa)**2))
            rmse_b = np.sqrt(np.mean((tv - pb)**2))

            ax.text(0.97, 0.12,
                    f'Multivariate RMSE: {rmse_a:.3f}\nUnivariate RMSE: {rmse_b:.3f}',
                    transform=ax.transAxes, fontsize=10, ha='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTHS_CN, fontsize=10)
        ax.set_ylabel('海冰面积 (百万平方公里)', fontsize=12)
        ax.set_title(f'{label} Year: {yr}', fontsize=13, fontweight='bold')
        ax.legend(loc='lower left', framealpha=0.9, fontsize=10)
        ax.grid(alpha=0.3)

    # Subfigure tags
    axes[0].text(0.03, 0.97, '(a)', transform=axes[0].transAxes, fontsize=16,
                 fontweight='bold', va='top')
    axes[1].text(0.03, 0.97, '(b)', transform=axes[1].transAxes, fontsize=16,
                 fontweight='bold', va='top')

    plt.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    # Load predictions from Phase 1 or conditional skill results
    p1_dir = config.RESULTS_PHASE1_DIR
    cond_dir = config.RESULTS_COND_DIR

    y_true, y_pred_a, y_pred_b = None, None, None
    test_years, test_months = None, None

    # Try loading from conditional skill results first (contains full predictions)
    if os.path.exists(cond_dir):
        for fname in os.listdir(cond_dir):
            if fname.endswith(".json"):
                with open(os.path.join(cond_dir, fname), "r") as f:
                    d = json.load(f)
                    if "y_true" in d:
                        y_true = np.array(d["y_true"])
                        y_pred_a = np.array(d["y_pred_a"])
                        y_pred_b = np.array(d["y_pred_b"])
                        test_years = np.array(d.get("test_years", []))
                        test_months = np.array(d.get("test_months", []))
                        break

    if y_true is not None and y_pred_a is not None and y_pred_b is not None:
        out_path = os.path.join(OUTPUT_DIR, "fig_bestworst_multi.png")
        plot_bestworst(y_true, y_pred_a, y_pred_b, test_years, test_months, out_path)
    else:
        print("No prediction data found. Run analysis_conditional_skill.py first.")
        print("Or ensure Phase 1 results include y_pred arrays.")
