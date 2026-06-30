# gen_fig_combination_heatmap.py
# Fig 4.2: Variable combination RMSE heatmap (pairs vs triples).
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


def build_heatmap_data(phase1_results, phase2_results):
    """Build a matrix of RMSE for variable combinations."""
    # Extract single-variable RMSEs
    single = {}
    for eid, r in phase1_results.items():
        vars_key = eid  # e.g., "E8" -> NAO, "E14" -> AO
        single[eid] = r.get("rmse_mean", r.get("ensemble_rmse", 0))

    # Extract combination RMSEs
    combos = {}
    for eid, r in phase2_results.items():
        combos[eid] = {
            "rmse": r.get("rmse_mean", r.get("ensemble_rmse", 0)),
            "desc": r.get("desc", eid),
            "aux_vars": r.get("aux_vars", []),
        }
    return single, combos


def plot_combination_heatmap(single_rmse, combo_results, save_path):
    """Plot a heatmap-style comparison of variable combinations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Build table data
    all_entries = []

    # Single variables
    for eid, rmse_val in sorted(single_rmse.items(), key=lambda x: x[1]):
        all_entries.append({
            "label": eid, "rmse": rmse_val, "type": "single",
            "n_vars": 1,
        })

    # Combinations
    for eid, info in sorted(combo_results.items(), key=lambda x: x[1]["rmse"]):
        all_entries.append({
            "label": f"{eid} ({'+'.join(info['aux_vars'])})",
            "rmse": info["rmse"], "type": "combo",
            "n_vars": len(info["aux_vars"]),
        })

    # Plot as horizontal bar chart, ordered by RMSE
    all_entries.sort(key=lambda x: x["rmse"])
    labels = [e["label"] for e in all_entries]
    values = [e["rmse"] for e in all_entries]
    n_vars = [e["n_vars"] for e in all_entries]

    colors = []
    for e in all_entries:
        if e["type"] == "single":
            colors.append("#2166ac")
        else:
            # Color by number of variables
            if e["n_vars"] == 2:
                colors.append("#1b7837")
            elif e["n_vars"] == 3:
                colors.append("#d73027")
            else:
                colors.append("#762a83")

    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Mark univariate baseline
    e1_val = single_rmse.get("E1", None)
    if e1_val:
        ax.axvline(x=e1_val, color='#2166ac', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'E1 Univariate: {e1_val:.4f}')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('RMSE (百万平方公里)', fontsize=12)
    ax.invert_yaxis()
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2166ac', label='Single variable'),
        Patch(facecolor='#1b7837', label='2-variable combo'),
        Patch(facecolor='#d73027', label='3-variable combo'),
        Patch(facecolor='#762a83', label='4+ variable combo'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=9)

    ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    p1_dir = config.RESULTS_PHASE1_DIR
    p2_dir = config.RESULTS_PHASE2_DIR

    single_rmse = {}
    combo_rmse = {}

    if os.path.exists(p1_dir):
        for fname in os.listdir(p1_dir):
            if fname.endswith(".json") and not fname.startswith("phase1_summary"):
                with open(os.path.join(p1_dir, fname), "r") as f:
                    d = json.load(f)
                    single_rmse[fname.replace(".json", "")] = {
                        "rmse_mean": d.get("rmse_mean", d.get("ensemble_rmse")),
                    }

    if os.path.exists(p2_dir):
        for fname in os.listdir(p2_dir):
            if fname.endswith(".json") and not fname.startswith("phase2_summary"):
                with open(os.path.join(p2_dir, fname), "r") as f:
                    d = json.load(f)
                    combo_rmse[fname.replace(".json", "")] = {
                        "rmse": d.get("rmse_mean", d.get("ensemble_rmse")),
                        "desc": d.get("desc", ""),
                        "aux_vars": d.get("aux_vars", []),
                    }

    if single_rmse or combo_rmse:
        out_path = os.path.join(OUTPUT_DIR, "fig_combination_heatmap.png")
        plot_combination_heatmap(single_rmse, combo_rmse, out_path)
    else:
        print("No results found. Run Phase 1 and Phase 2 experiments first.")
