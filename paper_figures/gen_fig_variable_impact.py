# gen_fig_variable_impact.py
# Fig 4.1: Single-variable incremental experiment RMSE bar chart.
# Compares: E1 (univariate), E7v1 (AO+SST), E8-E14 (each new variable alone).
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

def plot_variable_impact(results_dict, save_path):
    """
    Args:
        results_dict: {exp_id: {"rmse_mean": float, "rmse_std": float, "desc": str, "n_aux_ch": int}}
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Sort by RMSE (lower = better)
    sorted_items = sorted(results_dict.items(), key=lambda x: x[1].get("rmse_mean", 999))
    labels = [f"{eid}\n{item['desc']}" for eid, item in sorted_items]
    values = [item["rmse_mean"] for eid, item in sorted_items]
    stds = [item.get("rmse_std", 0) for eid, item in sorted_items]

    colors = []
    for eid, _ in sorted_items:
        if eid == "E1":
            colors.append("#2166ac")       # Blue: univariate baseline
        elif eid == "E7v1_ref":
            colors.append("#1b7837")       # Green: E7 reference
        else:
            colors.append("#d73027")       # Red: new variables

    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=stds, color=colors, edgecolor="black", linewidth=0.5,
                  capsize=4, alpha=0.85)

    # E1 baseline line
    e1_val = results_dict.get("E1", {}).get("rmse_mean", None)
    if e1_val:
        ax.axhline(y=e1_val, color="#2166ac", linestyle="--", linewidth=1.5,
                   alpha=0.6, label=f"Univariate baseline (RMSE={e1_val:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("RMSE (百万平方公里)", fontsize=12)
    ax.set_xlabel("实验", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Subfigure tag
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")


def load_phase1_results(results_dir):
    results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith(".json") and not fname.startswith("phase1_summary"):
            exp_id = fname.replace(".json", "")
            with open(os.path.join(results_dir, fname), "r") as f:
                data = json.load(f)
                results[exp_id] = {
                    "rmse_mean": data.get("rmse_mean", data.get("ensemble_rmse")),
                    "rmse_std": data.get("rmse_std", 0),
                    "desc": data.get("desc", exp_id),
                    "n_aux_ch": data.get("n_aux_ch", 0),
                }
    return results


if __name__ == "__main__":
    results_dir = config.RESULTS_PHASE1_DIR
    if os.path.exists(results_dir):
        results = load_phase1_results(results_dir)
        if results:
            out_path = os.path.join(OUTPUT_DIR, "fig_variable_impact.png")
            plot_variable_impact(results, out_path)
        else:
            print(f"No Phase 1 results found in {results_dir}")
            print("Run run_phase1_single_variable.py first.")
    else:
        print(f"Results directory not found: {results_dir}")
        print("Run run_phase1_single_variable.py first to generate results.")
