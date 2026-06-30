"""
统计显著性检验 — 配对t检验 + Diebold-Mariano检验，E7 vs E1单变量LSTM
"""
import sys, os, json, numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
from scipy import stats

RESULTS_DIR = config.RESULTS_DIR

def load_predictions():
    """Load E7 and E1 predictions from result CSVs"""
    import pandas as pd
    fp = os.path.join(config.BASE_DIR, "outputs", "experiment_results_2026-05-11",
                      "metrics", "predictions_12m.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        return df
    # Fallback: use summary CSV
    sfp = os.path.join(RESULTS_DIR, "all_experiments_overview.csv")
    if os.path.exists(sfp):
        df = pd.read_csv(sfp)
        print("Loaded overview:", sfp)
        return df
    return None

# Load results
df = load_predictions()
if df is not None:
    print("Results loaded:", df.shape)
    print(df.to_string())

# Also compute from direct result files
ov_path = os.path.join(RESULTS_DIR, "all_experiments_overview.csv")
if os.path.exists(ov_path):
    import pandas as pd
    df_ov = pd.read_csv(ov_path)
    print("\n=== Experiment Overview ===")
    print(df_ov.to_string())

print("\nNote: For full statistical tests, run the models and extract per-sample errors.")
print("Required: E7v1 and E1 prediction arrays on the SAME test samples.")
print("Then compute:")
print("  from scipy import stats")
print("  t_stat, p_value = stats.ttest_rel(abs_errors_e7, abs_errors_e1)")
print("  print(f'Paired t-test: t={t_stat:.4f}, p={p_value:.4f}')")
print()
print("  # Diebold-Mariano test:")
print("  d = errors_e7_sq - errors_e1_sq  # squared error difference per sample")
print("  n = len(d)")
print("  dm_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(n))")
print("  p_dm = 2 * (1 - stats.norm.cdf(abs(dm_stat)))")
print("  print(f'DM test: stat={dm_stat:.4f}, p={p_dm:.4f}')")
