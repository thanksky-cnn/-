"""
预测不确定性图 — 基于5-seed E7 ensembles的预测区间
需要先运行 run_optimize_e7.py 获得5个seed的预测结果
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import sys, os, json, numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
from src.data_preprocessing import load_dual_encoder_data, create_dual_targets
from src.model import SeaIceDualEncoderLSTM
from src.utils import set_seed

warnings = __import__('warnings')
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(config.BASE_DIR, "outputs", "plots", "paper")
MODELS_DIR = config.MODELS_DIR
LAGGED_CSV = os.path.join(config.BASE_DIR, "data", "lagged_features.csv")
DE_PARAMS = os.path.join(config.RESULTS_DIR, "dual_encoder_best_params.json")
ol = config.OUTPUT_LEN
target = config.TARGET_COLUMN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
X_main, X_aux, df_de, scaler_de = load_dual_encoder_data(LAGGED_CSV, target)
y_de, years_de = create_dual_targets(df_de, 12, ol, 3, target)
n = min(len(X_main), len(y_de))
X_main, X_aux, y_de, years_de = X_main[:n], X_aux[:n], y_de[:n], years_de[:n]

te_mask = (years_de >= 2016) & (years_de <= 2025)
Xm_te = torch.tensor(X_main[te_mask], dtype=torch.float32)
y_te = y_de[te_mask]
# Build test target months for calendar-month grouping
from src.data_preprocessing import create_sequences
df_months = df_de['month'].values

# Load E7 hyperparameters
with open(DE_PARAMS) as f:
    bp = json.load(f).get("12")
MH = bp.get('main_hidden', 128)
AH = bp.get('aux_hidden', 32)
DR = bp.get('dropout', 0.1)
AD = bp.get('aux_dropout', 0.6)
ASL = bp.get('aux_seq_len', 3)

# Load 5 ensemble models and predict
seeds = [42, 43, 44, 45, 46]
all_preds = []
for s in seeds:
    model_path = os.path.join(MODELS_DIR, f"best_model_de_s{s}.pth")
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found, trying best_model_de.pth")
        model_path = os.path.join(MODELS_DIR, "best_model_de.pth")

    mdl = SeaIceDualEncoderLSTM(
        input_size=1, main_hidden=MH, aux_hidden=AH, aux_input_size=7,
        aux_seq_len=ASL, num_layers=1, output_len=ol,
        dropout=DR, aux_dropout=AD).to(device)

    state = torch.load(model_path, map_location=device)
    mdl.load_state_dict(state)
    mdl.eval()

    Xa_te = torch.tensor(X_aux[te_mask][:, -ASL:, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        yp = mdl(Xm_te.to(device), Xa_te).cpu().numpy()
    y_pred = scaler_de.inverse_transform(yp)
    all_preds.append(y_pred)

all_preds = np.array(all_preds)  # shape: (5, n_samples, 12)
y_true = scaler_de.inverse_transform(y_te)

# Compute ensemble mean and std
ens_mean = all_preds.mean(axis=0)
ens_std = all_preds.std(axis=0)

# Calendar-month grouping for uncertainty
n_samples = len(y_te)
ol_val = y_te.shape[1]
all_target_months = np.array([df_months[12 + i: 12 + i + ol_val] for i in range(n)])
test_target_months = all_target_months[te_mask]

unc_by_month = {m: [] for m in range(1, 13)}
for i in range(n_samples):
    for k in range(ol_val):
        cal_month = int(test_target_months[i, k])
        unc_by_month[cal_month].append(ens_std[i, k])

monthly_mean_unc = np.array([np.mean(unc_by_month[m]) for m in range(1, 13)])
monthly_std_unc = np.array([np.std(unc_by_month[m]) for m in range(1, 13)])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Prediction interval for one representative sample
ax = axes[0]
sample_idx = 0
x_axis = range(1, ol_val + 1)
ax.plot(x_axis, y_true[sample_idx], 'o-', color='#d73027', label='观测值', markersize=6)
ax.plot(x_axis, ens_mean[sample_idx], 's--', color='#2166ac', label='Ensemble均值', markersize=6)
ax.fill_between(x_axis,
                ens_mean[sample_idx] - 1.96 * ens_std[sample_idx],
                ens_mean[sample_idx] + 1.96 * ens_std[sample_idx],
                color='#2166ac', alpha=0.2, label='95%预测区间')
ax.set_xlabel('预测月数', fontsize=10)
ax.set_ylabel('海冰面积 (百万平方公里)', fontsize=10)
ax.legend(loc='lower left', framealpha=0.9, fontsize=8)
ax.set_title(f'测试样本示例 (索引={sample_idx})', fontsize=10)
ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
ax.grid(True, alpha=0.3)

# (b) Monthly mean prediction uncertainty (std across ensemble)
ax2 = axes[1]
months = range(1, 13)
ax2.bar(months, monthly_mean_unc, color='#2166ac', edgecolor='white',
        yerr=monthly_std_unc, capsize=3, alpha=0.8)
ax2.set_xlabel('日历月', fontsize=10)
ax2.set_ylabel('预测不确定性 (Ensemble Std, 百万km²)', fontsize=10)
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig_uncertainty.png")
fig.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
print(f"Mean prediction uncertainty: {np.mean(ens_std):.4f} M km²")
print(f"Monthly mean uncertainty: {monthly_mean_unc}")
