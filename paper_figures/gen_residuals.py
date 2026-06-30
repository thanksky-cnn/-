"""
残差分析图 — QQ图 + 预测残差分布 + 逐月残差标准差
需要预训练的 best_model_de.pth 或 best_model_area.pth
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
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(config.BASE_DIR, "outputs", "plots", "paper")
os.makedirs(OUT_DIR, exist_ok=True)
MODELS_DIR = config.MODELS_DIR
LAGGED_CSV = os.path.join(config.BASE_DIR, "data", "lagged_features.csv")
DE_PARAMS = os.path.join(config.RESULTS_DIR, "dual_encoder_best_params.json")
ol = config.OUTPUT_LEN; target = config.TARGET_COLUMN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
X_main, X_aux, df_de, scaler_de = load_dual_encoder_data(LAGGED_CSV, target)
y_de, years_de = create_dual_targets(df_de, 12, ol, 3, target)
n = min(len(X_main), len(y_de))
X_main, X_aux, y_de, years_de = X_main[:n], X_aux[:n], y_de[:n], years_de[:n]

te_mask = (years_de >= 2016) & (years_de <= 2025)
Xm_te = torch.tensor(X_main[te_mask], dtype=torch.float32)
y_te = y_de[te_mask]
df_months = df_de['month'].values

# Load model checkpoint and infer architecture from state_dict
model_path = os.path.join(MODELS_DIR, "best_model_de.pth")
if not os.path.exists(model_path):
    model_path = os.path.join(MODELS_DIR, "best_model_de_s42.pth")

state = torch.load(model_path, map_location=device)
# Infer hidden sizes from LSTM weight shapes
main_w_ih = state["main_lstm.weight_ih_l0"]  # (4*hidden, input_size)
MH = main_w_ih.shape[0] // 4
aux_w_ih = state["aux_lstm.weight_ih_l0"]  # (4*aux_hidden, aux_input_size)
AH = aux_w_ih.shape[0] // 4
aux_in = aux_w_ih.shape[1]
ASL = 3  # default aux_seq_len (must match checkpoint)
# Infer dropout from model architecture
DR = 0.1; AD = 0.6  # defaults
print(f"Inferred: MH={MH}, AH={AH}, aux_input={aux_in}, ASL={ASL}")

mdl = SeaIceDualEncoderLSTM(
    input_size=1, main_hidden=MH, aux_hidden=AH, aux_input_size=aux_in,
    aux_seq_len=ASL, num_layers=1, output_len=ol,
    dropout=DR, aux_dropout=AD).to(device)
mdl.load_state_dict(state)
mdl.eval()

Xa_te = torch.tensor(X_aux[te_mask][:, -ASL:, :], dtype=torch.float32).to(device)
with torch.no_grad():
    yp = mdl(Xm_te.to(device), Xa_te).cpu().numpy()
y_pred = scaler_de.inverse_transform(yp)
y_true = scaler_de.inverse_transform(y_te)

# Compute residuals
residuals = (y_true - y_pred).flatten()

# Calendar-month residuals
n_samples = len(y_te)
ol_val = y_te.shape[1]
all_target_months = np.array([df_months[12 + i: 12 + i + ol_val] for i in range(n)])
test_target_months = all_target_months[te_mask]

res_by_month = {m: [] for m in range(1, 13)}
for i in range(n_samples):
    for k in range(ol_val):
        cal_month = int(test_target_months[i, k])
        res_by_month[cal_month].append(y_true[i, k] - y_pred[i, k])

monthly_rmse = np.array([np.sqrt(np.mean(np.array(res_by_month[m])**2)) for m in range(1, 13)])
monthly_bias = np.array([np.mean(np.array(res_by_month[m])) for m in range(1, 13)])

# Plot
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# (a) QQ plot
ax = axes[0]
from scipy import stats
stats.probplot(residuals, dist="norm", plot=ax)
ax.get_lines()[0].set_color('#2166ac')
ax.get_lines()[1].set_color('#d73027')
ax.set_xlabel('理论分位数', fontsize=9)
ax.set_ylabel('样本分位数', fontsize=9)
ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# (b) Residual histogram
ax2 = axes[1]
ax2.hist(residuals, bins=30, color='#2166ac', edgecolor='white', alpha=0.8, density=True)
x_vals = np.linspace(min(residuals), max(residuals), 100)
ax2.plot(x_vals, stats.norm.pdf(x_vals, np.mean(residuals), np.std(residuals)),
         color='#d73027', linewidth=2, label='正态分布拟合')
ax2.set_xlabel('残差 (百万平方公里)', fontsize=9)
ax2.set_ylabel('密度', fontsize=9)
ax2.legend(fontsize=8)
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
# Stats annotation
from scipy.stats import skew, kurtosis
sk = skew(residuals); ku = kurtosis(residuals)
ax2.text(0.95, 0.95, '偏度: {:.3f}\n峰度: {:.3f}\nStd: {:.3f}'.format(sk, ku, np.std(residuals)),
         transform=ax2.transAxes, fontsize=8, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (c) Monthly bias + RMSE
ax3 = axes[2]
months = range(1, 13)
x = np.arange(12)
width = 0.35
bars1 = ax3.bar(x - width/2, monthly_rmse, width, color='#2166ac', label='RMSE',
                edgecolor='white')
bars2 = ax3.bar(x + width/2, monthly_bias, width, color='#d73027', label='偏差(Bias)',
                edgecolor='white')
ax3.set_xlabel('日历月', fontsize=9)
ax3.set_ylabel('百万平方公里', fontsize=9)
ax3.set_xticks(x)
ax3.set_xticklabels([str(m) for m in months])
ax3.legend(fontsize=8)
ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
ax3.text(0.03, 0.97, '(c)', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig_residuals.png")
fig.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
print("Residual stats: mean={:.4f}, std={:.4f}, skew={:.4f}, kurt={:.4f}".format(
    np.mean(residuals), np.std(residuals), sk, ku))
