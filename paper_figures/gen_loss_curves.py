"""三变量 LSTM vs 双编码器 LSTM — 损失曲线对比。"""

import sys, os, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from src.data_preprocessing import (
    load_trivariate_data, load_dual_encoder_data, create_sequences, create_dual_targets,
    train_val_test_split_by_year
)
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM, SeaIceDualEncoderLSTM
from src.utils import set_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

AO_CSV = os.path.join(config.BASE_DIR, "data", "ao_monthly.csv")
SST_CSV = os.path.join(config.BASE_DIR, "data", "arctic_sst_monthly.csv")
LAGGED_CSV = os.path.join(config.BASE_DIR, "data", "lagged_features.csv")
TRI_PARAMS_PATH = os.path.join(config.RESULTS_DIR, "trivariate_best_params.json")
DE_PARAMS_PATH = os.path.join(config.RESULTS_DIR, "dual_encoder_best_params.json")

OUT = os.path.join(config.PLOTS_DIR, "paper")
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ==================== 1. Trivariate training ====================

def train_trivariate_loss(ol=12):
    """Train trivariate LSTM and return (train_losses, val_losses)."""
    with open(TRI_PARAMS_PATH, 'r') as f:
        tri_p = json.load(f)[str(ol)]

    set_seed(42)

    df3, sc_ice, sc_ao, sc_sst = load_trivariate_data(
        config.DATA_DIR, AO_CSV, SST_CSV, config.TARGET_COLUMN)
    data_in = df3[[f'{config.TARGET_COLUMN}_scaled', 'ao_scaled', 'sst_scaled']].values
    data_tgt = df3[f'{config.TARGET_COLUMN}_scaled'].values
    X, y = create_sequences(data_in, 12, ol, target_data=data_tgt)

    (Xt, yt), (Xv, yv), _ = train_val_test_split_by_year(df3, X, y, config.TARGET_COLUMN)

    b = tri_p['batch_size']
    tldr = DataLoader(SeaIceDataset(Xt, yt), batch_size=b, shuffle=True)
    vldr = DataLoader(SeaIceDataset(Xv, yv), batch_size=b)

    model = SeaIceLSTM(
        input_size=3, hidden_size=tri_p['hidden_size'],
        num_layers=tri_p['num_layers'], output_len=ol,
        dropout=tri_p['dropout']).to(device)

    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=tri_p['learning_rate'],
                           weight_decay=tri_p['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.7, 30)

    tl, vl = [], []
    best_vl = float('inf')
    patience = 0
    be = 0

    for ep in range(1000):
        model.train()
        tl2 = 0
        for Xb, yb in tldr:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(Xb), yb).backward()
            opt.step()
            tl2 += crit(model(Xb), yb).item()
        tl.append(tl2 / len(tldr))

        model.eval()
        vl2 = 0
        with torch.no_grad():
            for Xb, yb in vldr:
                Xb, yb = Xb.to(device), yb.to(device)
                vl2 += crit(model(Xb), yb).item()
        vl2 /= len(vldr)
        vl.append(vl2)
        sched.step(vl2)

        if vl2 < best_vl:
            best_vl = vl2
            patience = 0
            be = ep + 1
        else:
            patience += 1
        if patience >= 30:
            break

    print(f"Trivariate: trained {len(tl)} epochs, best_val_loss={best_vl:.6f}, best_epoch={be}")
    return tl, vl, be


# ==================== 2. Dual-Encoder training ====================


class DualEncoderDataset(Dataset):
    def __init__(self, X_main, X_aux, y):
        self.X_main = torch.tensor(X_main, dtype=torch.float32)
        self.X_aux = torch.tensor(X_aux, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_main)

    def __getitem__(self, idx):
        return self.X_main[idx], self.X_aux[idx], self.y[idx]


def train_dual_encoder_loss(ol=12):
    """Train dual-encoder LSTM and return (train_losses, val_losses)."""
    with open(DE_PARAMS_PATH, 'r') as f:
        de_p = json.load(f)[str(ol)]

    set_seed(42)

    X_main, X_aux, df_de, scaler_de = load_dual_encoder_data(
        LAGGED_CSV, config.TARGET_COLUMN)
    y_de, years_de = create_dual_targets(df_de, 12, ol, de_p.get('aux_seq_len', 3),
                                         config.TARGET_COLUMN)
    n = min(len(X_main), len(y_de))
    X_main, X_aux, y_de, years_de = X_main[:n], X_aux[:n], y_de[:n], years_de[:n]

    # Year-based split
    train_m = (years_de >= 1979) & (years_de <= 2010)
    val_m = (years_de >= 2011) & (years_de <= 2015)
    Xm_tr, Xa_tr, y_tr = X_main[train_m], X_aux[train_m], y_de[train_m]
    Xm_v, Xa_v, y_v = X_main[val_m], X_aux[val_m], y_de[val_m]

    b = de_p['batch_size']
    tldr = DataLoader(DualEncoderDataset(Xm_tr, Xa_tr, y_tr),
                      batch_size=b, shuffle=True)
    vldr = DataLoader(DualEncoderDataset(Xm_v, Xa_v, y_v), batch_size=b)

    model = SeaIceDualEncoderLSTM(
        input_size=1, main_hidden=de_p['main_hidden'],
        aux_hidden=de_p['aux_hidden'], aux_input_size=7,
        aux_seq_len=de_p.get('aux_seq_len', 3), num_layers=1,
        output_len=ol, dropout=de_p['dropout'],
        aux_dropout=de_p['aux_dropout']).to(device)

    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=de_p['lr'],
                           weight_decay=de_p['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.7, 30)

    tl, vl = [], []
    best_vl = float('inf')
    patience = 0
    be = 0

    for ep in range(1000):
        model.train()
        tl2 = 0
        for Xm_b, Xa_b, yb in tldr:
            Xm_b, Xa_b, yb = Xm_b.to(device), Xa_b.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(Xm_b, Xa_b), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl2 += crit(model(Xm_b, Xa_b), yb).item()
        tl.append(tl2 / len(tldr))

        model.eval()
        vl2 = 0
        with torch.no_grad():
            for Xm_b, Xa_b, yb in vldr:
                Xm_b, Xa_b, yb = Xm_b.to(device), Xa_b.to(device), yb.to(device)
                vl2 += crit(model(Xm_b, Xa_b), yb).item()
        vl2 /= len(vldr)
        vl.append(vl2)
        sched.step(vl2)

        if vl2 < best_vl:
            best_vl = vl2
            patience = 0
            be = ep + 1
        else:
            patience += 1
        if patience >= 30:
            break

    print(f"Dual-Encoder: trained {len(tl)} epochs, best_val_loss={best_vl:.6f}, best_epoch={be}")
    return tl, vl, be


# ==================== 3. Plot ====================

def _add_zoom_inset(ax, tl, vl, be, color_train, color_val, zoom_start):
    """在 ax 上添加后期局部放大的 inset 子图。"""
    # 聚焦收敛趋势：从 best_epoch 前20轮开始到结束
    zs = max(zoom_start, be - 20)
    zs = min(zs, be - 5)  # 确保至少能看到一些收敛前的变化
    axins = ax.inset_axes([0.52, 0.24, 0.42, 0.30])
    ep = range(1, len(tl) + 1)
    axins.plot(ep, tl, color=color_train, linewidth=1.0)
    axins.plot(ep, vl, color=color_val, linewidth=1.3,
               linestyle='--', dashes=(5, 3))
    axins.set_xlim(zs, len(tl))
    # y 轴范围：聚焦在后期验证损失的波动区间
    y_slice = vl[zs - 1:]
    y_min, y_max = min(y_slice), max(y_slice)
    y_rng = y_max - y_min
    axins.set_ylim(y_min - y_rng * 0.25, y_max + y_rng * 0.15)
    # 收敛点
    val_be = vl[be - 1]
    axins.axvline(x=be, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    axins.axhline(y=val_be, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    axins.scatter(be, val_be, color=color_val, s=40, zorder=5, marker='*')
    axins.tick_params(labelsize=6)
    axins.grid(True, alpha=0.2)
    # 在主图上画矩形框标记放大区域
    x0, x1 = zs, len(tl)
    y_lo_rect, y_hi_rect = y_min - y_rng * 0.25, y_max + y_rng * 0.15
    rect = plt.Rectangle((x0, y_lo_rect), x1 - x0, y_hi_rect - y_lo_rect,
                          linewidth=1, edgecolor='gray', facecolor='none',
                          linestyle='-', alpha=0.5)
    ax.add_patch(rect)


def plot_loss_curves(tri_tl, tri_vl, tri_be, de_tl, de_vl, de_be, save_path):
    """绘制三变量 vs 双编码器损失曲线，含后期局部放大。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    tri_val = tri_vl[tri_be - 1]
    de_val = de_vl[de_be - 1]
    zoom_start = 20  # 第20轮之后为"后期"

    # --- Left: 三变量 LSTM ---
    ax = axes[0]
    ax.plot(range(1, len(tri_tl) + 1), tri_tl, color='#2166ac', linewidth=1.5,
            label='训练损失')
    ax.plot(range(1, len(tri_vl) + 1), tri_vl, color='#d73027', linewidth=1.5,
            linestyle='--', dashes=(5, 3), label='验证损失')
    ax.axvline(x=tri_be, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=tri_val, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.scatter(tri_be, tri_val, color='#d73027', s=80, zorder=5, marker='*')
    ax.annotate(f'{tri_be}',
                xy=(tri_be, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -18), textcoords='offset points',
                fontsize=9, color='#d73027', ha='center', va='top', fontweight='bold')
    ax.annotate(f'{tri_val:.5f}',
                xy=(0, tri_val), xycoords=('axes fraction', 'data'),
                xytext=(-54, 0), textcoords='offset points',
                fontsize=8, color='#d73027', ha='right', va='center')
    ax.set_xlabel('训练轮次', fontsize=12)
    ax.set_ylabel('均方误差损失', fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')
    _add_zoom_inset(ax, tri_tl, tri_vl, tri_be, '#2166ac', '#d73027', zoom_start)

    # --- Right: 双编码器 LSTM ---
    ax = axes[1]
    ax.plot(range(1, len(de_tl) + 1), de_tl, color='#2166ac', linewidth=1.5,
            label='训练损失')
    ax.plot(range(1, len(de_vl) + 1), de_vl, color='#d73027', linewidth=1.5,
            linestyle='--', dashes=(5, 3), label='验证损失')
    ax.axvline(x=de_be, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=de_val, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.scatter(de_be, de_val, color='#d73027', s=80, zorder=5, marker='*')
    ax.annotate(f'{de_be}',
                xy=(de_be, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -18), textcoords='offset points',
                fontsize=9, color='#d73027', ha='center', va='top', fontweight='bold')
    ax.annotate(f'{de_val:.5f}',
                xy=(0, de_val), xycoords=('axes fraction', 'data'),
                xytext=(-54, 0), textcoords='offset points',
                fontsize=8, color='#d73027', ha='right', va='center')
    ax.set_xlabel('训练轮次', fontsize=12)
    ax.set_ylabel('均方误差损失', fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.97, '(b)', transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')
    _add_zoom_inset(ax, de_tl, de_vl, de_be, '#2166ac', '#d73027', zoom_start)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # --- 合并版本：四线叠加 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, len(tri_tl) + 1), tri_tl, color='#2166ac', linewidth=1.2,
            alpha=0.5, label='三变量 — 训练损失')
    ax.plot(range(1, len(tri_vl) + 1), tri_vl, color='#d73027', linewidth=1.8,
            linestyle='--', dashes=(5, 3), label='三变量 — 验证损失')
    ax.plot(range(1, len(de_tl) + 1), de_tl, color='#4393c3', linewidth=1.2,
            alpha=0.5, label='双编码器 — 训练损失')
    ax.plot(range(1, len(de_vl) + 1), de_vl, color='#f4a582', linewidth=1.8,
            linestyle='--', dashes=(5, 3), label='双编码器 — 验证损失')
    # 收敛标注
    ax.axvline(x=tri_be, color='#d73027', linestyle=':', linewidth=1, alpha=0.4)
    ax.axvline(x=de_be, color='#f4a582', linestyle=':', linewidth=1, alpha=0.4)
    ax.axhline(y=tri_val, color='#d73027', linestyle=':', linewidth=1, alpha=0.4)
    ax.axhline(y=de_val, color='#f4a582', linestyle=':', linewidth=1, alpha=0.4)
    ax.scatter(tri_be, tri_val, color='#d73027', s=80, zorder=5, marker='*')
    ax.scatter(de_be, de_val, color='#f4a582', s=80, zorder=5, marker='*')
    ax.annotate(f'{tri_be}',
                xy=(tri_be, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -12), textcoords='offset points',
                fontsize=8, color='#d73027', ha='center', va='top', fontweight='bold')
    ax.annotate(f'{de_be}',
                xy=(de_be, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -24), textcoords='offset points',
                fontsize=8, color='#f4a582', ha='center', va='top', fontweight='bold')
    ax.annotate(f'{tri_val:.5f}',
                xy=(0, tri_val), xycoords=('axes fraction', 'data'),
                xytext=(-58, 0), textcoords='offset points',
                fontsize=8, color='#d73027', ha='right', va='center')
    ax.annotate(f'{de_val:.5f}',
                xy=(0, de_val), xycoords=('axes fraction', 'data'),
                xytext=(-58, 0), textcoords='offset points',
                fontsize=8, color='#f4a582', ha='right', va='center')
    ax.set_xlabel('训练轮次', fontsize=12)
    ax.set_ylabel('均方误差损失', fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    # 组合图的放大区域：聚焦收敛趋势
    zs_comb = max(20, min(tri_be, de_be) - 20)
    max_ep = max(len(tri_tl), len(de_tl))
    # y 范围：取两个模型在 zs_comb 之后的 val loss 范围
    y_tri_slice = tri_vl[zs_comb - 1:]
    y_de_slice = de_vl[zs_comb - 1:]
    y_min_z = min(min(y_tri_slice), min(y_de_slice))
    y_max_z = max(max(y_tri_slice), max(y_de_slice))
    y_rng = y_max_z - y_min_z
    y_lo_z, y_hi_z = y_min_z - y_rng * 0.25, y_max_z + y_rng * 0.15
    rect = plt.Rectangle((zs_comb, y_lo_z), max_ep - zs_comb, y_hi_z - y_lo_z,
                          linewidth=1, edgecolor='gray', facecolor='none',
                          linestyle='-', alpha=0.5)
    ax.add_patch(rect)
    # 放大inset
    axins = ax.inset_axes([0.36, 0.26, 0.42, 0.30])
    axins.plot(range(1, len(tri_tl) + 1), tri_tl, color='#2166ac', linewidth=1.0, alpha=0.5)
    axins.plot(range(1, len(tri_vl) + 1), tri_vl, color='#d73027', linewidth=1.3,
               linestyle='--', dashes=(5, 3))
    axins.plot(range(1, len(de_tl) + 1), de_tl, color='#4393c3', linewidth=1.0, alpha=0.5)
    axins.plot(range(1, len(de_vl) + 1), de_vl, color='#f4a582', linewidth=1.3,
               linestyle='--', dashes=(5, 3))
    axins.set_xlim(zs_comb, max_ep)
    axins.set_ylim(y_lo_z, y_hi_z)
    axins.axvline(x=tri_be, color='#d73027', linestyle=':', linewidth=0.7, alpha=0.5)
    axins.axvline(x=de_be, color='#f4a582', linestyle=':', linewidth=0.7, alpha=0.5)
    axins.scatter(tri_be, tri_val, color='#d73027', s=40, zorder=5, marker='*')
    axins.scatter(de_be, de_val, color='#f4a582', s=40, zorder=5, marker='*')
    axins.tick_params(labelsize=6)
    axins.grid(True, alpha=0.2)
    plt.tight_layout()
    combined_path = save_path.replace('.png', '_combined.png')
    plt.savefig(combined_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {combined_path}")


# ==================== Main ====================

if __name__ == "__main__":
    ol = config.OUTPUT_LEN
    print(f"Output length: {ol} months")

    print("\n[1/3] Training Trivariate LSTM ...")
    tri_tl, tri_vl, tri_be = train_trivariate_loss(ol)

    print("\n[2/3] Training Dual-Encoder LSTM ...")
    de_tl, de_vl, de_be = train_dual_encoder_loss(ol)

    print("\n[3/3] Plotting loss curves ...")
    save_path = os.path.join(OUT, f"loss_tri_vs_de_out{ol}.png")
    plot_loss_curves(tri_tl, tri_vl, tri_be, de_tl, de_vl, de_be, save_path)

    print("\nDone!")
