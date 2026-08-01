"""单变量LSTM长期预测 — 训练损失曲线。"""
import sys, os, numpy as np, torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from src.data_preprocessing import load_and_merge_data, create_sequences, train_val_test_split_by_year
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM
from src.utils import set_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import nature_figure_config  # nature-figure: 600 DPI + Arial + clean spines


fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(config.PLOTS_DIR, "paper")
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ol = config.OUTPUT_LEN


def main():
    set_seed(config.RANDOM_SEED)

    df, scaler = load_and_merge_data(config.DATA_DIR, config.TARGET_COLUMN)
    data = df['area_scaled'].values
    X, y = create_sequences(data, 12, ol)
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = \
        train_val_test_split_by_year(df, X, y, config.TARGET_COLUMN)

    tldr = DataLoader(SeaIceDataset(X_tr, y_tr), batch_size=config.BATCH_SIZE, shuffle=True)
    vldr = DataLoader(SeaIceDataset(X_v, y_v), batch_size=config.BATCH_SIZE)

    model = SeaIceLSTM(1, config.HIDDEN_SIZE, config.NUM_LAYERS, ol, config.DROPOUT).to(device)
    crit = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
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

    print(f"Trained {len(tl)} epochs, best_val_loss={best_vl:.6f}, best_epoch={be}")

    epochs = np.arange(1, len(tl) + 1)
    be_val = vl[be - 1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, tl, color='#2166ac', linewidth=1.5, label='训练损失')
    ax.plot(epochs, vl, color='#d73027', linewidth=1.8,
            linestyle='--', dashes=(5, 3), label='验证损失')

    # 收敛点
    ax.axvline(x=be, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=be_val, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.scatter(be, be_val, color='#d73027', s=100, zorder=5, marker='*')
    ax.annotate(f'{be}',
                xy=(be, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -18), textcoords='offset points',
                fontsize=9, color='#d73027', ha='center', va='top', fontweight='bold')
    ax.annotate(f'{be_val:.5f}',
                xy=(0, be_val), xycoords=('axes fraction', 'data'),
                xytext=(-54, 0), textcoords='offset points',
                fontsize=8, color='#d73027', ha='right', va='center')

    ax.set_xlabel('训练轮次', fontsize=13)
    ax.set_ylabel('均方误差损失', fontsize=13)
    ax.legend(fontsize=12, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 后期趋势放大
    zs = max(20, be - 20)
    axins = ax.inset_axes([0.50, 0.22, 0.45, 0.33])
    axins.plot(epochs, tl, color='#2166ac', linewidth=1.0)
    axins.plot(epochs, vl, color='#d73027', linewidth=1.3, linestyle='--', dashes=(5, 3))
    axins.set_xlim(zs, len(tl))
    y_slice = vl[zs - 1:]
    y_min, y_max = min(y_slice), max(y_slice)
    y_rng = y_max - y_min
    axins.set_ylim(y_min - y_rng * 0.25, y_max + y_rng * 0.15)
    axins.axvline(x=be, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    axins.scatter(be, be_val, color='#d73027', s=50, zorder=5, marker='*')
    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.2)

    # 主图矩形框
    x0, x1 = zs, len(tl)
    y_lo_rect, y_hi_rect = y_min - y_rng * 0.25, y_max + y_rng * 0.15
    rect = plt.Rectangle((x0, y_lo_rect), x1 - x0, y_hi_rect - y_lo_rect,
                          linewidth=1, edgecolor='gray', facecolor='none',
                          linestyle='-', alpha=0.5)
    ax.add_patch(rect)

    plt.tight_layout()
    save_path = os.path.join(OUT, "fig_lstm_loss_curve.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
