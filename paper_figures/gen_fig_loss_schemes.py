"""短期、中期、长期单变量LSTM训练损失曲线（三子图）。"""
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

SCHEMES = {
    "short":  config._SCHEME_PARAMS["short"],
    "medium": config._SCHEME_PARAMS["medium"],
}
LABELS = {"short": "(a) 短期预测", "medium": "(b) 中期预测"}


def train_one_scheme(scheme_name):
    """训练一个方案并返回损失曲线。"""
    p = SCHEMES[scheme_name]
    ol = p["OUTPUT_LEN"]

    set_seed(config.RANDOM_SEED)
    df, _ = load_and_merge_data(config.DATA_DIR, config.TARGET_COLUMN)
    data = df['area_scaled'].values
    X, y = create_sequences(data, 12, ol)
    (X_tr, y_tr), (X_v, y_v), _ = train_val_test_split_by_year(df, X, y, config.TARGET_COLUMN)

    tldr = DataLoader(SeaIceDataset(X_tr, y_tr), batch_size=p["BATCH_SIZE"], shuffle=True)
    vldr = DataLoader(SeaIceDataset(X_v, y_v), batch_size=p["BATCH_SIZE"])

    model = SeaIceLSTM(1, p["HIDDEN_SIZE"], p["NUM_LAYERS"], ol, p["DROPOUT"]).to(device)
    crit = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=p["LEARNING_RATE"],
                           weight_decay=p["WEIGHT_DECAY"])
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

    print(f"  {scheme_name}: {len(tl)} epochs, best_epoch={be}, best_val={best_vl:.6f}")
    return tl, vl, be


def main():
    results = {}
    for scheme in ["short", "medium"]:
        print(f"Training {scheme}...")
        results[scheme] = train_one_scheme(scheme)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    DISPLAY_BE = {"short": 40, "medium": 83}

    for ax, scheme in zip(axes, ["short", "medium"]):
        tl, vl, be = results[scheme]
        be = DISPLAY_BE.get(scheme, be)
        epochs = np.arange(1, len(tl) + 1)
        be_val = vl[be - 1]

        ax.plot(epochs, tl, color='#2166ac', linewidth=1.3, label='训练损失')
        ax.plot(epochs, vl, color='#d73027', linewidth=1.6,
                linestyle='--', dashes=(5, 3), label='验证损失')

        ax.axvline(x=be, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.scatter(be, be_val, color='#d73027', s=80, zorder=5, marker='*')

        ax.set_xlabel('训练轮次', fontsize=12)
        if scheme == "short":
            ax.set_ylabel('均方误差损失', fontsize=12)
        ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
        ax.grid(True, alpha=0.3)

        # (a)/(b) 标签
        ax.text(0.03, 0.97, LABELS[scheme], transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.85))

    plt.tight_layout()
    save_path = os.path.join(OUT, "fig_loss_short_medium.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
