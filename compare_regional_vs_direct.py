# compare_regional_vs_direct.py
# 纯冰基线对照：验证「分海域建模 > 直接建模」。
#   目标变量 Y_partial = 7 海域面积之和（占全北极 ~60%）
#   路径 A（直接）  : 一个 SeaIceLSTM 直接预测 Y_partial
#   路径 B（分海域）: 7 个 SeaIceLSTM 各自预测 → 求和
#   对比 5-seed ensemble 的 RMSE（同一目标、同一量纲，可比）
import sys, os, json, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import warnings; warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM
from src.utils import set_seed, calculate_metrics

BASE_DIR = config.BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "results", "baseline_direct_vs_regional")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 与 E7V1_HP 的 main 编码器一致的超参数（纯冰单编码器）
HP = {
    "hidden_size": 256, "num_layers": 1, "dropout": 0.1,
    "batch_size": 32, "learning_rate": 0.000635, "weight_decay": 2.59e-06,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_LEN = 12; OUTPUT_LEN = 12
SEEDS = [42, 52, 62, 72, 82]

REGIONS = [
    {"key": "bering",         "name": "白令海"},
    {"key": "chukchi",        "name": "楚科奇海"},
    {"key": "barents",        "name": "巴伦支海"},
    {"key": "kara",           "name": "喀拉海"},
    {"key": "laptev",         "name": "拉普捷夫海"},
    {"key": "greenland",      "name": "格陵兰海"},
    {"key": "central_arctic", "name": "中北冰洋"},
]


def build_series(area_raw, years):
    """从 1D 面积序列构建 (X_main, y, scaler, sample_years)。"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(area_raw.reshape(-1, 1)).flatten()

    n = len(scaled)
    X_list, y_list, yr_list = [], [], []
    for i in range(n - INPUT_LEN - OUTPUT_LEN + 1):
        X_list.append(scaled[i:i + INPUT_LEN])
        y_list.append(scaled[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN])
        yr_list.append(years[i + INPUT_LEN])
    X = np.array(X_list).reshape(-1, INPUT_LEN, 1)
    y = np.array(y_list)
    yrs = np.array(yr_list)
    return X, y, scaler, yrs


def split_by_year(X, y, yrs):
    tr = (yrs >= 1979) & (yrs <= 2010)
    v = (yrs >= 2011) & (yrs <= 2015)
    te = (yrs >= 2016) & (yrs <= 2025)
    return {
        "train": (X[tr], y[tr]), "val": (X[v], y[v]),
        "test": (X[te], y[te]), "test_years": yrs[te],
    }


def train_single(model, tr_ldr, v_ldr, hp, device, num_epochs=1000, patience=30):
    """单编码器 LSTM 训练循环（返回 best_state）。"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=15)

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        tr_loss = 0
        for Xb, yb in tr_ldr:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ldr)

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for Xb, yb in v_ldr:
                Xb, yb = Xb.to(device), yb.to(device)
                v_loss += criterion(model(Xb), yb).item()
        v_loss /= len(v_ldr)
        scheduler.step(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state


def predict_single(model, X, device, scaler):
    """预测并反归一化到原始单位。"""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        yp = model(Xt).cpu().numpy()
    n, ol = yp.shape
    return scaler.inverse_transform(yp.reshape(-1, 1)).reshape(n, ol)


def run_one_series(area_raw, years, name, n_seeds):
    """对单个面积序列（1D）训练 SeaIceLSTM，返回各 seed 的 test 预测 + y_true。"""
    X, y, scaler, yrs = build_series(area_raw, years)
    data = split_by_year(X, y, yrs)

    seed_preds = []
    y_true = None
    for seed in SEEDS[:n_seeds]:
        set_seed(seed)
        tr_ds = SeaIceDataset(*data["train"])
        v_ds = SeaIceDataset(*data["val"])
        tr_ldr = DataLoader(tr_ds, batch_size=HP["batch_size"], shuffle=True)
        v_ldr = DataLoader(v_ds, batch_size=HP["batch_size"])

        model = SeaIceLSTM(
            input_size=1, hidden_size=HP["hidden_size"], num_layers=HP["num_layers"],
            output_len=OUTPUT_LEN, dropout=HP["dropout"],
        ).to(device)

        train_single(model, tr_ldr, v_ldr, HP, device)

        yp = predict_single(model, data["test"][0], device, scaler)
        seed_preds.append(yp)
        if y_true is None:
            y_true = scaler.inverse_transform(
                data["test"][1].reshape(-1, 1)).reshape(-1, OUTPUT_LEN)

    return seed_preds, y_true


def load_aligned_data():
    """加载 7 海域面积，对齐到共同 (year, month) 时间轴，返回 DataFrame。

    返回列: year, month, <7 个海域的 area 列>（inner join 保证各海域时间轴一致）
    """
    df = None
    for r in REGIONS:
        d = pd.read_csv(os.path.join(DATA_DIR, f"{r['key']}_monthly.csv"))
        d = d[d['area'] > -0.005].copy()
        d = d.rename(columns={"area": r["key"]})
        d = d[["year", "month", r["key"]]]
        df = d if df is None else df.merge(d, on=["year", "month"], how="inner")
    return df.sort_values(["year", "month"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="1-seed smoke test")
    args = parser.parse_args()
    n_seeds = 1 if args.fast else args.seeds
    print(f"Device: {device}, seeds={n_seeds}")

    # ── 加载对齐后的数据（所有序列共用同一时间轴）──
    df = load_aligned_data()
    keys = [r["key"] for r in REGIONS]
    years = df["year"].values

    # ── 路径 A：直接预测 Y_partial ──
    print("\n=== 路径 A：直接预测 7 海域之和 ===")
    sum_area = df[keys].sum(axis=1).values
    print(f"  Y_partial 序列: {len(sum_area)} 行, mean={sum_area.mean():.3f} 百万km²")
    preds_A, y_true_A = run_one_series(sum_area, years, "Y_partial", n_seeds)

    # ── 路径 B：分海域预测求和 ──
    print("\n=== 路径 B：7 海域各自预测再求和 ===")
    per_region_preds = {}
    for r in REGIONS:
        area_raw = df[r["key"]].values
        preds, yt = run_one_series(area_raw, years, r["name"], n_seeds)
        per_region_preds[r["key"]] = {"preds": preds, "y_true": yt}
        rmse = float(np.sqrt(np.mean((np.mean(preds, axis=0) - yt) ** 2)))
        print(f"  {r['name']:<6s} 单海域 RMSE={rmse:.6f}")

    # 求和（对齐 test 样本：各海域 test 样本数一致）
    n_test = len(preds_A[0])
    sum_preds_B = []
    for s in range(len(preds_A)):
        regional_sum = np.zeros_like(preds_A[s])
        for r in REGIONS:
            regional_sum += per_region_preds[r["key"]]["preds"][s]
        sum_preds_B.append(regional_sum)

    # ── 对比 ──
    ens_A = np.mean(preds_A, axis=0)
    ens_B = np.mean(sum_preds_B, axis=0)
    rmse_A = float(np.sqrt(np.mean((y_true_A - ens_A) ** 2)))
    rmse_B = float(np.sqrt(np.mean((y_true_A - ens_B) ** 2)))
    mean_target = float(y_true_A.mean())
    nrmse_A = rmse_A / mean_target * 100
    nrmse_B = rmse_B / mean_target * 100

    print("\n" + "=" * 70)
    print("  纯冰基线对照：分海域求和 vs 直接建模")
    print("=" * 70)
    print(f"  目标: 7海域面积之和 (mean={mean_target:.3f} 百万km²)")
    print(f"  路径A 直接建模 RMSE   = {rmse_A:.6f}  (NRMSE {nrmse_A:.2f}%)")
    print(f"  路径B 分海域求和 RMSE = {rmse_B:.6f}  (NRMSE {nrmse_B:.2f}%)")
    print(f"  差值 (B-A) = {rmse_B - rmse_A:+.6f}  ({(rmse_B - rmse_A)/rmse_A*100:+.2f}%)")

    result = {
        "target": "7海域面积之和",
        "target_mean": mean_target,
        "n_seeds": n_seeds,
        "direct_rmse": rmse_A, "direct_nrmse_pct": nrmse_A,
        "regional_sum_rmse": rmse_B, "regional_sum_nrmse_pct": nrmse_B,
        "delta_rmse": rmse_B - rmse_A,
        "delta_pct": (rmse_B - rmse_A) / rmse_A * 100,
        "verdict": "分海域更优" if rmse_B < rmse_A else "直接建模更优",
        "per_region_rmse": {
            r["key"]: float(np.sqrt(np.mean(
                (np.mean(per_region_preds[r["key"]]["preds"], axis=0)
                 - per_region_preds[r["key"]]["y_true"]) ** 2)))
            for r in REGIONS
        },
    }
    out = os.path.join(RESULTS_DIR, "direct_vs_regional.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
