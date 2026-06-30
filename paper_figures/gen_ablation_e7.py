"""
图4.x: E7 Ablation Study
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import sys, os, json, time, warnings, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
from src.data_preprocessing import load_dual_encoder_data, create_dual_targets
from src.model import SeaIceDualEncoderLSTM, SeaIceLSTM, count_parameters
from src.utils import set_seed, calculate_metrics
warnings.filterwarnings('ignore')

LAGGED_CSV = os.path.join(config.BASE_DIR, "data", "lagged_features.csv")
DE_PARAMS = os.path.join(config.RESULTS_DIR, "dual_encoder_best_params.json")
ol = config.OUTPUT_LEN
target = config.TARGET_COLUMN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = os.path.join(config.BASE_DIR, 'outputs', 'plots', 'paper')

# Hyperparameters
try:
    with open(DE_PARAMS) as f: bp = json.load(f).get("12")
except: bp = {}
MH = bp.get("main_hidden",128); AH = bp.get("aux_hidden",32)
DR = bp.get("dropout",0.1); ADR = bp.get("aux_dropout",0.6)
LR = bp.get("lr",0.0026); BS = bp.get("batch_size",8)
WD = bp.get("weight_decay",5e-6); ASL = bp.get("aux_seq_len",3)
SEEDS = [42,43,44,45,46]; EPOCHS = 200
set_seed(config.RANDOM_SEED)

print('Loading data for ablation study...')
X_main, X_aux, df_de, scaler_de = load_dual_encoder_data(LAGGED_CSV, target)
y_de, years_de = create_dual_targets(df_de, 12, ol, 3, target)
n = min(len(X_main), len(y_de))
X_main, X_aux, y_de, years_de = X_main[:n], X_aux[:n], y_de[:n], years_de[:n]
tr = (years_de>=1979)&(years_de<=2010)
vl = (years_de>=2011)&(years_de<=2015)
te = (years_de>=2016)&(years_de<=2025)
Xmt,Xat,yt = X_main[tr],X_aux[tr],y_de[tr]
Xmv,Xav,yv = X_main[vl],X_aux[vl],y_de[vl]
Xmte,Xate,yte = X_main[te],X_aux[te],y_de[te]
print(f'Train:{Xmt.shape[0]} Val:{Xmv.shape[0]} Test:{Xmte.shape[0]}')

class DED(Dataset):
    def __init__(s,Xm,Xa,y):
        s.Xm=torch.tensor(Xm,dtype=torch.float32)
        s.Xa=torch.tensor(Xa,dtype=torch.float32)
        s.y=torch.tensor(y,dtype=torch.float32)
    def __len__(s): return len(s.Xm)
    def __getitem__(s,i): return s.Xm[i],s.Xa[i],s.y[i]

def train_abl(name, Xa_tr, Xa_v, Xa_te, aux_n, aux_dr=ADR, single_enc=False):
    preds, times = [], []
    for si, s in enumerate(SEEDS):
        set_seed(s)
        Xas_tr = Xa_tr[:,-ASL:,:]; Xas_v = Xa_v[:,-ASL:,:]; Xas_te = Xa_te[:,-ASL:,:]
        tds = DED(Xmt, Xas_tr, yt); vds = DED(Xmv, Xas_v, yv)
        tdl = DataLoader(tds, BS, True); vdl = DataLoader(vds, BS)
        if single_enc:
            mdl = SeaIceLSTM(1+aux_n, MH, 1, ol, DR).to(device)
        else:
            mdl = SeaIceDualEncoderLSTM(1, MH, AH, aux_n, ASL, 1, ol, DR, aux_dr).to(device)
        crit = nn.MSELoss()
        opt = torch.optim.AdamW(mdl.parameters(), LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.7, 30)
        best_vl = float("inf"); patience = 0; best_state = None; t0 = time.time()
        for ep in range(EPOCHS):
            mdl.train(); tl = 0
            for Xb, Xab, yb in tdl:
                Xb,yb = Xb.to(device),yb.to(device); Xab = Xab.to(device); opt.zero_grad()
                if single_enc:
                    loss = crit(mdl(torch.cat([Xb, torch.cat([torch.zeros(Xb.shape[0],Xb.shape[1]-Xab.shape[1],Xab.shape[2],device=device), Xab[:,:,:aux_n]], dim=1)], -1)), yb)
                else:
                    loss = crit(mdl(Xb,Xab), yb)
                loss.backward()
                if config.GRAD_CLIP_NORM:
                    torch.nn.utils.clip_grad_norm_(mdl.parameters(), config.GRAD_CLIP_NORM)
                opt.step(); tl += loss.item()
            tl /= len(tdl); mdl.eval(); vl_val = 0
            with torch.no_grad():
                for Xb, Xab, yb in vdl:
                    Xb,yb = Xb.to(device),yb.to(device); Xab = Xab.to(device)
                    if single_enc:
                        vl_val += crit(mdl(torch.cat([Xb, torch.cat([torch.zeros(Xb.shape[0],Xb.shape[1]-Xab.shape[1],Xab.shape[2],device=device), Xab[:,:,:aux_n]], dim=1)], -1)), yb).item()
                    else:
                        vl_val += crit(mdl(Xb,Xab), yb).item()
            vl_val /= len(vdl); sched.step(vl_val)
            if vl_val < best_vl:
                best_vl = vl_val; patience = 0
                best_state = {k: v.cpu().clone() for k,v in mdl.state_dict().items()}
            else:
                patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                break
        train_time = time.time() - t0
        mdl.load_state_dict(best_state); mdl.eval()
        with torch.no_grad():
            Xtt = torch.tensor(Xmte, dtype=torch.float32).to(device)
            Xat_t = torch.tensor(Xas_te, dtype=torch.float32).to(device)
            if single_enc:
                pad_t = torch.zeros(Xtt.shape[0], Xtt.shape[1]-Xat_t.shape[1], Xat_t.shape[2], device=device); Xat_pad = torch.cat([pad_t, Xat_t[:,:,:aux_n]], dim=1); yp = mdl(torch.cat([Xtt, Xat_pad], -1)).cpu().numpy()
            else:
                yp = mdl(Xtt,Xat_t).cpu().numpy()
        y_pred = scaler_de.inverse_transform(yp)
        preds.append(y_pred); times.append(train_time)
        print(f"  {name}[{si}] S={s}: ep={ep+1} vl={best_vl:.5f} t={train_time:.0f}s")
    return preds, times

print("\n" + "="*60)
print("Ablation Experiment: 5 variants x 5 seeds")
print("="*60)

print("\n[1/5] E7-full (baseline)")
p0,t0 = train_abl("full", Xat, Xav, Xate, 7, ADR, False)

print("\n[2/5] E7-no-lag: aux only 3 channels (ao, sst, mask)")
X_nl = np.concatenate([X_aux[:,:,0:1], X_aux[:,:,1:2], X_aux[:,:,6:7]], -1)
p1,t1 = train_abl("nolag", X_nl[tr], X_nl[vl], X_nl[te], 3, ADR, False)

print("\n[3/5] E7-no-mask: aux 6 channels (no SST mask)")
X_nm = X_aux[:,:,:6]
p2,t2 = train_abl("nomask", X_nm[tr], X_nm[vl], X_nm[te], 6, ADR, False)

print("\n[4/5] E7-single-enc: merged into single LSTM")
p3,t3 = train_abl("single", Xat, Xav, Xate, 7, ADR, True)

print("\n[5/5] E7-no-auxdrop: aux_dropout=0")
p4,t4 = train_abl("nodrop", Xat, Xav, Xate, 7, 0.0, False)

# Results
print("\n" + "="*80)
print("ABLATION RESULTS")
print("="*80)
order = ["full","nolag","nomask","single","nodrop"]
cn = {"full":"E7(基线)","nolag":"移除滞后","nomask":"移除掩码","single":"单编码器","nodrop":"aux D=0"}
allp = {"full":p0,"nolag":p1,"nomask":p2,"single":p3,"nodrop":p4}
yt = scaler_de.inverse_transform(yte)
rs = {}
for k in order:
    ens = np.mean(allp[k], 0)
    m = calculate_metrics(yt, ens)
    m["mrmse"] = [np.sqrt(np.mean((ens[:,i]-yt[:,i])**2)) for i in range(ol)]
    rs[k] = m
print("\n{:<16} {:<8} {:<8} {:<7} {:<8}".format("Variant","RMSE","MAE","MAPE","R2"))
print("-"*50)
for k in order:
    m = rs[k]
    print("{:<16} {:<8.4f} {:<8.4f} {:<7.2f} {:<8.4f}".format(cn[k],m["rmse"],m["mae"],m["mape"],m["r2"]))
base_rmse = rs["full"]["rmse"]
print("\nBaseline RMSE = {:.4f}".format(base_rmse))
for k in order[1:]:
    d = rs[k]["rmse"] - base_rmse
    print("  {}: {:+.4f} ({:+.1f}%)".format(cn[k], d, d/base_rmse*100))

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 9))
ax = axes[0]
lbl = [cn[k] for k in order]
vals = [rs[k]["rmse"] for k in order]
clr = ["#2166ac","#d73027","#fc8d59","#f46d43","#fdae61"]
bars = ax.bar(range(5), vals, color=clr, edgecolor="white", linewidth=0.5, width=0.6)
ax.set_xticks(range(5)); ax.set_xticklabels(lbl, fontsize=9)
ax.set_ylabel("RMSE (M km\u00b2)", fontsize=11)
ax.set_ylim(0, max(vals)*1.15)
for br, v in zip(bars, vals):
    ax.text(br.get_x()+br.get_width()/2, br.get_height()+0.002, "{:.4f}".format(v),
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(y=base_rmse, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(0.03,0.97,"(a)",transform=ax.transAxes,fontsize=14,fontweight="bold",va="top")
ax2 = axes[1]; months = range(1, ol+1)
styles = ["o-","s--","^-.","d:","x-"]
for i,k in enumerate(order):
    ax2.plot(months, rs[k]["mrmse"], styles[i], color=clr[i],
             label=cn[k], markersize=5, linewidth=1.3)
ax2.set_xlabel("Lead Month", fontsize=11)
ax2.set_ylabel("RMSE (M km\u00b2)", fontsize=11)
ax2.legend(loc="upper left", framealpha=0.9, fontsize=8)
ax2.set_xlim(0.5, ol+0.5); ax2.set_xticks(months); ax2.grid(True, alpha=0.3)
ax2.text(0.03,0.97,"(b)",transform=ax2.transAxes,fontsize=14,fontweight="bold",va="top")
plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "fig_ablation_e7.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
print("\nSaved: {}".format(out_path))
import pandas as pd
csv_path = os.path.join(config.RESULTS_DIR, "ablation_e7_results.csv")
rows = []
for k in order:
    rows.append({"variant":cn[k],"rmse":rs[k]["rmse"],"mae":rs[k]["mae"],"mape":rs[k]["mape"],"r2":rs[k]["r2"]})
pd.DataFrame(rows).to_csv(csv_path, index=False)
print("Saved: {}".format(csv_path))
print("\nDone!")