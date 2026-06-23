# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arctic Sea Ice Prediction using LSTM neural networks. Forecasts sea ice area/extent for the next 12 months based on historical monthly NSIDC data (1979-present). Also integrates Arctic Oscillation (AO) index and Arctic sea surface temperature (SST) as auxiliary predictors via a dual-encoder architecture.

## Script Hierarchy (visible in filenames)

```
config.py              ← configuration hub
main.py                ← core training pipeline
compare_models.py      ← baseline comparison (LR vs RNN vs LSTM)

run_*.py               ← ⭐ primary experiments (main results)
  run_trivariate.py        三变量对比
  run_e7_baselines.py      E7 vs LR vs RNN
  run_optimize_e7.py       最终优化版 (50-trial + 5-seed 集成)

data_*.py              ← ⭐ data pipeline
  data_parse_ao.py         AO 文本 → CSV
  data_extract_sst.py      SST NetCDF → CSV
  data_build_lagged.py     滞后特征 + 掩码

tune_*.py              ← ⭐ hyperparameter tuning
  tune_univariate.py       单变量调优
  tune_bivariate.py        双变量调优
  tune_trivariate.py       三变量调优

util_*.py              ← ⭐ utilities
  util_check.py            数据路径诊断
  util_monthly.py          逐月误差分解
  util_export.py           逐样本预测导出

summary.py             ← ⭐ final report (all experiments → tables + charts)

paper_figures/         ← ⭐ 论文图表生成 (11 scripts → outputs/plots/paper/)
  gen_fig_lstm_seasonal.py      图3.2 单变量季节均值 ±1σ
  gen_fig_lstm_short_seasonal.py 图3.3 短期预测季节均值
  gen_fig34.py                  图3.4 单变量最好/最坏年
  gen_fig37.py                  图3.7 三种模型逐月RMSE
  gen_fig42.py                  图4.2 双编码器E7v1季节均值
  gen_fig_e4_seasonal.py        图4.3 三变量E4季节均值
  gen_fig_e4_e7_rmse.py         图4.4 E4 vs E7v1 逐月RMSE
  gen_fig_e7_bestworst.py       图4.5 E7最好/最坏年
  gen_fig_lstm_loss.py          图3.1 单变量损失曲线
  gen_loss_curves.py            图4.1 E4 vs E7 损失曲线
  gen_fig_loss_schemes.py       图3.5 短期+中期损失曲线
```

## Python Environment

This project requires the conda PyTorch environment:

```bash
E:/anaconda/envs/pytorch/python.exe    # PyTorch 2.10.0+cu128
```

Set `PYTHONIOENCODING=utf-8` when running scripts that output Unicode (e.g., `²`):

```bash
PYTHONIOENCODING=utf-8 E:/anaconda/envs/pytorch/python.exe gen_fig42.py
```

## Quick Reference (run commands)

```bash
# Install
pip install -r requirements.txt

# Core
python main.py                      # basic LSTM train + eval
python compare_models.py            # LR vs RNN vs LSTM baseline

# Primary experiments
python run_trivariate.py            # uni vs bi vs tri LSTM
python run_e7_baselines.py          # E7 dual-encoder vs LR vs RNN
python run_optimize_e7.py           # final: 50-trial Optuna + 5-seed ensemble

# Data prep (run once)
python data_parse_ao.py             # AO → CSV
python data_extract_sst.py          # SST → CSV
python data_build_lagged.py         # lagged features + mask

# Tuning
python tune_univariate.py           # univariate LSTM
python tune_bivariate.py            # bivariate LSTM
python tune_trivariate.py           # trivariate LSTM

# Reporting
python summary.py                   # compile all results → tables + charts
python util_check.py                # diagnose data paths

# Paper figures (→ outputs/plots/paper/)
python paper_figures/gen_fig_lstm_seasonal.py     # univariate seasonal mean ±1σ
python paper_figures/gen_fig_lstm_short_seasonal.py  # short-term LSTM seasonal mean
python paper_figures/gen_fig34.py                 # univariate best/worst year (2020/2016)
python paper_figures/gen_fig37.py                 # three-model calendar-month RMSE (LR/RNN/LSTM)
python paper_figures/gen_fig42.py                 # dual-encoder E7v1 seasonal mean ±1σ
python paper_figures/gen_fig_e4_seasonal.py       # trivariate E4 seasonal mean ±1σ
python paper_figures/gen_fig_e4_e7_rmse.py        # E4 vs E7v1 calendar-month RMSE
python paper_figures/gen_fig_e7_bestworst.py      # E7 best/worst year (2022/2016)
python paper_figures/gen_loss_curves.py           # E4 vs E7 loss curves with zoom insets
python paper_figures/gen_fig_loss_schemes.py      # short + medium prediction loss curves
```

## Critical: Hardcoded Paths

[config.py](config.py) uses **hardcoded absolute Windows paths** at lines 5-6. These must be updated before running on any other machine. [util_check.py](util_check.py) also has its own path — update it too.

## Prediction Scheme Config

The `PREDICTION_SCHEME` in [config.py](config.py) (`"short"`, `"medium"`, or `"long"`) selects a preset from `_SCHEME_PARAMS`. The selected dict's keys are injected as module globals (line 74-76): changing `PREDICTION_SCHEME` atomically switches `OUTPUT_LEN`, `HIDDEN_SIZE`, `NUM_LAYERS`, `DROPOUT`, `BATCH_SIZE`, `LEARNING_RATE`, and `WEIGHT_DECAY`. Don't set these manually — change `PREDICTION_SCHEME` instead.

## Data Files

**Required for any run:**
- `data/raw/N_01_extent_v4.0.csv` … `N_12_extent_v4.0.csv` — monthly sea ice (year, month, extent, area)

**For multivariate/dual-encoder runs:**
- `data/ao_monthly.csv` — year, month, ao (1950-2025)
- `data/arctic_sst_monthly.csv` — year, month, sst (1981-2023)
- `data/lagged_features.csv` — year, month, area, ao, sst, lag1, lag2, sst_mask

**Source files:**
- `data/NOAA .CPC. AO.monthly.txt` — raw AO
- `data/sst.mnmean.nc` — global SST NetCDF

## Key Architecture

### Data Pipeline ([src/data_preprocessing.py](src/data_preprocessing.py))

| Function | Purpose | Used by |
|----------|---------|---------|
| `load_and_merge_data()` | Load + merge 12 ice CSVs, normalize | main, compare_models, optuna_tuning |
| `load_multivariate_data()` | Load ice + AO, normalize independently | compare_univariate_bivariate, compare_trivariate |
| `load_trivariate_data()` | Load ice + AO + SST, normalize independently | optuna_tuning_trivariate |
| `load_dual_encoder_data()` | Load lagged features (7 aux channels) | compare_dual_encoder, optimize_e7 |
| `create_sequences()` | Sliding window: 12 in → N out | All scripts |
| `train_val_test_split_by_year()` | Year-based: train 1979-2010, val 2011-2015, test 2016-2025 | Main scripts |
| `train_val_test_split()` | Ratio split 70/15/15 (faster, for tuning) | Optuna scripts |

### Models ([src/model.py](src/model.py))

| Model | Input | Params | Notes |
|-------|-------|--------|-------|
| `LinearRegressionModel` | (12,) | 156 | Strong baseline — ice seasonality is linear |
| `SimpleRNNModel` | (12, 1) | ~5K | Validation comparison |
| `SeaIceLSTM` | (12, n) | 18K-270K | Main model; simplified mode (NUM_LAYERS=1) recommended |
| `SeaIceDualEncoderLSTM` | main:(12,1) + aux:(3,7) | 74K-274K | **Best performer** — separate encoders for ice + lagged AO/SST |

The `SeaIceDualEncoderLSTM` architecture:
- **Main encoder:** LSTM(12-mo sea ice) → main_hidden
- **Aux encoder:** LSTM(last 3-6 mo of 7-channel lagged AO/SST/mask) → aux_hidden
- **Aux dropout=0.6** suppresses noise; main dropout=0.1
- Concat → Linear → output

### Training ([src/train.py](src/train.py))
- Adam/AdamW with weight_decay, ReduceLROnPlateau, gradient clipping, early stopping
- `train_model()` returns `(train_losses, val_losses, best_model_path, best_epoch, train_time_s)`

## Key Constraints

- ~500 training samples — favors simpler models; NUM_LAYERS=1 recommended
- Year-based split prevents leakage (can't train on 2020 to predict 2015)
- Optuna tuning uses 50-epoch trials (not full 1000) for speed
- SST data starts 1981 — use left-join + mask, NOT inner join (loses 2 years)
- Independent normalization per variable (don't mix scales)

## Experiment Results (quick reference)

| Rank | Model | RMSE | Key insight |
|:----:|-------|:----:|-------------|
| 1 | Dual-Encoder (E7opt, ens5) | 0.5014 | Best multivariate; 5-seed ensemble |
| 2 | Dual-Encoder (E7v1) | 0.4974 | Best single-seed multivariate |
| 3 | Univariate LSTM (E1) | 0.4975 | Bare ice baseline |
| 4 | SimpleRNN | 0.5255 | Basic RNN |
| 5 | LinearRegression | 0.5317 | **156 params!** |
| 6 | Bivariate tuned (E3) | 0.5397 | AO added; worse than univariate |
| 7 | Trivariate (E4) | 0.6151 | AO+SST inner join; worst |

Full results: `outputs/summary/complete_report.txt`

## Pre-trained Model Checkpoints

All in `outputs/models/`. Architecture must match **exactly** when loading:

| File | Model Class | Key Shape Params |
|------|------------|-----------------|
| `best_model_area.pth` | `SeaIceLSTM` | input_size=1, hidden_size=256, num_layers=1, output_len=12, dropout=0.1 |
| `best_model_de.pth` | `SeaIceDualEncoderLSTM` | input_size=1, main_hidden=128, aux_hidden=32, aux_input_size=7, aux_seq_len=3, num_layers=1, output_len=12, dropout=0.1, aux_dropout=0.6 |
| `paper_e4.pth` | `SeaIceLSTM` | input_size=3, hidden_size=64, num_layers=1, output_len=12, dropout=0.1 |

To inspect an unknown checkpoint: read `state_dict` key shapes — `main_lstm.weight_ih_l0` shape[0]÷4 = main_hidden, `aux_lstm.weight_ih_l0` shape[0]÷4 = aux_hidden.

## Paper Plotting Conventions (CRITICAL)

All paper figures (`gen_*.py` → `outputs/plots/paper/`) follow these rules. **Do not deviate.**

### Setup (required in every gen_*.py)
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
```

### Style rules
- **Labels:** Chinese throughout, format `"中文名称 (单位)"` — e.g. `'海冰面积 (百万平方公里)'`, `'均方根误差 (百万平方公里)'`
- **No titles** on figures (学术论文规范)
- **Subfigure tags** (a), (b): upper-left of each panel, `fontsize=16`, `fontweight='bold'`, via `ax.text(0.03, 0.97, ...)`
- **Legends:** `loc='lower left'`, `framealpha=0.9`
- **DPI:** 200 for all saves; format: PNG
- **Unicode:** use `平方公里` not `km²` (avoids charmap encoding errors)

### Color scheme (per model)
| Model | Color | Hex | Line Style |
|-------|-------|-----|-----------|
| Univariate LSTM | Blue | `#2166ac` | `s--` (square dash) |
| Univariate LSTM (alt) | Red | `#d73027` | `o-` |
| E7v1 Dual-Encoder | Green | `#1b7837` | `s--` |
| E4 Trivariate | Blue | `#2166ac` | `s--` |
| LinearRegression | Blue | `#3498db` | (varies) |
| SimpleRNN | Green | `#2ecc71` | (varies) |
| Observed/True values | Red | `#e74c3c` or `#d73027` | `o-` |

### Calendar-month grouping (MANDATORY for multi-year mean/RMSE)
**Never** use `y_pred.mean(axis=0)` — this mixes predictions from different calendar months (e.g. January forecast-step-1 with July forecast-step-1), flattening the seasonal cycle. Instead, map each prediction to its actual target calendar month:

```python
# Build target-month array from df_months (NOT from step index)
all_target_months = np.array([df_months[12 + i : 12 + i + ol] for i in range(n_total)])
test_target_months = all_target_months[test_mask]

# Group values by calendar month
pred_by_month = {m: [] for m in range(1, 13)}
for i in range(len(y_pred)):
    for k in range(ol):
        cal_month = int(test_target_months[i, k])
        pred_by_month[cal_month].append(y_pred[i, k])
```

### Calendar-year aggregation for best/worst year
For per-year plots, assemble all 12 calendar months across multiple prediction samples:

```python
def get_year_data(yr):
    pred_vals, true_vals = {}, {}
    for i in range(len(y_pred_all)):
        for k in range(ol):
            if int(test_yrs[i, k]) == yr:
                m = int(test_mos[i, k])
                pred_vals[m] = y_pred_all[i, k]
                true_vals[m] = y_true_all[i, k]
    if len(pred_vals) == 12:
        ms = sorted(pred_vals.keys())
        return np.array([pred_vals[m] for m in ms]), np.array([true_vals[m] for m in ms])
    return None, None
```

### Metrics display
- RMSE/MAE/MAPE **not shown on figures** (printed to console only)
- If needed in text, reference from console output

## Known Issues

- [main.py](main.py) has duplicate `set_all_seeds()` (lines 29-38 and 44-53); second shadows the first
- [util_check.py](util_check.py) has its own hardcoded path (separate from config.py)
- [src/data_preprocessing.py](src/data_preprocessing.py) `__main__` block has stale hardcoded path
- [src/hybrid_model.py](src/hybrid_model.py) is an incomplete work-in-progress (`TinyLSTM` class only, file truncated mid-line) — not used by any script
