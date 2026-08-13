# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Arctic Sea Ice Area (SIA) prediction using LSTM neural networks with multi-source climate indices. Forecasts monthly sea ice area for the next 12 months from historical NSIDC data (1979-present). Uses a **dual-encoder LSTM architecture** — main encoder processes 12-month sea ice history, auxiliary encoder processes lagged climate indices (AO, NAO, PNA, SST, Nino3.4).

The project spans **38 experiments across 4 phases**:
1. **Phase 1** — Single-variable increment (E1, E7v1, E8-E14): measure each climate index's individual contribution
2. **Phase 2** — Variable combinations (E15-E19, E23-E24): test cross-sector complementarity vs within-sector redundancy
3. **Phase 3** — Ablation verification (E20-E22, E25): remove each variable from full set to identify necessity
4. **Phase 4** — Regional spatial heterogeneity (E26-E32): matched vs mismatched climate indices for 7 Arctic seas

Results feed into a Chinese-language thesis (~40 figures, compiled via `write/` → .docx).

## Script Hierarchy

```
config.py              ← configuration hub + PREDICTION_SCHEME presets
nature_figure_config.py ← global matplotlib 600-DPI Nature-style settings

=== Core Pipeline ===
main.py                ← basic LSTM train + eval (long scheme)
compare_models.py      ← LR vs RNN vs LSTM baseline

=== Phase 1-4 Experiments (⭐ primary results) ===
run_phase1_single_variable.py  ← E1, E7v1, E8-E14: single-variable increment
run_phase2_combinations.py     ← E15-E19, E23-E24: variable combinations
run_phase3_ablation.py         ← E20-E22, E25: ablation verification
run_phase4_regional.py         ← E26-E32: 7-region spatial heterogeneity
tune_phase4_regional.py        ← independent hyperparameter tuning per region
run_trivariate.py              ← uni vs bi vs tri LSTM (legacy)
run_e7_baselines.py            ← E7 vs LR vs RNN (legacy)
run_optimize_e7.py             ← 50-trial Optuna + 5-seed ensemble (legacy)

=== Data Pipeline ===
data_parse_ao.py               ← AO text → CSV
data_parse_cpc.py              ← NAO, PNA, Nino3.4, PDO text → CSV (multi-index)
data_extract_sst.py            ← SST NetCDF → CSV
data_extract_t2m.py            ← T2M extraction
data_extract_slp.py            ← SLP extraction
data_build_lagged.py           ← v1: AO + SST lags + mask (7 aux channels)
data_build_lagged_v2.py        ← v2: all 5 climate indices + extended lags (~22 channels)

=== Hyperparameter Tuning ===
tune_univariate.py             ← univariate LSTM
tune_bivariate.py              ← bivariate LSTM
tune_trivariate.py             ← trivariate LSTM

=== Utilities ===
util_check.py                  ← diagnose data paths
util_monthly.py                ← monthly error decomposition (6m/12m)
util_export.py                 ← per-sample prediction export

=== Reporting ===
summary.py                     ← legacy: all experiments → tables + charts
summary_v2.py                  ← updated summary (Phases 1-4)

=== Thesis Figures (35 scripts → outputs/plots/paper/) ===
paper_figures/thesis_data.py   ← ⭐ centralized hardcoded experiment results
paper_figures/nature_figure_config.py (symlink/import from root)

  Chapter 2 (Data & Methods):
    gen_fig_arctic_map.py          图2.1 Arctic regional map (cartopy)
    gen_fig_ch2_data_overview.py   图2.2 SIE trend + seasonal cycle + climate indices
    gen_acf.py                     图2.3 SIE autocorrelation function

  Chapter 3 (Model Validation & Phases 1-4):
    gen_fig_ch3_model_comparison.py    图3.1 LR/RNN/LSTM monthly RMSE
    gen_fig_ch3_seasonal_prediction.py 图3.2 LSTM seasonal mean +/- 1sigma
    gen_fig_ch3_best_worst_year.py     图3.3 Best/worst prediction year
    gen_fig_single_variable.py         图3.4 Phase 1 single-variable RMSE bars
    gen_fig_combinations.py            图3.5 Phase 2 combination ranking
    gen_fig_ablation.py                图3.6 Phase 3 ablation delta-RMSE
    gen_fig_diagnostics.py             图3.7 Taylor + bootstrap forest + correlation
    gen_fig38_regional_baseline.py     图3.8 Regional baseline RMSE (7 seas)
    gen_fig39_matching_heatmap.py      图3.9 Sector-matching heatmap
    gen_fig310_tuning_comparison.py    图3.10 Tuning before/after paired bars
    gen_fig311_hyperparams_shift.py    图3.11 Hyperparameter shift scatter

  Chapter 4 (Multivariate Extension, legacy figures):
    gen_fig_lstm_loss.py          图4.1 Univariate LSTM loss curve
    gen_fig_lstm_seasonal.py      图4.2 Univariate seasonal mean
    gen_fig_lstm_short_seasonal.py图4.3 Short-term LSTM seasonal mean
    gen_fig34.py                  图4.4 Univariate best/worst year
    gen_fig_loss_schemes.py       图4.5 Short + medium loss curves
    gen_fig37.py                  图4.6 Three-model monthly RMSE
    gen_fig42.py                  图4.7 E7v1 seasonal mean
    gen_fig_e4_seasonal.py        图4.8 E4 trivariate seasonal mean
    gen_fig_e4_e7_rmse.py         图4.9 E4 vs E7v1 monthly RMSE
    gen_fig_e7_bestworst.py       图4.10 E7 best/worst year
    gen_loss_curves.py            图4.11 E4 vs E7 loss curves

  Supplementary / Analysis:
    gen_fig_variable_impact.py        Phase 1 variable impact bar chart
    gen_fig_combination_heatmap.py    Variable combination RMSE heatmap
    gen_ablation_e7.py                E7 ablation: 5 variants x 5 seeds
    gen_fig_bestworst_multi.py        Best/worst multi-model comparison
    gen_fig_conditional_skill.py      Conditional skill (monthly, lead-time, bootstrap)
    gen_residuals.py                  QQ plot + residual histogram + monthly bias
    gen_scheme_comparison.py          Short/medium/long scheme RMSE comparison
    gen_uncertainty.py                E7 5-seed ensemble prediction intervals
    gen_data_overview.py              Legacy data overview (3-panel)
    gen_fig_ablation.py               Legacy ablation (E7-focused)

=== Thesis Compilation (write/) ===
write/paper.md                  ← Main thesis manuscript (372 lines, 4 chapters)
write/chapters/_write_ch3.py    ← Chapter 3 compilation script
write/chapters/_write_ch4.py    ← Chapter 4 compilation script
write/chapters/build_final.py   ← Markdown → .docx (python-docx)
```

## Python Environment

Conda PyTorch environment:

```bash
E:/anaconda/envs/pytorch/python.exe    # PyTorch 2.10.0+cu128
```

Always set `PYTHONIOENCODING=utf-8` for Unicode output:

```bash
PYTHONIOENCODING=utf-8 E:/anaconda/envs/pytorch/python.exe script.py
```

Key dependencies: `numpy`, `pandas`, `matplotlib`, `cartopy`, `xarray`, `netCDF4`, `optuna`, `scipy`, `scikit-learn`, `svg.path`, `pyproj`, `python-docx`.

## Quick Reference (run commands)

All commands run from project root. Use `PYTHONIOENCODING=utf-8` prefix for scripts with Chinese output.

```bash
# ===== Data prep (run once, in order) =====
python data_parse_ao.py              # AO text → data/ao_monthly.csv
python data_parse_cpc.py             # NAO/PNA/Nino3.4/PDO → CSVs
python data_extract_sst.py           # SST NetCDF → data/arctic_sst_monthly.csv
python data_build_lagged_v2.py       # unified lagged features (~22 aux channels)

# ===== Phase 1-4 Experiments (main results) =====
python run_phase1_single_variable.py           # E1, E7v1, E8-E14: 5-seed ensemble
python run_phase1_single_variable.py --exp E1  # single experiment
python run_phase1_single_variable.py --fast    # 1-seed fast mode for dev
python run_phase1_single_variable.py --list    # list all experiments
python run_phase2_combinations.py              # E15-E19, E23-E24: combinations
python run_phase3_ablation.py                  # E20-E22, E25: ablation
python run_phase4_regional.py                  # E26-E32: 7 regional seas

# ===== Legacy experiments =====
python run_trivariate.py             # uni vs bi vs tri LSTM
python run_e7_baselines.py           # E7 dual-encoder vs LR vs RNN
python run_optimize_e7.py            # 50-trial Optuna + 5-seed ensemble

# ===== Tuning =====
python tune_univariate.py            # univariate LSTM
python tune_bivariate.py             # bivariate LSTM
python tune_trivariate.py            # trivariate LSTM
python tune_phase4_regional.py       # per-region independent tuning

# ===== Reporting =====
python summary.py                    # legacy: all experiments → tables + charts
python summary_v2.py                 # Phases 1-4 summary
python util_check.py                 # diagnose data paths

# ===== Thesis figures (→ outputs/plots/paper/ as PNG+SVG at 600 DPI) =====
# Chapter 2
python paper_figures/gen_fig_arctic_map.py
python paper_figures/gen_fig_ch2_data_overview.py
python paper_figures/gen_acf.py

# Chapter 3 (Phases 1-4)
python paper_figures/gen_fig_ch3_model_comparison.py
python paper_figures/gen_fig_ch3_seasonal_prediction.py
python paper_figures/gen_fig_ch3_best_worst_year.py
python paper_figures/gen_fig_single_variable.py
python paper_figures/gen_fig_combinations.py
python paper_figures/gen_fig_ablation.py
python paper_figures/gen_fig_diagnostics.py
python paper_figures/gen_fig38_regional_baseline.py
python paper_figures/gen_fig39_matching_heatmap.py
python paper_figures/gen_fig310_tuning_comparison.py
python paper_figures/gen_fig311_hyperparams_shift.py

# Chapter 4 (legacy figures)
python paper_figures/gen_fig_lstm_loss.py
python paper_figures/gen_fig_lstm_seasonal.py
python paper_figures/gen_fig_lstm_short_seasonal.py
python paper_figures/gen_fig34.py
python paper_figures/gen_fig37.py
python paper_figures/gen_fig42.py
python paper_figures/gen_fig_e4_seasonal.py
python paper_figures/gen_fig_e4_e7_rmse.py
python paper_figures/gen_fig_e7_bestworst.py
python paper_figures/gen_loss_curves.py
python paper_figures/gen_fig_loss_schemes.py

# Supplementary
python paper_figures/gen_fig_variable_impact.py
python paper_figures/gen_fig_combination_heatmap.py
python paper_figures/gen_ablation_e7.py
python paper_figures/gen_fig_bestworst_multi.py
python paper_figures/gen_fig_conditional_skill.py
python paper_figures/gen_residuals.py
python paper_figures/gen_scheme_comparison.py
python paper_figures/gen_uncertainty.py
```

## Critical: Hardcoded Paths

[config.py](config.py) uses **hardcoded absolute Windows paths** at lines 5-6. These must be updated before running on any other machine. [util_check.py](util_check.py) also has its own path — update it too.

## Prediction Scheme Config

The `PREDICTION_SCHEME` in [config.py](config.py) (`"short"`, `"medium"`, or `"long"`) selects a preset from `_SCHEME_PARAMS`. The selected dict's keys are injected as module globals (line 74-76): changing `PREDICTION_SCHEME` atomically switches `OUTPUT_LEN`, `HIDDEN_SIZE`, `NUM_LAYERS`, `DROPOUT`, `BATCH_SIZE`, `LEARNING_RATE`, and `WEIGHT_DECAY`. Don't set these manually — change `PREDICTION_SCHEME` instead.

## Data Files

### Primary ice data (required for all runs)
- `data/raw/N_01_extent_v4.0.csv` … `N_12_extent_v4.0.csv` — NSIDC Sea Ice Index v4.0 (year, month, extent, area)

### Climate indices (for multivariate/dual-encoder)
| CSV File | Columns | Source Script | Period |
|----------|---------|---------------|--------|
| `data/ao_monthly.csv` | year, month, ao | `data_parse_ao.py` | 1950-2025 |
| `data/nao_monthly.csv` | year, month, nao | `data_parse_cpc.py` | 1950-2025 |
| `data/pna_monthly.csv` | year, month, pna | `data_parse_cpc.py` | 1950-2025 |
| `data/nino34_monthly.csv` | year, month, nino34 | `data_parse_cpc.py` | 1950-2025 |
| `data/pdo_monthly.csv` | year, month, pdo | `data_parse_cpc.py` | 1950-2025 |
| `data/arctic_sst_monthly.csv` | year, month, sst | `data_extract_sst.py` | 1981-2023 |

### Processed feature stores
- `data/lagged_features.csv` — **v1**: 7 aux channels (ao, sst, ao_lag1-2, sst_lag1-2, sst_mask), 564 rows
- `data/lagged_features_v2.csv` — **v2**: ~22 aux channels (all 5 indices + extended lags), used by Phases 1-4

### Regional data (Phase 4)
Seven CSV files — `data/{region}_monthly.csv` for: `bering`, `chukchi`, `barents`, `kara`, `laptev`, `greenland`, `central_arctic`. Extracted from NSIDC Regional Monthly Data Excel workbook.

### Source files
- `data/NOAA .CPC. AO.monthly.txt` — raw NOAA CPC AO
- `data/NOAA_CPC_NAO_monthly.txt` — raw NAO
- `data/NOAA_CPC_PNA_monthly.txt` — raw PNA
- `data/NOAA_CPC_nino34_monthly.txt` — raw Nino3.4
- `data/sst.mnmean.nc` — global NOAA OI SST V2 NetCDF
- `data/nsidc_arctic_region_mask_12500.nc` — 12.5 km EASE-Grid 2.0 regional mask (864x864), used for map figure
- `data/N_Sea_Ice_Index_Regional_Monthly_Data_G02135_v4.0.xlsx` — NSIDC regional monthly data

## Key Architecture

### Data Pipeline ([src/data_preprocessing.py](src/data_preprocessing.py))

| Function | Purpose | Used by |
|----------|---------|---------|
| `load_and_merge_data()` | Load + merge 12 ice CSVs, normalize | main, compare_models, optuna_tuning |
| `load_multivariate_data()` | Load ice + AO, normalize independently | compare_univariate_bivariate, compare_trivariate |
| `load_trivariate_data()` | Load ice + AO + SST, normalize independently | optuna_tuning_trivariate |
| `load_dual_encoder_data()` | Load v1 lagged features (7 aux channels) | compare_dual_encoder, optimize_e7 |
| `load_dual_encoder_data_v2()` | Load v2 lagged features (N aux channels, flexible) | run_phase1/2/3/4 |
| `create_sequences()` | Sliding window: 12 in → N out | All scripts |
| `train_val_test_split_by_year()` | Year-based: train 1979-2010, val 2011-2015, test 2016-2025 | Main scripts |
| `train_val_test_split()` | Ratio split 70/15/15 (faster, for tuning) | Optuna scripts |

The v2 data loader supports a variable number of auxiliary channels via `aux_vars` parameter, enabling the Phase 1-4 experiments with different climate index subsets.

### Dataset Classes ([src/data_preprocessing.py](src/data_preprocessing.py))

| Class | Purpose |
|-------|---------|
| `SeaIceDataset` | Standard sliding-window dataset for single-encoder LSTM |
| `DualEncoderDataset` | Two-input dataset: main sequence (12, 1) + aux sequence (aux_seq_len, n_aux_vars) |

### Models ([src/model.py](src/model.py))

| Model | Input | Params | Notes |
|-------|-------|--------|-------|
| `LinearRegressionModel` | (12,) | 156 | Strong baseline — ice seasonality is linear |
| `SimpleRNNModel` | (12, 1) | ~5K | Validation comparison |
| `SeaIceLSTM` | (12, n) | 18K-270K | Main single-encoder model; NUM_LAYERS=1 recommended |
| `SeaIceDualEncoderLSTM` | main:(12,1) + aux:(3-6, N) | 74K-274K | **Best performer** — two separate LSTM encoders |

The `SeaIceDualEncoderLSTM` architecture (E7v1 baseline: main_hidden=256, aux_hidden=64, aux_seq_len=6):
- **Main encoder:** LSTM(12-month sea ice) → main_hidden
- **Aux encoder:** LSTM(aux_seq_len months of N-channel lagged climate indices) → aux_hidden
- **Aux dropout=0.6** suppresses noise; main dropout=0.1
- Concat → Linear → 12-month output

### Training ([src/train.py](src/train.py))
- Adam/AdamW with weight_decay, ReduceLROnPlateau, gradient clipping, early stopping
- `train_model()` — single-encoder: returns `(train_losses, val_losses, best_model_path, best_epoch, train_time_s)`
- `train_dual_encoder()` — dual-encoder version with two DataLoaders
- `predict_dual_encoder()` — dual-encoder prediction with auxiliary input handling

### Unified Hyperparameter Strategy (E7V1_HP)

All Phases 1-4 use identical hyperparameters for apples-to-apples comparison (from Optuna-optimized E7v1):

```python
E7V1_HP = dict(
    main_hidden=256, aux_hidden=64, aux_seq_len=6,
    num_layers=1, dropout=0.1, aux_dropout=0.6,
    batch_size=32, lr=0.000635, wd=2.59e-06
)
```

Phase 4 regional experiments start with E7V1_HP, then independently tune per region via `tune_phase4_regional.py`.

## Key Constraints

- ~500 training samples — favors simpler models; NUM_LAYERS=1 recommended
- Year-based split prevents leakage (can't train on 2020 to predict 2015)
- Optuna tuning uses 50-epoch trials (not full 1000) for speed
- SST data starts 1981 — use left-join + mask, NOT inner join (loses 2 years)
- Independent normalization per variable (don't mix scales)

## Experiment Registry (E1-E32)

### Key results summary

| Rank | Experiment | Description | RMSE | delta vs E1 |
|:----:|-----------|-------------|:----:|:-----------:|
| 1 | E10 (+PNA) | Single-variable: best individual index | 0.5089 | **-0.0147 (-2.81%)** |
| 2 | E14 (+AO) | Single-variable: AO alone | 0.5130 | -0.0106 |
| 3 | E9 (+Nino3.4) | Single-variable: tropical SST | 0.5180 | -0.0055 |
| 4 | E8 (+NAO) | Single-variable: NAO alone | 0.5192 | -0.0044 |
| 5 | E1 (pure ice) | Univariate LSTM baseline | 0.5236 | 0.0000 |
| 6 | E7v1 (+AO+SST) | Dual-encoder with AO+SST | 0.5420 | +0.0184 (degrades!) |
| 7 | E17 (SST+Nino3.4) | Best 2-var combo (ocean-internal) | 0.5098 | — |
| 8 | E19 (ALL5) | All 5 climate indices | 0.5243 | — |
| 9 | E24 (AO+NAO+PNA) | Worst combo (within-atmosphere redundancy) | 0.5383 | — |

**Key finding:** PNA is the single best climate index predictor for Arctic SIA. AO is systematically harmful when combined with SST (E7v1). "Less is more" — small, cross-sector combinations outperform full-variable models. Phase 4 shows regional tuning releases significant gains (up to -9.9% for Kara Sea).

Full experiment summaries in `outputs/results/phase{1,2,3,4}/` and `paper_figures/thesis_data.py`.

### Legacy baseline results

| Rank | Model | RMSE | Key insight |
|:----:|-------|:----:|-------------|
| 1 | Dual-Encoder (E7opt, ens5) | 0.5014 | Best multivariate; 5-seed ensemble |
| 2 | Univariate LSTM (E1) | 0.4975 | Bare ice baseline |
| 3 | Dual-Encoder (E7v1) | 0.4974 | Best single-seed (note: with old data) |
| 4 | SimpleRNN | 0.5255 | Basic RNN |
| 5 | LinearRegression | 0.5317 | **156 params!** |
| 6 | Bivariate tuned (E3) | 0.5397 | AO added; worse than univariate |
| 7 | Trivariate (E4) | 0.6151 | AO+SST inner join; worst |

## Pre-trained Model Checkpoints

All in `outputs/models/` (30+ .pth files). Architecture must match **exactly** when loading.

### Key checkpoints

| File | Model Class | Key Shape Params |
|------|------------|-----------------|
| `best_model_area.pth` | `SeaIceLSTM` | input_size=1, hidden_size=256, num_layers=1, output_len=12, dropout=0.1 |
| `best_model_de.pth` | `SeaIceDualEncoderLSTM` | input_size=1, main_hidden=128, aux_hidden=32, aux_input_size=7, aux_seq_len=3, num_layers=1, output_len=12, dropout=0.1, aux_dropout=0.6 |
| `best_model_de_s42.pth` … `s46.pth` | `SeaIceDualEncoderLSTM` | 5-seed ensemble variants of best_model_de |
| `paper_e4.pth` | `SeaIceLSTM` | input_size=3, hidden_size=64, num_layers=1, output_len=12, dropout=0.1 |
| `paper_lstm_short.pth` | `SeaIceLSTM` | Short-term (12→1) prediction |
| `paper_e7v1_final.pth` | `SeaIceDualEncoderLSTM` | E7v1 final optimized |

### Inspecting an unknown checkpoint

Read `state_dict` key shapes from the .pth file:
- `main_lstm.weight_ih_l0.shape[0] / 4` = main_hidden
- `aux_lstm.weight_ih_l0.shape[0] / 4` = aux_hidden (if dual-encoder)
- Look for `aux_lstm` key prefix to distinguish `SeaIceDualEncoderLSTM` from `SeaIceLSTM`

## Paper Plotting Conventions (CRITICAL)

All paper figures output to `outputs/plots/paper/`. **Do not deviate from these rules.**

### Setup (required in every gen_*.py)

Two options — importing `nature_figure_config` handles everything automatically:

```python
# Preferred: use shared config (new thesis figures)
from nature_figure_config import *
# This sets: 600 DPI, Arial+SimHei fonts, clean spines, no titles,
#            SVG+PNG output, monkey-patched savefig

# Legacy / standalone (old figures)
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
```

**IMPORTANT:** `nature_figure_config.py` monkey-patches `plt.savefig` and `Figure.savefig` to force **600 DPI** and output both **PNG + SVG** formats. The old AGENTS.md claim of "DPI: 200, format: PNG" is **stale** — all current figures use 600 DPI SVG+PNG.

### Style rules
- **Labels:** Chinese throughout, format `'中文名称 (单位)'` — e.g. `'海冰面积 (百万平方公里)'`, `'均方根误差 (百万平方公里)'`
- **No titles** on figures (学术论文规范)
- **Subfigure tags** (a), (b): upper-left of each panel, `fontsize=16`, `fontweight='bold'`, via `ax.text(0.03, 0.97, ...)`
- **Legends:** `loc='lower left'`, `framealpha=0.9` (legacy); `frameon=False` (new thesis style via nature_figure_config)
- **DPI:** 600 for all saves; format: **PNG + SVG** (both saved)
- **Fonts:** Arial (primary) + SimHei (Chinese fallback) + DejaVu Sans (symbol fallback)
- **Spines:** Right and top spines disabled; axis linewidth 0.8; tick direction 'in'
- **Unicode:** use `平方公里` not `km²` (avoids charmap encoding errors)

### Thesis color scheme (new, from nature_figure_config / thesis_data)

| Element | Hex | Usage |
|---------|-----|-------|
| Blue-purple | `#7884B4` | Primary model / LSTM |
| Deep blue | `#484878` | Dark variant / ensemble mean |
| Pink | `#F0C0CC` | Secondary / comparison model |
| Green | `#2E9E44` | Improvement / positive delta |
| Red | `#E53935` | Degradation / negative delta / observation |
| Gray | `#D8D8D8` | Baseline / grid lines |

### Legacy color scheme (old figures, still used in some gen_*.py)

| Model | Color | Hex | Line Style |
|-------|-------|-----|-----------|
| Univariate LSTM | Blue | `#2166ac` | `s--` (square dash) |
| E7v1 Dual-Encoder | Green | `#1b7837` | `s--` |
| E4 Trivariate | Blue | `#2166ac` | `s--` |
| LinearRegression | Blue | `#3498db` | (varies) |
| SimpleRNN | Green | `#2ecc71` | (varies) |
| Observed/True values | Red | `#e74c3c` or `#d73027` | `o-` |
| Univariate LSTM (alt) | Red | `#d73027` | `o-` |

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

## Centralized Data: thesis_data.py

`paper_figures/thesis_data.py` is the **single source of truth** for all experiment results used in figures. It contains hardcoded Python dicts (manually copied from JSON result files), NOT dynamic file loading. This means it can go stale if experiments are re-run.

Key data structures:
- `PHASE1_SINGLE_VAR` — E1, E7v1, E8-E14 single-variable RMSE + ensemble stats
- `PHASE2_COMBINATIONS` — E15-E19, E23-E24 sorted by RMSE
- `PHASE3_ABLATION` — E20-E22, E25 delta-RMSE from E19 ALL5 baseline
- `PHASE4_UNTUNED` / `PHASE4_TUNED` — per-region RMSE with unified vs tuned params
- `REGIONS` — 7 region metadata with sector classification and mean_area
- `LEGACY_BASELINES` — LR, RNN, E1_old RMSE values
- `PAN_ARCTIC_RANKING` — combined Phase 1+2 ranking

If you re-run any Phase 1-4 experiment, update the corresponding dict in `thesis_data.py`.

## Figure Dependencies

Most `gen_fig*.py` scripts depend on pre-computed data:

| Dependency | Used by |
|-----------|---------|
| `thesis_data.py` (hardcoded) | All ch3/ch4 figure scripts |
| `outputs/results/phase1/*.json` | `gen_fig_single_variable.py`, `gen_fig_variable_impact.py`, `gen_fig_combination_heatmap.py` |
| `outputs/results/phase2/*.json` | `gen_fig_combinations.py`, `gen_fig_combination_heatmap.py` |
| `outputs/results/phase3/*.json` | `gen_fig_ablation.py` |
| `outputs/results/phase4/*.json` | `gen_fig38_regional_baseline.py`, `gen_fig39_matching_heatmap.py`, `gen_fig310_tuning_comparison.py` |
| `outputs/summary/03_monthly_rmse.csv` | `gen_fig_ch3_model_comparison.py` |
| `outputs/models/best_model_de.pth` | `gen_fig42.py`, `gen_fig_e7_bestworst.py`, `gen_residuals.py` |
| `outputs/models/best_model_area.pth` | `gen_fig34.py`, `gen_fig_lstm_seasonal.py` |
| Conditional skill JSONs | `gen_fig_bestworst_multi.py`, `gen_fig_conditional_skill.py` |
| Pre-computed loss CSVs | `gen_loss_curves.py`, `gen_fig_loss_schemes.py` |
| Raw data (ice CSVs, climate CSVs) | `gen_fig_ch2_data_overview.py`, `gen_acf.py`, `gen_data_overview.py` |
| NSIDC mask NetCDF | `gen_fig_arctic_map.py` |

Scripts that **train models from scratch** (no pre-computed data needed):
- `gen_fig37.py` — trains LR/RNN/LSTM for monthly RMSE
- `gen_ablation_e7.py` — trains 5 variants x 5 seeds
- `gen_fig_lstm_loss.py`, `gen_loss_curves.py` — trains models for loss curves
- `gen_fig_loss_schemes.py` — trains short+medium LSTM

## Thesis Compilation Pipeline (write/)

The `write/` directory contains the manuscript source and compilation tools:

```
write/paper.md                  ← Main thesis manuscript (372 lines)
write/chapters/                 ← Per-chapter markdown
write/chapters/_write_ch3.py    ← Chapter 3: injects figure refs
write/chapters/_write_ch4.py    ← Chapter 4: injects figure refs
write/chapters/build_final.py   ← Markdown → .docx via python-docx
write/chapters/_compile_final.py ← Final compilation
```

The workflow: edit `paper.md` → run `_write_ch*.py` to inject figures → run `build_final.py` to compile .docx. The compiled .docx is ~13MB and should be gitignored.

## gen_121238.js

A standalone Node.js test script that generates a minimal DOCX file using the `docx` npm library. **Not related to the project** — should be deleted or added to .gitignore.

## Known Issues

- [main.py](main.py) has duplicate `set_all_seeds()` (lines 29-38 and 44-53); second shadows the first. `run_e7_baselines.py` and `compare_models.py` also have their own copies instead of importing from `src.utils.set_seed()`.
- [util_check.py](util_check.py) has its own hardcoded path (separate from config.py, different folder name)
- [src/data_preprocessing.py](src/data_preprocessing.py) `__main__` block has stale hardcoded path with typos (`arctic_seaice` not `2 +ao arctic_seaice`, `date` not `data`)
- [src/hybrid_model.py](src/hybrid_model.py) is an incomplete work-in-progress (`TinyLSTM` class only, file truncated mid-line) — not used by any script
- `paper_figures/thesis_data.py` is manually maintained — re-running experiments requires manual dict updates
- `config.py` references `TRAIN_SPLIT`/`VAL_SPLIT` in log messages (main.py lines 338-339) but these attributes are never defined in config.py
- The untracked files on the `papaper-write` branch (`gen_121238.js`, all new `gen_fig*.py`, `nature_figure_config.py`, `thesis_data.py`, `write/`) need to be committed
