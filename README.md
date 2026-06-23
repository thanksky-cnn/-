# Arctic Sea Ice Prediction with LSTM

Predict Arctic sea ice area/extent 12 months ahead using LSTM neural networks trained on historical NSIDC observations (1979--present). Tests single-variable, multi-variable, and dual-encoder architectures.

## Data Sources

| Variable | Source | Period | Resolution |
|----------|--------|--------|------------|
| Sea Ice Area/Extent | [NSIDC Sea Ice Index v4](https://nsidc.org/data/g02202) | 1979--2025 | Monthly |
| Arctic Oscillation (AO) | [NOAA CPC](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/ao.shtml) | 1950--2025 | Monthly |
| Sea Surface Temperature (SST) | [NOAA OI SST V2](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html) | 1981--2023 | 1°×1° monthly |

## Installation

```bash
pip install -r requirements.txt
# optional: for SST NetCDF processing
pip install xarray netCDF4
```

## Quick Start

```bash
# Edit config.py first: set BASE_DIR and DATA_DIR to your absolute paths

# Basic LSTM training
python main.py

# Compare univariate vs bivariate vs trivariate
python run_trivariate.py

# Optimized dual-encoder with ensemble (best model)
python run_optimize_e7.py
```

## Project Structure

```
project/
├── config.py                  ← configuration hub
├── main.py                    ← core training pipeline
├── compare_models.py          ← baseline comparison (LR vs RNN vs LSTM)
│
├── run_trivariate.py          ← ⭐ uni vs bi vs tri LSTM
├── run_e7_baselines.py        ← ⭐ E7 dual-encoder vs LR vs RNN
├── run_optimize_e7.py         ← ⭐ final: 50-trial Optuna + 5-seed ensemble
│
├── data_parse_ao.py           ← data: parse AO
├── data_extract_sst.py        ← data: extract SST
├── data_build_lagged.py       ← data: lagged features + SST mask
│
├── tune_univariate.py         ← tuning: univariate
├── tune_bivariate.py          ← tuning: bivariate
├── tune_trivariate.py         ← tuning: trivariate
│
├── summary.py                 ← compile all results → tables + charts
├── util_check.py              ← data path diagnostic
├── util_monthly.py            ← per-month error breakdown
├── util_export.py             ← per-sample prediction CSV export
│
├── old_*.py                   ← 📦 archive (superseded scripts)
│
├── data/
│   ├── raw/                   # N_01_extent_v4.0.csv ... N_12_extent_v4.0.csv
│   ├── ao_monthly.csv         # Parsed AO (year, month, ao)
│   ├── arctic_sst_monthly.csv # Arctic-averaged SST
│   └── lagged_features.csv    # Lagged AO/SST with SST mask
│
├── src/
│   ├── data_preprocessing.py  # Data loading: uni/multi/dual-encoder
│   ├── model.py               # SeaIceLSTM, SeaIceDualEncoderLSTM, LR, RNN
│   ├── train.py               # Training loop with early stopping
│   ├── dataset.py             # SeaIceDataset
│   └── utils.py               # Metrics, seed setting
│
├── outputs/
│   ├── models/                # Saved .pth files
│   ├── plots/                 # All comparison plots
│   ├── results/               # Metrics files + best_params JSONs
│   └── summary/               # Final compiled report (CSV + charts)
│
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Configuration

Edit `config.py`:

```python
BASE_DIR = r"your/project/path"   # MUST update for your machine
DATA_DIR = r"your/data/path"

PREDICTION_SCHEME = "long"        # "short"(1mo) | "medium"(6mo) | "long"(12mo)
TARGET_COLUMN = "area"            # "area" or "extent"
```

The `PREDICTION_SCHEME` selects a preset from `_SCHEME_PARAMS` that atomically sets `OUTPUT_LEN`, `HIDDEN_SIZE`, `NUM_LAYERS`, `DROPOUT`, `BATCH_SIZE`, `LEARNING_RATE`, and `WEIGHT_DECAY`.

## Model Architectures

### 1. SeaIceLSTM (single encoder)
```
Input: (batch, 12, n_features)
  LSTM(12mo) -> last hidden -> Dropout -> Linear -> (batch, output_len)
```
- Simplified mode (`NUM_LAYERS=1`): single LSTM layer, recommended for ~500 samples
- Complex mode (`NUM_LAYERS>1`): 2-layer LSTM + LayerNorm + seasonal/trend branches

### 2. LinearRegressionModel
```
Input: (batch, 12) -> Linear(12, output_len)
```
156-parameter linear baseline. Stronger than expected due to ice seasonality.

### 3. SimpleRNNModel
```
Input: (batch, 12, 1) -> RNN(64) -> Dropout -> Linear -> (batch, output_len)
```
Basic RNN for validation comparison.

### 4. SeaIceDualEncoderLSTM (E7)
```
Main encoder:  LSTM(12mo sea ice) -> main_hidden      ┐
                                                        ├─ Concat -> Linear -> output
Aux encoder:   LSTM(last N mo lagged AO/SST/mask) -> aux_hidden ┘
               7 channels: [ao, ao_lag1, ao_lag2, sst, sst_lag1, sst_lag2, sst_mask]
               Higher dropout on aux path (0.6) to suppress noise
```
Prevents AO/SST noise from contaminating the main LSTM's recurrent path.

## Experiment Results Summary

All results use **input=12 months, output=12 months**, year-based split (train: 1979--2010, val: 2011--2015, test: 2016--2025).

| ID | Model | Features | Data | Params | RMSE | MAE | MAPE |
|:--:|-------|----------|------|-------:|-----:|-----:|-----:|
| E1 | Univariate LSTM | area | 1979--2025 | 268K | 0.4975 | 0.3982 | 6.43% |
| E2 | Bivariate LSTM | area+ao | 1979--2025 | 269K | 0.5474 | 0.4300 | 7.00% |
| E3 | Bivariate LSTM (tuned) | area+ao | 1979--2025 | 69K | 0.5397 | 0.4222 | 7.07% |
| E4 | Trivariate LSTM | area+ao+sst | 1981--2023 | 18K | 0.6151 | 0.4952 | 7.35% |
| E7v1 | Dual-Encoder | area+lag ao/sst | 1979--2025* | 74K | 0.4974 | 0.3998 | 6.45% |
| **E7opt** | **Dual-Encoder (ens5)** | **area+lag ao/sst** | **1979--2025*** | **274K×5** | **0.5014** | **0.3967** | **6.46%** |
| LR | LinearRegression | area | 1979--2025 | **156** | 0.5317 | 0.4174 | 6.80% |
| RNN | SimpleRNN | area | 1979--2025 | 5K | 0.5255 | 0.4097 | 6.90% |

*\* SST masked (set to 0) pre-1981 to preserve full ice record.*

## Key Findings

1. **Univariate LSTM is a strong baseline** (RMSE 0.4975). Linear regression (156 params) achieves RMSE 0.5317 -- seasonality is highly linear.

2. **Simple equal-weight multivariate models ALL underperform.** Concatenating AO/SST as extra features into a single LSTM allows noise to contaminate the hidden state across all 12 time steps.

3. **Dual-Encoder (E7) is the only multivariate architecture to beat univariate.** It achieves RMSE 0.4974 (single seed) / 0.5014 (5-seed ensemble), outperforming LinearRegression by 5.7%.

4. **Four key improvements make E7 work:**
   - **Lagged features** (t, t-1, t-2) to capture delayed atmospheric/oceanic forcing
   - **Separate encoders** -- main LSTM sees only sea ice; aux LSTM sees only recent AO/SST
   - **SST mask** -- left-join preserves 1979--1980 data; mask channel tells the model when SST is missing
   - **Aux regularization** -- dropout=0.6 on aux path forces the model to ignore noise

5. **Ensemble stabilizes small-sample training.** 5-seed averaging reduces RMSE variance from ~0.02 to <0.005.

6. **Do NOT truncate data for SST.** Inner join (E4) loses 2 years of training data; left join + mask (E7) uses all available data.

## Complete Results

See `outputs/summary/` for the full compiled report:
- `01_overview.csv` -- All experiment metrics
- `02_hyperparams.csv` -- All hyperparameters
- `03_monthly_rmse.csv` -- Monthly RMSE breakdown
- `complete_report.txt` -- Formatted text report with findings
- `chart_*.png` -- 7 comparison charts

## References

- NSIDC Sea Ice Index: https://nsidc.org/data/g02202
- NOAA CPC AO Index: https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/
- NOAA OI SST V2: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html
- PyTorch: https://pytorch.org
- Optuna: https://optuna.org
