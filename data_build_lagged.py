"""⭐ DATA: Build lagged AO/SST features (t, t-1, t-2) + SST mask for dual-encoder."""

For each (year, month):
  - ao, ao_lag1, ao_lag2:  current + 1-month + 2-month lagged AO
  - sst, sst_lag1, sst_lag2: current + lagged SST (0 if missing)
  - sst_mask: 1 if SST available for all 3 months, else 0

Output: data/lagged_features.csv
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICE_DIR = os.path.join(BASE_DIR, "data", "raw")
AO_PATH = os.path.join(BASE_DIR, "data", "ao_monthly.csv")
SST_PATH = os.path.join(BASE_DIR, "data", "arctic_sst_monthly.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "lagged_features.csv")


def build_lagged_features():
    # --- 1. Load sea ice data (full range, all months) ---
    import glob
    files = sorted(glob.glob(os.path.join(ICE_DIR, "N_*_extent_v4.0.csv")))
    df_list = []
    for f in files:
        month = int(os.path.basename(f).split('_')[1])
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        df['month'] = month
        df_list.append(df)
    df_ice = pd.concat(df_list, ignore_index=True)
    df_ice = df_ice.sort_values(['year', 'month']).reset_index(drop=True)
    df_ice = df_ice[['year', 'month', 'extent', 'area']]
    df_ice = df_ice[df_ice['area'] > 0]  # drop -9999
    print(f"Sea ice: {len(df_ice)} records ({df_ice['year'].min()}-{df_ice['year'].max()})")

    # --- 2. Load AO (1950-2025) ---
    df_ao = pd.read_csv(AO_PATH)
    print(f"AO: {len(df_ao)} records ({df_ao['year'].min()}-{df_ao['year'].max()})")

    # --- 3. Load SST (1981-2023) ---
    df_sst = pd.read_csv(SST_PATH)
    print(f"SST: {len(df_sst)} records ({df_sst['year'].min()}-{df_sst['year'].max()})")

    # --- 4. Merge: ice + AO (left join, keep all ice years) ---
    df = df_ice.merge(df_ao, on=['year', 'month'], how='left')
    print(f"Ice+AO merge: {len(df)} records")

    # --- 5. Merge SST (left join, SST NaN where missing) ---
    df = df.merge(df_sst, on=['year', 'month'], how='left')
    print(f"Ice+AO+SST merge: {len(df)} records ({df['year'].min()}-{df['year'].max()})")

    # --- 6. Compute lagged features ---
    # Shift to get lag1 and lag2 for AO and SST
    df['ao_lag1'] = df['ao'].shift(1)
    df['ao_lag2'] = df['ao'].shift(2)
    df['sst_lag1'] = df['sst'].shift(1)
    df['sst_lag2'] = df['sst'].shift(2)

    # SST mask: 1 if ALL of (sst, sst_lag1, sst_lag2) are non-NaN, else 0
    df['sst_mask'] = (
        df['sst'].notna() & df['sst_lag1'].notna() & df['sst_lag2'].notna()
    ).astype(int)

    # Fill missing SST values with 0 (mask will tell model to ignore them)
    for col in ['sst', 'sst_lag1', 'sst_lag2']:
        df[col] = df[col].fillna(0.0)

    # Fill missing AO values (only first 2 rows due to lag) with 0
    for col in ['ao', 'ao_lag1', 'ao_lag2']:
        df[col] = df[col].fillna(0.0)

    # --- 7. Drop first 2 rows (invalid lags) ---
    df = df.iloc[2:].reset_index(drop=True)

    # --- 8. Save ---
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved lagged features to: {OUT_PATH}")
    print(f"  Records: {len(df)} ({df['year'].min()}-{df['year'].max()})")
    print(f"  SST available (mask=1): {df['sst_mask'].sum()} records "
          f"({df['sst_mask'].sum()/len(df)*100:.0f}%)")
    print(f"  SST missing (mask=0): {(df['sst_mask']==0).sum()} records")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSample (first 3 rows):")
    print(df[['year', 'month', 'area', 'ao', 'ao_lag1', 'ao_lag2',
              'sst', 'sst_lag1', 'sst_lag2', 'sst_mask']].head(3))
    print(f"\nSample (post-1981, SST available):")
    post81 = df[df['sst_mask'] == 1]
    print(post81[['year', 'month', 'area', 'ao', 'ao_lag1', 'ao_lag2',
                  'sst', 'sst_lag1', 'sst_lag2', 'sst_mask']].head(3))


if __name__ == "__main__":
    build_lagged_features()
