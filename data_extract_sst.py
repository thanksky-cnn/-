"""
Extract Arctic SST (30.98°N–90°N, all longitudes) from NOAA OI SST V2 NetCDF.
Averages over the Arctic region for each month, then saves as CSV.

Input:  data/sst.mnmean.nc
Output: data/arctic_sst_monthly.csv  (year, month, sst)
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "sst.mnmean.nc")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "arctic_sst_monthly.csv")


def extract_arctic_sst(input_path, output_path):
    ds = xr.open_dataset(input_path)

    # Select Arctic latitudes (30.98°N to 90°N), all longitudes
    sst_arctic = ds.sst.sel(lat=ds.lat[ds.lat >= 30.98])

    # Spatial mean over (lat, lon) → 1D time series
    sst_mean = sst_arctic.mean(dim=("lat", "lon"))

    # Convert to DataFrame
    times = pd.to_datetime(sst_mean.time.values)
    records = []
    for t, val in zip(times, sst_mean.values):
        if np.isnan(val):
            continue  # skip any NaN months
        records.append({"year": t.year, "month": t.month, "sst": float(val)})

    df = pd.DataFrame(records)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"Arctic SST extracted: {len(df)} records")
    print(f"  Time range: {df['year'].min()}-{df['month'].min():02d} to "
          f"{df['year'].max()}-{df['month'].max():02d}")
    print(f"  SST range:  {df['sst'].min():.2f} to {df['sst'].max():.2f} °C")
    print(f"Saved to: {output_path}")
    ds.close()


if __name__ == "__main__":
    extract_arctic_sst(INPUT_FILE, OUTPUT_FILE)
