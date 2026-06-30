"""⭐ DATA: 从ERA5月平均NetCDF提取北极区域平均海平面气压 (SLP).

Input: ERA5 monthly averaged mean sea level pressure NetCDF (需要从CDS下载)
Output: data/arctic_slp_monthly.csv (year, month, slp)

数据下载 (CDS API):
  Dataset: "reanalysis-era5-single-levels-monthly-means"
  Variable: "mean_sea_level_pressure"
  Area: [90, -180, 60, 180]  (N, W, S, E) → 北极60°N-90°N
  Time: 1940-01-01 to present
  Format: NetCDF

CDS下载脚本示例:
```python
import cdsapi
c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'mean_sea_level_pressure',
        'year': [str(y) for y in range(1940, 2026)],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'time': '00:00',
        'area': [90, -180, 60, 180],
        'format': 'netcdf',
    },
    'data/era5_arctic_slp_monthly.nc'
)
```
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "era5_arctic_slp_monthly.nc")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "arctic_slp_monthly.csv")


def extract_arctic_slp(input_path, output_path, lat_min=60, lat_max=90):
    """
    从ERA5 NetCDF提取北极区域平均海平面气压。

    Args:
        input_path: ERA5 NetCDF文件路径
        output_path: 输出CSV路径
        lat_min: 最低纬度（默认60°N）
        lat_max: 最高纬度（默认90°N）
    """
    import xarray as xr

    if not os.path.exists(input_path):
        print(f"⚠ ERA5 SLP文件不存在: {input_path}")
        print(f"  请先通过CDS API下载数据:")
        print(f"  Dataset: reanalysis-era5-single-levels-monthly-means")
        print(f"  Variable: mean_sea_level_pressure")
        print(f"  Area: [90, -180, 60, 180]")
        print(f"  输出到: {input_path}")
        return

    print(f"正在读取ERA5 SLP数据: {input_path}")
    ds = xr.open_dataset(input_path)

    # ERA5变量名通常是 'msl'，单位: Pa
    var_name = None
    for candidate in ["msl", "mslp", "mean_sea_level_pressure"]:
        if candidate in ds.data_vars:
            var_name = candidate
            break

    if var_name is None:
        print(f"可用变量: {list(ds.data_vars.keys())}")
        raise KeyError("找不到海平面气压变量 (msl/mslp)，请检查NetCDF文件内容")

    slp = ds[var_name]

    # 纬度筛选
    lat = ds["latitude"] if "latitude" in ds.coords else ds["lat"]
    lon = ds["longitude"] if "longitude" in ds.coords else ds["lon"]
    lat_values = lat.values

    # 判断纬度方向
    if lat_values[0] > lat_values[-1]:  # 北到南
        lat_mask = (lat_values <= lat_max) & (lat_values >= lat_min)
    else:
        lat_mask = (lat_values >= lat_min) & (lat_values <= lat_max)

    # 空间平均（cos(lat)面积加权）
    weights = np.cos(np.deg2rad(lat_values[lat_mask]))
    slp_arctic = slp.where(lat_mask, drop=True).weighted(weights).mean(dim=["latitude", "longitude"])

    # 提取时间坐标
    time_coord = ds["valid_time"] if "valid_time" in ds.coords else ds["time"]
    time_values = pd.to_datetime(time_coord.values)

    # 构建DataFrame
    records = []
    for i, t in enumerate(time_values):
        val = float(slp_arctic.values[i])
        # Pa → hPa (百帕)
        val_hpa = val / 100.0
        records.append({"year": t.year, "month": t.month, "slp": round(val_hpa, 4)})

    df = pd.DataFrame(records)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"✅ 北极SLP (60°N-90°N) 提取完成:")
    print(f"  记录数: {len(df)} ({df['year'].min()}-{df['year'].max()})")
    print(f"  均值: {df['slp'].mean():.2f} hPa, 范围: [{df['slp'].min():.2f}, {df['slp'].max():.2f}] hPa")
    print(f"  输出: {output_path}")

    ds.close()


if __name__ == "__main__":
    extract_arctic_slp(INPUT_FILE, OUTPUT_FILE)
