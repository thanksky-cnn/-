"""⭐ DATA: 从ERA5月平均NetCDF提取北极区域平均2m气温 (T2M).

Input: ERA5 monthly averaged 2m temperature NetCDF (需要从CDS下载)
Output: data/arctic_t2m_monthly.csv (year, month, t2m)

数据下载 (CDS API):
  Dataset: "reanalysis-era5-single-levels-monthly-means"
  Variable: "2m_temperature"
  Area: [90, -180, 60, 180]  (N, W, S, E) → 北极60°N-90°N
  Time: 1940-01-01 to present
  Format: NetCDF

CDS下载脚本示例 (另存为 download_era5_t2m.py):
```python
import cdsapi
c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': [str(y) for y in range(1940, 2026)],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'time': '00:00',
        'area': [90, -180, 60, 180],
        'format': 'netcdf',
    },
    'data/era5_arctic_t2m_monthly.nc'
)
```

如果CDS API不可用，也可以从ERA5月度再分析数据手动下载NetCDF文件。
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "era5_arctic_t2m_monthly.nc")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "arctic_t2m_monthly.csv")


def extract_arctic_t2m(input_path, output_path, lat_min=60, lat_max=90):
    """
    从ERA5 NetCDF提取北极区域平均2m气温。

    Args:
        input_path: ERA5 NetCDF文件路径
        output_path: 输出CSV路径
        lat_min: 最低纬度（默认60°N）
        lat_max: 最高纬度（默认90°N）
    """
    import xarray as xr

    if not os.path.exists(input_path):
        print(f"⚠ ERA5 T2M文件不存在: {input_path}")
        print(f"  请先通过CDS API下载数据:")
        print(f"  Dataset: reanalysis-era5-single-levels-monthly-means")
        print(f"  Variable: 2m_temperature")
        print(f"  Area: [90, -180, 60, 180]")
        print(f"  输出到: {input_path}")
        return

    print(f"正在读取ERA5 T2M数据: {input_path}")
    ds = xr.open_dataset(input_path)

    # ERA5变量名通常是 't2m'，单位: K
    if "t2m" in ds.data_vars:
        var_name = "t2m"
    elif "var167" in ds.data_vars:
        var_name = "var167"  # 某些格式下的GRIB编码
    else:
        print(f"可用变量: {list(ds.data_vars.keys())}")
        raise KeyError("找不到2m气温变量 (t2m)，请检查NetCDF文件内容")

    t2m = ds[var_name]

    # 纬度筛选（ERA5纬度可能是从北到南或从南到北）
    lat = ds["latitude"] if "latitude" in ds.coords else ds["lat"]
    lat_values = lat.values

    # 判断纬度方向
    if lat_values[0] > lat_values[-1]:  # 北到南
        lat_mask = (lat_values <= lat_max) & (lat_values >= lat_min)
    else:  # 南到北
        lat_mask = (lat_values >= lat_min) & (lat_values <= lat_max)

    # 空间平均（cos(lat)面积加权）
    weights = np.cos(np.deg2rad(lat_values[lat_mask]))
    t2m_arctic = t2m.where(lat_mask, drop=True).weighted(weights).mean(dim=["latitude", "longitude"])

    # 提取时间坐标
    time_coord = ds["valid_time"] if "valid_time" in ds.coords else ds["time"]
    time_values = pd.to_datetime(time_coord.values)

    # 构建DataFrame
    records = []
    for i, t in enumerate(time_values):
        val = float(t2m_arctic.values[i])
        # Kelvin → Celsius
        val_c = val - 273.15
        records.append({"year": t.year, "month": t.month, "t2m": round(val_c, 4)})

    df = pd.DataFrame(records)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"✅ 北极T2M (60°N-90°N) 提取完成:")
    print(f"  记录数: {len(df)} ({df['year'].min()}-{df['year'].max()})")
    print(f"  均值: {df['t2m'].mean():.2f}°C, 范围: [{df['t2m'].min():.2f}, {df['t2m'].max():.2f}]°C")
    print(f"  输出: {output_path}")

    ds.close()


if __name__ == "__main__":
    extract_arctic_t2m(INPUT_FILE, OUTPUT_FILE)
