"""⭐ 下载辅助：NOAA CPC气候指数 + ERA5再分析数据下载指南。

本脚本提供下载URL和CDS API示例，帮助获取思路2所需的全部数据。
也可作为Python下载脚本直接运行（需要requests和cdsapi库）。
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ============================================================
# Tier 1: NOAA CPC 气候指数 (文本文件，免费，无需API Key)
# ============================================================

CPC_DOWNLOADS = {
    "NAO": {
        "url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii",
        "save_as": "NOAA_CPC_NAO_monthly.txt",
        "description": "North Atlantic Oscillation (1950-present)",
        "note": "CPC标准格式：年行 + 12月值",
    },
    "Nino3.4": {
        "url": "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices",
        "save_as": "NOAA_CPC_nino34_monthly.txt",
        "description": "ENSO Nino 3.4 Index (1950-present)",
        "note": "多列格式，脚本自动提取Nino3.4列",
    },
    "PDO": {
        "url": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
        "save_as": "NOAA_NCEI_PDO_monthly.txt",
        "description": "Pacific Decadal Oscillation (1900-present)",
        "note": "NCEI两列格式（year+month, value）",
    },
}

# ============================================================
# Tier 2: ERA5 再分析数据 (NetCDF，需要CDS API Key)
# ============================================================

ERA5_DOWNLOADS = {
    "T2M": {
        "dataset": "reanalysis-era5-single-levels-monthly-means",
        "variable": "2m_temperature",
        "save_as": "era5_arctic_t2m_monthly.nc",
        "description": "2m air temperature, Arctic (60°N-90°N)",
        "area": [90, -180, 60, 180],  # N, W, S, E
    },
    "SLP": {
        "dataset": "reanalysis-era5-single-levels-monthly-means",
        "variable": "mean_sea_level_pressure",
        "save_as": "era5_arctic_slp_monthly.nc",
        "description": "Mean sea level pressure, Arctic (60°N-90°N)",
        "area": [90, -180, 60, 180],
    },
}


def print_cpc_instructions():
    """打印CPC数据下载说明。"""
    print("=" * 70)
    print("  Tier 1: NOAA CPC 气候指数 — 浏览器直接下载")
    print("=" * 70)
    print()
    for name, info in CPC_DOWNLOADS.items():
        save_path = os.path.join(DATA_DIR, info["save_as"])
        exists = "✅ 已有" if os.path.exists(save_path) else "❌ 需要下载"
        print(f"  [{name}] {info['description']}")
        print(f"    URL:  {info['url']}")
        print(f"    保存: {info['save_as']}")
        print(f"    格式: {info['note']}")
        print(f"    状态: {exists}")
        print()


def print_era5_instructions():
    """打印ERA5 CDS API下载说明。"""
    print("=" * 70)
    print("  Tier 2: ERA5 再分析数据 — CDS API 下载")
    print("=" * 70)
    print()
    print("  前置条件:")
    print("    1. 注册 CDS 账号: https://cds.climate.copernicus.eu/")
    print("    2. 安装 CDS API: pip install cdsapi")
    print("    3. 配置 ~/.cdsapirc 文件（放入URL和Key）")
    print()

    for name, info in ERA5_DOWNLOADS.items():
        save_path = os.path.join(DATA_DIR, info["save_as"])
        exists = "✅ 已有" if os.path.exists(save_path) else "❌ 需要下载"
        print(f"  [{name}] {info['description']}")
        print(f"    Dataset: {info['dataset']}")
        print(f"    Variable: {info['variable']}")
        print(f"    Area: {info['area']}  (N,W,S,E)")
        print(f"    保存: {info['save_as']}")
        print(f"    状态: {exists}")
        print()


def print_era5_python_script():
    """打印可直接运行的CDS API下载脚本。"""
    print("=" * 70)
    print("  可运行的CDS API下载脚本 (另存为 download_era5.py)")
    print("=" * 70)
    print("""
import cdsapi

c = cdsapi.Client()
YEARS = [str(y) for y in range(1940, 2026)]
MONTHS = [f'{m:02d}' for m in range(1, 13)]
AREA = [90, -180, 60, 180]  # Arctic: 60N-90N

# 下载 T2M
print("Downloading ERA5 2m temperature...")
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': YEARS,
        'month': MONTHS,
        'time': '00:00',
        'area': AREA,
        'format': 'netcdf',
    },
    'data/era5_arctic_t2m_monthly.nc'
)
print("T2M download complete.")

# 下载 SLP
print("Downloading ERA5 mean sea level pressure...")
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'mean_sea_level_pressure',
        'year': YEARS,
        'month': MONTHS,
        'time': '00:00',
        'area': AREA,
        'format': 'netcdf',
    },
    'data/era5_arctic_slp_monthly.nc'
)
print("SLP download complete.")
""")


def try_download_cpc():
    """尝试用requests下载CPC数据（可能因网络限制失败）。"""
    try:
        import requests
    except ImportError:
        print("⚠ requests未安装，跳过自动下载。请手动从浏览器下载。")
        print("  安装: pip install requests")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    for name, info in CPC_DOWNLOADS.items():
        save_path = os.path.join(DATA_DIR, info["save_as"])
        if os.path.exists(save_path):
            print(f"⏭ {name}: 已存在，跳过")
            continue

        print(f"⬇ 下载 {name} → {info['save_as']}...")
        try:
            resp = requests.get(info["url"], timeout=30)
            resp.raise_for_status()

            # 对于Nino3.4多列文件，保留原始格式
            content = resp.text

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)

            lines = content.strip().split("\n")
            print(f"  ✅ 成功: {len(lines)} 行")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            print(f"  请手动从浏览器下载: {info['url']}")
            print(f"  保存为: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="数据下载辅助工具 — 打印下载说明 / 尝试自动下载CPC数据"
    )
    parser.add_argument("--download", action="store_true",
                        help="尝试自动下载Tier 1 CPC数据 (需要requests)")
    parser.add_argument("--cpc", action="store_true",
                        help="仅显示CPC下载说明")
    parser.add_argument("--era5", action="store_true",
                        help="仅显示ERA5下载说明 + Python脚本")
    args = parser.parse_args()

    if args.download:
        print("正在尝试下载Tier 1 CPC数据...\n")
        try_download_cpc()
        print("\n下载完成。运行 python data_parse_cpc.py 解析数据。")
    elif args.cpc:
        print_cpc_instructions()
    elif args.era5:
        print_era5_instructions()
        print_era5_python_script()
    else:
        print_cpc_instructions()
        print_era5_instructions()
        print_era5_python_script()
        print("\n" + "=" * 70)
        print("  下载后运行以下命令完成数据准备：")
        print("=" * 70)
        print("  python data_parse_cpc.py           # 解析所有CPC指数 → CSV")
        print("  python data_extract_t2m.py         # 提取北极T2M")
        print("  python data_extract_slp.py         # 提取北极SLP")
        print("  python data_build_lagged_v2.py    # 构建通用滞后特征")
        print()


if __name__ == "__main__":
    main()
