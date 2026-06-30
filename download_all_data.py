#!/usr/bin/env python
"""一键下载思路2所需全部气候数据。

网络限制：如果在公司/学校网络内下载失败，请用手机热点或VPN重试。
所有数据均为美国政府公开数据（NOAA/ECMWF），免费且无需注册。

Usage:
  python download_all_data.py              # 下载所有
  python download_all_data.py --cpc-only   # 仅CPC指数
  python download_all_data.py --check      # 检查已有文件
"""

import os, sys, subprocess, argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# CPC 气候指数下载 (Tier 1 — 免费，无需API Key)
# ============================================================
CPC_FILES = {
    "NAO": {
        "url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii",
        "file": "NOAA_CPC_NAO_monthly.txt",
        "desc": "North Atlantic Oscillation",
    },
    "Nino3.4": {
        "url": "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices",
        "file": "NOAA_CPC_nino34_monthly.txt",
        "desc": "ENSO Nino 3.4 Index",
    },
    "PDO": {
        # PDO from NCEI (ERSST v5)
        "url": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
        "file": "NOAA_NCEI_PDO_monthly.txt",
        "desc": "Pacific Decadal Oscillation",
    },
}


def download_file(url, save_path, desc):
    """Download a single file using curl or Python requests."""
    fname = os.path.basename(save_path)

    if os.path.exists(save_path):
        # Check if file has content
        with open(save_path, "r") as f:
            content = f.read().strip()
        if len(content) > 100:
            print(f"  ✅ {desc}: already exists ({len(content.split(chr(10)))} lines)")
            return True
        else:
            print(f"  ⚠ {desc}: file exists but empty, re-downloading...")

    # Try curl first
    try:
        result = subprocess.run(
            ["curl", "-L", "--connect-timeout", "15", "--max-time", "30",
             "-o", save_path, url],
            capture_output=True, text=True, timeout=35
        )
        if result.returncode == 0 and os.path.exists(save_path):
            with open(save_path, "r") as f:
                lines = len(f.readlines())
            if lines > 5:
                print(f"  ✅ {desc}: downloaded ({lines} lines)")
                return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: Python requests
    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content = resp.text
        if len(content) > 100:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
            lines = content.count("\n")
            print(f"  ✅ {desc}: downloaded via requests ({lines} lines)")
            return True
    except Exception as e:
        print(f"  ❌ {desc}: download failed — {e}")
        print(f"     Manual download: {url}")
        print(f"     Save as: {save_path}")
        return False

    return False


def check_existing():
    """Check which data files already exist."""
    print("=" * 60)
    print("  数据文件检查")
    print("=" * 60)

    expected_files = [
        ("data/ao_monthly.csv", "AO (Arctic Oscillation)"),
        ("data/NOAA_CPC_NAO_monthly.txt", "NAO text (to parse)"),
        ("data/nao_monthly.csv", "NAO (parsed CSV)"),
        ("data/NOAA_CPC_nino34_monthly.txt", "Nino3.4 text (to parse)"),
        ("data/nino34_monthly.csv", "Nino3.4 (parsed CSV)"),
        ("data/NOAA_NCEI_PDO_monthly.txt", "PDO text (to parse)"),
        ("data/pdo_monthly.csv", "PDO (parsed CSV)"),
        ("data/arctic_sst_monthly.csv", "Arctic SST"),
        ("data/era5_arctic_t2m_monthly.nc", "ERA5 T2M (NetCDF)"),
        ("data/arctic_t2m_monthly.csv", "T2M (parsed CSV)"),
        ("data/era5_arctic_slp_monthly.nc", "ERA5 SLP (NetCDF)"),
        ("data/arctic_slp_monthly.csv", "SLP (parsed CSV)"),
        ("data/lagged_features_v2.csv", "⭐ Lagged features v2 (all variables)"),
    ]

    for rel_path, desc in expected_files:
        full_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(full_path):
            size_kb = os.path.getsize(full_path) / 1024
            print(f"  ✅ {desc:40s} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {desc:40s} — missing")


def main():
    parser = argparse.ArgumentParser(description="Download all climate data for multivariate study")
    parser.add_argument("--cpc-only", action="store_true", help="Only download CPC indices")
    parser.add_argument("--check", action="store_true", help="Check existing files only")
    args = parser.parse_args()

    if args.check:
        check_existing()
        return

    print("=" * 60)
    print("  Tier 1: NOAA CPC Climate Indices")
    print("=" * 60)

    for name, info in CPC_FILES.items():
        save_path = os.path.join(DATA_DIR, info["file"])
        download_file(info["url"], save_path, f"{name} ({info['desc']})")

    if args.cpc_only:
        print("\n✅ CPC downloads complete. Now run:")
        print("  python data_parse_cpc.py")
        print("  python data_build_lagged_v2.py --vars ao,sst,nao,nino34,pdo")
        return

    print("\n" + "=" * 60)
    print("  Tier 2: ERA5 Reanalysis Data")
    print("=" * 60)
    print("  ERA5 requires CDS API setup:")
    print("    1. Register: https://cds.climate.copernicus.eu/")
    print("    2. Install:  pip install cdsapi")
    print("    3. Configure: ~/.cdsapirc with your key")
    print()
    print("  Then run the CDS download script:")
    print("  python download_data_helper.py --era5")
    print()
    print("=" * 60)
    print("  Next Steps")
    print("=" * 60)
    print("  1. python data_parse_cpc.py         # Parse CPC text -> CSV")
    print("  2. python data_extract_t2m.py       # Extract ERA5 T2M (after download)")
    print("  3. python data_extract_slp.py       # Extract ERA5 SLP (after download)")
    print("  4. python data_build_lagged_v2.py  # Build lagged features")
    print("  5. python run_phase1_single_variable.py  # Run experiments!")


if __name__ == "__main__":
    main()
