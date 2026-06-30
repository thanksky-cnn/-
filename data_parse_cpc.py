"""⭐ DATA: 通用NOAA CPC月指数文本解析器 → CSV (year, month, value).

支持所有NOAA CPC标准格式的月指数数据（年+12月空格分隔）：
  - AO (Arctic Oscillation)
  - NAO (North Atlantic Oscillation)
  - Nino3.4 (ENSO index)
  - PDO (Pacific Decadal Oscillation)
  - SOI (Southern Oscillation Index)
  - 以及其他CPC格式的指数...

用法:
  python data_parse_cpc.py                          # 解析所有已配置的指数
  python data_parse_cpc.py --index nao              # 只解析NAO
  python data_parse_cpc.py --index nino34,pdo       # 解析多个

输入: data/<变量名>.txt (NOAA CPC标准格式)
输出: data/<变量名>_monthly.csv

CPC数据下载地址:
  - AO:  https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii
  - NAO: https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/nao.shtml
  - Nino3.4: https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices
  - PDO: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat
"""

import os
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 指数配置：{变量名: {input_file, output_column}}
# 所有指数默认使用NOAA CPC年+12月空格分隔格式
CPC_INDICES = {
    "ao": {
        "input": "NOAA .CPC. AO.monthly.txt",
        "output": "ao_monthly.csv",
        "column": "ao",
        "description": "Arctic Oscillation",
    },
    "nao": {
        "input": "NOAA_CPC_NAO_monthly.txt",
        "output": "nao_monthly.csv",
        "column": "nao",
        "description": "North Atlantic Oscillation",
    },
    "nino34": {
        "input": "NOAA_CPC_nino34_monthly.txt",
        "output": "nino34_monthly.csv",
        "column": "nino34",
        "description": "Nino 3.4 ENSO Index",
    },
    "pdo": {
        "input": "NOAA_NCEI_PDO_monthly.txt",
        "output": "pdo_monthly.csv",
        "column": "pdo",
        "description": "Pacific Decadal Oscillation",
    },
}


def parse_cpc_text(input_path, output_path, column_name, description=""):
    """通用CPC文本解析：年行 + 12月空格分隔值 → (year, month, value) CSV.

    CPC标准格式：
      Line 1: 月份缩写表头 (可跳过)
      Line 2+: <year> <Jan> <Feb> ... <Dec> (空格分隔，至少13个token)
    最后一行可能不足12个月（当年未完成）→ 跳过

    Args:
        input_path: CPC文本文件路径
        output_path: 输出CSV路径
        column_name: CSV中的列名 (e.g., "ao", "nao")
        description: 变量描述（仅用于日志）
    """
    if not os.path.exists(input_path):
        print(f"⚠ 文件不存在，跳过: {input_path}")
        print(f"  请从NOAA CPC下载数据后重试。")
        return

    records = []
    with open(input_path, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:  # 跳过表头行
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 13:  # 不完整的最后一年（当年未结束）
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue  # 跳过非数据行
        for month_idx in range(1, 13):
            try:
                value = float(parts[month_idx])
            except (ValueError, IndexError):
                continue
            records.append({"year": year, "month": month_idx, column_name: value})

    df = pd.DataFrame(records)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"✅ {description} ({column_name}): {len(df)} records "
          f"({df['year'].min()}-{df['year'].max()}) → {output_path}")


def parse_sstoi_format(input_path, output_path, column_name="nino34", description=""):
    """解析NOAA CPC SSTOI多指数格式，提取Nino3.4列。

    SSTOI文件格式（sstoi.indices）：
      YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
      1950   1   21.73 -0.72  25.93  0.01  28.37 -0.24  26.49  0.11

    包含4个Nino区域的SST和异常值，共10列。
    我们提取Nino3.4 SST（第9列）或Nino3.4 ANOM（第10列）。

    Args:
        input_path: SSTOI文本文件路径
        output_path: 输出CSV路径
        column_name: 输出列名（默认nino34）
        description: 变量描述
    """
    if not os.path.exists(input_path):
        print(f"⚠ 文件不存在，跳过: {input_path}")
        print(f"  请从 https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices 下载")
        return

    records = []
    with open(input_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("YR"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            # Nino3.4 SST 是第9列（0-indexed: 8），ANOM是第10列（9）
            # 使用ANOM（异常值）更常用，均值为0
            nino34_anom = float(parts[9])  # Nino3.4 anomaly
            records.append({"year": year, "month": month, column_name: nino34_anom})
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(records)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"✅ {description} ({column_name}): {len(df)} records "
          f"({df['year'].min()}-{df['year'].max()}) [SSTOI格式, Nino3.4 ANOM] → {output_path}")

    # 打印统计信息
    if len(df) > 0:
        el_nino_months = (df[column_name] > 0.5).sum()
        la_nina_months = (df[column_name] < -0.5).sum()
        print(f"   El Nino months (>+0.5): {el_nino_months} ({el_nino_months/len(df)*100:.0f}%)")
        print(f"   La Nina months (<-0.5): {la_nina_months} ({la_nina_months/len(df)*100:.0f}%)")


def parse_ncep_pdo_format(input_path, output_path, column_name="pdo", description=""):
    """解析NCEI PDO格式（可能与标准CPC格式不同）。

    NCEI PDO格式通常是两列：year+month, value
    尝试自动检测格式。
    """
    if not os.path.exists(input_path):
        print(f"⚠ 文件不存在，跳过: {input_path}")
        return

    with open(input_path, "r") as f:
        content = f.read()

    lines = [l.strip() for l in content.split("\n") if l.strip()]

    # 尝试判断格式：第一行数据是否能拆成13个以上token？
    sample_parts = lines[0].split()
    is_cpc_format = len(sample_parts) >= 13

    if is_cpc_format:
        parse_cpc_text(input_path, output_path, column_name, description)
    else:
        # 尝试两列格式：year+month 或 year month
        records = []
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                # 尝试 year month value 格式
                if len(parts) >= 3:
                    year = int(parts[0])
                    month = int(parts[1])
                    value = float(parts[2])
                else:
                    # 尝试 year_month value 格式
                    ym = parts[0]
                    if len(ym) == 6:  # YYYYMM
                        year = int(ym[:4])
                        month = int(ym[4:6])
                    else:
                        continue
                    value = float(parts[1])
                records.append({"year": year, "month": month, column_name: value})
            except (ValueError, IndexError):
                continue

        df = pd.DataFrame(records)
        df = df.sort_values(["year", "month"]).reset_index(drop=True)
        df.to_csv(output_path, index=False)
        print(f"✅ {description} ({column_name}): {len(df)} records "
              f"({df['year'].min()}-{df['year'].max()}) [NCEI 2-col格式] → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="通用NOAA CPC月指数解析器 → CSV"
    )
    parser.add_argument(
        "--index", type=str, default="all",
        help="逗号分隔的指数名 (e.g., 'nao,nino34,pdo')，默认'all'解析全部"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有已配置的指数及其预期输入文件"
    )
    args = parser.parse_args()

    if args.list:
        print("已配置的CPC指数：")
        for name, cfg in CPC_INDICES.items():
            full_input = os.path.join(DATA_DIR, cfg["input"])
            exists = "✅" if os.path.exists(full_input) else "❌"
            print(f"  {name:10s} → {cfg['output']:20s}  ({cfg['description']}) {exists}")
        return

    if args.index == "all":
        indices_to_parse = list(CPC_INDICES.keys())
    else:
        indices_to_parse = [s.strip() for s in args.index.split(",")]

    for name in indices_to_parse:
        if name not in CPC_INDICES:
            print(f"⚠ 未知指数: {name}，跳过。可用: {list(CPC_INDICES.keys())}")
            continue

        cfg = CPC_INDICES[name]
        input_path = os.path.join(DATA_DIR, cfg["input"])
        output_path = os.path.join(DATA_DIR, cfg["output"])

        # Nino3.4使用SSTOI专用格式
        if name == "nino34":
            parse_sstoi_format(input_path, output_path, cfg["column"], cfg["description"])
        elif name == "pdo":
            parse_ncep_pdo_format(input_path, output_path, cfg["column"], cfg["description"])
        else:
            parse_cpc_text(input_path, output_path, cfg["column"], cfg["description"])


if __name__ == "__main__":
    main()
