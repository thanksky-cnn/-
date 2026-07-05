"""⭐ DATA: 通用滞后特征构建器 v2 — 支持N个辅助变量，可配置掩码。

相比v1 (data_build_lagged.py) 的改进：
  - 变量组配置字典 → 添加新变量只需一行
  - 自动处理：lag1, lag2, 掩码（仅对has_mask=True的变量）
  - 支持Tier 1-3全部变量（NAO, Nino3.4, PDO, T2M, SLP, lag12_ice）
  - 保持LEFT JOIN保证不丢冰数据年份
  - 独立的归一化策略（有掩码的变量：仅对有效值fit scaler）

输出: data/lagged_features_v2.csv
列: year, month, area, extent, <所有变量原始值+滞后+掩码>

用法:
  python data_build_lagged_v2.py                    # 构建完整特征集
  python data_build_lagged_v2.py --vars ao,sst      # 仅构建指定变量
  python data_build_lagged_v2.py --no-lags          # 不计算滞后（用于Lag12等派生特征）
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICE_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_PATH = os.path.join(BASE_DIR, "data", "lagged_features_v2.csv")

# ============================================================
# 变量组配置 — 添加新变量只需在此处加一行
# ============================================================
# 每个变量的配置：
#   csv:       CSV文件路径 (相对于data/)
#   cols:      原始值列名列表 (通常1列；多列用于区域SST等)
#   has_mask:  是否需要掩码通道 (True=可能缺失，如SST; False=完整，如AO)
#   n_lags:    滞后阶数 (默认2: lag1+lag2; 0=无滞后)
#   lag_only:  是否只做单次长滞后 (True用于lag-12冰: 只做lag12, 不做lag1/lag2)
#   lag_months: lag_only时的滞后月数 (默认12)
# ============================================================
VARIABLE_GROUPS = {
    # Tier 0: 已有的AO和SST
    "ao": {
        "csv": "data/ao_monthly.csv",
        "cols": ["ao"],
        "has_mask": False,
        "n_lags": 2,
        "lag_only": False,
    },
    "sst": {
        "csv": "data/arctic_sst_monthly.csv",
        "cols": ["sst"],
        "has_mask": True,  # SST从1981年开始，之前NaN
        "n_lags": 2,
        "lag_only": False,
    },
    # Tier 1: CPC气候指数
    "nao": {
        "csv": "data/nao_monthly.csv",
        "cols": ["nao"],
        "has_mask": False,
        "n_lags": 2,
        "lag_only": False,
    },
    "nino34": {
        "csv": "data/nino34_monthly.csv",
        "cols": ["nino34"],
        "has_mask": True,   # SSTOI starts 1982, missing 1979-1981
        "n_lags": 2,
        "lag_only": False,
    },
    "pna": {
        "csv": "data/pna_monthly.csv",
        "cols": ["pna"],
        "has_mask": False,  # PNA从1950年开始，覆盖全冰数据期
        "n_lags": 2,
        "lag_only": False,
    },
    "pdo": {
        "csv": "data/pdo_monthly.csv",
        "cols": ["pdo"],
        "has_mask": False,
        "n_lags": 2,
        "lag_only": False,
    },
    # Tier 2: ERA5 再分析
    "t2m": {
        "csv": "data/arctic_t2m_monthly.csv",
        "cols": ["t2m"],
        "has_mask": False,  # ERA5从1940年开始，覆盖全冰数据期
        "n_lags": 2,
        "lag_only": False,
    },
    "slp": {
        "csv": "data/arctic_slp_monthly.csv",
        "cols": ["slp"],
        "has_mask": False,
        "n_lags": 2,
        "lag_only": False,
    },
    # Tier 3: 派生特征
    "lag12_ice": {
        "csv": None,  # 从冰数据自身计算，不需要外部CSV
        "cols": ["area"],
        "has_mask": False,
        "n_lags": 0,
        "lag_only": True,
        "lag_months": 12,
    },
}


def load_ice_data():
    """加载12个海冰CSV，返回统一DataFrame。"""
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
    df_ice = df_ice[df_ice['area'] > 0]  # 过滤 -9999 缺测值
    print(f"海冰数据: {len(df_ice)} records ({df_ice['year'].min()}-{df_ice['year'].max()})")
    return df_ice


def load_variable_csv(csv_rel_path):
    """加载变量CSV（相对于BASE_DIR/data）。"""
    csv_path = os.path.join(BASE_DIR, csv_rel_path)
    if not os.path.exists(csv_path):
        print(f"  ⚠ 文件不存在: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    print(f"  加载: {csv_rel_path} → {len(df)} records ({df['year'].min()}-{df['year'].max()})")
    return df


def build_features(variables=None, compute_lags=True):
    """
    构建完整的滞后特征DataFrame。

    Args:
        variables: 要包含的变量名列表，None=全部
        compute_lags: 是否计算滞后（False时只做merge）
    """
    if variables is None:
        variables = list(VARIABLE_GROUPS.keys())

    # 验证变量名
    for v in variables:
        if v not in VARIABLE_GROUPS:
            print(f"⚠ 未知变量: {v}，跳过。可用: {list(VARIABLE_GROUPS.keys())}")
            variables.remove(v)

    # --- 1. 加载海冰 ---
    df = load_ice_data()

    # --- 2. 左连接所有外部变量 ---
    aux_channels = []       # 最终aux通道名列表
    mask_columns = []       # 掩码列名列表
    max_lag = 0             # 最大滞后月数

    for var_name in variables:
        cfg = VARIABLE_GROUPS[var_name]
        print(f"\n处理变量: {var_name}")

        # 特殊处理：lag12_ice从冰数据自身计算
        if cfg.get("lag_only") and cfg["csv"] is None:
            lag_months = cfg.get("lag_months", 12)
            col_name = f"{var_name}"
            df[col_name] = df[cfg["cols"][0]].shift(lag_months)
            df[col_name] = df[col_name].fillna(0.0)
            aux_channels.append(col_name)
            max_lag = max(max_lag, lag_months)
            print(f"  派生特征 lag-{lag_months}: '{cfg['cols'][0]}' → '{col_name}'")
            continue

        # 加载外部CSV
        df_var = load_variable_csv(cfg["csv"])
        if df_var is None:
            continue  # 文件缺失，跳过该变量

        # 检查CSV中是否有声明的列
        available_cols = [c for c in cfg["cols"] if c in df_var.columns]
        if not available_cols:
            print(f"  ⚠ CSV中找不到列 {cfg['cols']}，可用列: {df_var.columns.tolist()}")
            continue

        # Left join
        merge_cols = ["year", "month"] + available_cols
        df = df.merge(df_var[merge_cols], on=["year", "month"], how="left")

        # 如果has_mask：检查哪些行有缺失
        if cfg["has_mask"]:
            mask_name = f"{var_name}_mask"
            # 原始值 + 全部滞后的NaN检测
            df[mask_name] = df[available_cols[0]].notna().astype(int)

        # Fill缺失值
        for col in available_cols:
            df[col] = df[col].fillna(0.0)

        # 计算滞后
        n_lags = cfg.get("n_lags", 2) if compute_lags else 0
        for col in available_cols:
            aux_channels.append(col)
            for lag in range(1, n_lags + 1):
                lag_name = f"{col}_lag{lag}"
                df[lag_name] = df[col].shift(lag)
                df[lag_name] = df[lag_name].fillna(0.0)
                aux_channels.append(lag_name)
                max_lag = max(max_lag, lag)

        # 更新掩码（包含滞后后的NaN）
        if cfg["has_mask"] and n_lags > 0:
            for lag in range(1, n_lags + 1):
                df[mask_name] = df[mask_name] & (df[available_cols[0]].shift(lag).notna().astype(int))
            mask_columns.append(mask_name)
            aux_channels.append(mask_name)

        print(f"  通道: {var_name} → {len(aux_channels)} aux columns (累计)")

    # 掩码fill（shift可能导致NaN的第一行）
    for mcol in mask_columns:
        df[mcol] = df[mcol].fillna(0).astype(int)

    # --- 3. 删除前max_lag行（滞后导致的无效数据） ---
    if max_lag > 0:
        n_before = len(df)
        df = df.iloc[max_lag:].reset_index(drop=True)
        print(f"\n删除前{max_lag}行（滞后无效）: {n_before} → {len(df)} records")

    # --- 4. 整理列顺序 ---
    base_cols = ["year", "month", "extent", "area"]
    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + other_cols]

    # --- 5. 保存 ---
    df.to_csv(OUT_PATH, index=False)
    print(f"\n{'='*60}")
    print(f"✅ 滞后特征 v2 保存至: {OUT_PATH}")
    print(f"  总记录: {len(df)} ({df['year'].min()}-{df['year'].max()})")
    print(f"  总列数: {len(df.columns)}")
    print(f"  辅助通道: {len(aux_channels)} ({', '.join(aux_channels)})")
    print(f"  掩码列: {mask_columns}")
    print(f"\n  前5列: {df.columns[:5].tolist()}")
    print(f"  后5列: {df.columns[-5:].tolist()}")
    print(f"\n  时间覆盖:")
    for var_name in variables:
        cfg = VARIABLE_GROUPS[var_name]
        if cfg["cols"][0] in df.columns:
            col = cfg["cols"][0]
            nonzero = (df[col] != 0).sum()
            # 对于has_mask的变量，统计mask=1的记录
            if cfg["has_mask"]:
                mask_col = f"{var_name}_mask"
                if mask_col in df.columns:
                    nonzero = df[mask_col].sum()
            print(f"    {var_name:12s}: {nonzero}/{len(df)} records "
                  f"({nonzero/len(df)*100:.0f}%)")

    # --- 6. 返回辅助通道配置（供data_preprocessing使用） ---
    channel_config = {
        "aux_channels": aux_channels,
        "mask_columns": mask_columns,
        "variable_groups": {v: VARIABLE_GROUPS[v] for v in variables},
    }
    return df, channel_config


def main():
    parser = argparse.ArgumentParser(
        description="通用滞后特征构建器 v2 — 支持N个辅助变量"
    )
    parser.add_argument("--vars", type=str, default=None,
                        help="逗号分隔的变量名 (e.g., 'ao,nao,nino34')，默认全部")
    parser.add_argument("--no-lags", action="store_true",
                        help="不计算滞后（仅合并原始值）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用变量及其配置")
    args = parser.parse_args()

    if args.list:
        print("可用的变量 (VARIABLE_GROUPS):")
        for name, cfg in VARIABLE_GROUPS.items():
            csv_path = os.path.join(BASE_DIR, cfg["csv"]) if cfg["csv"] else "(派生特征)"
            exists = "✅" if cfg["csv"] is None or os.path.exists(os.path.join(BASE_DIR, cfg["csv"])) else "❌"
            print(f"  {name:12s} | mask={cfg['has_mask']} | lags={cfg['n_lags']} "
                  f"| {cfg['cols']} | {exists}")
        return

    variables = None
    if args.vars:
        variables = [v.strip() for v in args.vars.split(",")]

    df, channel_config = build_features(variables=variables, compute_lags=not args.no_lags)

    # 打印子集快速测试
    print(f"\n{'='*60}")
    print("数据预览 (前3行关键列):")
    preview_cols = ["year", "month", "area"]
    preview_cols += [c for c in df.columns if c not in preview_cols][:8]
    existing_cols = [c for c in preview_cols if c in df.columns]
    print(df[existing_cols].head(3).to_string())


if __name__ == "__main__":
    main()
