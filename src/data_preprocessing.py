# src/data_preprocessing.py
import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler


def load_and_merge_data(data_dir, target_column="extent", start_year=None, end_year=None):
    """
    读取12个月的数据文件并合并为完整时间序列
    """
    # 获取所有月份文件
    file_pattern = os.path.join(data_dir, "N_*_extent_v4.0.csv")
    files = sorted(glob.glob(file_pattern))

    print(f"找到 {len(files)} 个文件")

    df_list = []
    for file in files:
        # 从文件名提取月份
        filename = os.path.basename(file)
        month = int(filename.split('_')[1])

        # 读取CSV，去除列名中的空格
        df = pd.read_csv(file)

        # 去除列名中的空格
        df.columns = df.columns.str.strip()

        df['month'] = month
        df_list.append(df)

    # 合并所有月份
    df_all = pd.concat(df_list, ignore_index=True)

    # 按年月排序
    df_all = df_all.sort_values(['year', 'month']).reset_index(drop=True)

    # 去除列名中的空格（再次确保）
    df_all.columns = df_all.columns.str.strip()

    # 确保目标列存在
    if target_column not in df_all.columns:
        print(f"可用的列名: {df_all.columns.tolist()}")
        raise KeyError(f"列 '{target_column}' 不存在！请检查上面的可用列名")

    # 剔除缺失值（-9999 表示缺失）
    df_all = df_all[df_all[target_column] > 0]

    # 创建日期列
    df_all['date'] = pd.to_datetime(df_all[['year', 'month']].assign(day=1))

    # 筛选年份范围
    if start_year:
        df_all = df_all[df_all['year'] >= start_year]
    if end_year:
        df_all = df_all[df_all['year'] <= end_year]

    # 提取目标序列
    values = df_all[target_column].values.reshape(-1, 1)

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values)

    # 添加归一化后的列
    df_all[f'{target_column}_scaled'] = values_scaled

    print(f"数据加载完成：{len(df_all)} 条记录")
    print(f"时间范围：{df_all['date'].min()} 到 {df_all['date'].max()}")
    print(f"Target: {target_column}, Range: {values.min():.2f} ~ {values.max():.2f} million km2")

    return df_all, scaler


def create_sequences(data, input_len=12, output_len=12, target_data=None):
    """
    创建输入输出序列。

    Args:
        data: 特征数组 (total_len,) 或 (total_len, n_features)
        input_len: 输入序列长度
        output_len: 输出序列长度
        target_data: 目标数组（可选，仅用于y）。如果为None，则使用data作为y。

    Returns:
        X: (n_samples, input_len, n_features)
        y: (n_samples, output_len)
    """
    if target_data is None:
        target_data = data

    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:i + input_len])
        y.append(target_data[i + input_len:i + input_len + output_len])

    X = np.array(X)
    if X.ndim == 2:
        X = X.reshape(-1, input_len, 1)
    y = np.array(y)

    print(f"序列创建完成：{len(X)} 个样本")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y


def train_val_test_split_by_year(df, X, y, target_column="area"):
    """
    按年份划分训练/验证/测试集
    训练集：1979-2010
    验证集：2011-2015
    测试集：2016-2025
    """
    # 获取对应年份（考虑新的序列长度）
    input_len = X.shape[1]  # 动态获取输入长度
    output_len = y.shape[1]  # 从 y 中动态获取输出长度
    years = df['year'].values[input_len:len(df)-output_len+1]  # 考虑序列长度

    # 创建掩码
    train_mask = (years >= 1979) & (years <= 2010)
    val_mask = (years >= 2011) & (years <= 2015)
    test_mask = (years >= 2016) & (years <= 2025)

    # 划分数据
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"按年份数据划分完成：")
    print(f"  训练集: {len(X_train)} 样本 (年份: {years[train_mask].min()}-{years[train_mask].max()})")
    print(f"  验证集: {len(X_val)} 样本 (年份: {years[val_mask].min()}-{years[val_mask].max()})")
    print(f"  测试集: {len(X_test)} 样本 (年份: {years[test_mask].min()}-{years[test_mask].max()})")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    按时间顺序划分训练/验证/测试集（旧版本，保持兼容性）
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"数据划分完成：")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_multivariate_data(data_dir, ao_csv_path, target_column="area",
                           start_year=None, end_year=None):
    """
    Load sea ice data and AO index, align by year/month,
    normalize each feature independently.

    Args:
        data_dir: path to directory containing N_*_extent_v4.0.csv files
        ao_csv_path: path to ao_monthly.csv (year, month, ao)
        target_column: "area" or "extent"
        start_year, end_year: optional year filters

    Returns:
        df: merged DataFrame with columns including
            {target_column}, {target_column}_scaled, ao, ao_scaled
        scaler_ice: MinMaxScaler fitted on sea ice values
        scaler_ao: MinMaxScaler fitted on AO values
    """
    # 1. Load sea ice data (reuse existing merge logic)
    file_pattern = os.path.join(data_dir, "N_*_extent_v4.0.csv")
    files = sorted(glob.glob(file_pattern))
    print(f"找到 {len(files)} 个海冰数据文件")

    df_list = []
    for file in files:
        filename = os.path.basename(file)
        month = int(filename.split('_')[1])
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df['month'] = month
        df_list.append(df)

    df_ice = pd.concat(df_list, ignore_index=True)
    df_ice = df_ice.sort_values(['year', 'month']).reset_index(drop=True)
    df_ice.columns = df_ice.columns.str.strip()

    if target_column not in df_ice.columns:
        raise KeyError(f"列 '{target_column}' 不存在！可用列: {df_ice.columns.tolist()}")

    df_ice = df_ice[df_ice[target_column] > 0]  # drop -9999 markers

    # 2. Load AO data
    df_ao = pd.read_csv(ao_csv_path)
    print(f"找到 {len(df_ao)} 条 AO 数据记录")

    # 3. Merge on year + month (inner join — only years present in both)
    df = df_ice.merge(df_ao, on=['year', 'month'], how='inner')
    print(f"合并后: {len(df)} 条记录 ({df['year'].min()}-{df['year'].max()})")

    # 4. Year filter
    if start_year:
        df = df[df['year'] >= start_year]
    if end_year:
        df = df[df['year'] <= end_year]

    # 5. Create date column
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    # 6. Independent normalization: sea ice
    ice_values = df[target_column].values.reshape(-1, 1)
    scaler_ice = MinMaxScaler(feature_range=(0, 1))
    df[f'{target_column}_scaled'] = scaler_ice.fit_transform(ice_values)

    # 7. Independent normalization: AO
    ao_values = df['ao'].values.reshape(-1, 1)
    scaler_ao = MinMaxScaler(feature_range=(0, 1))
    df['ao_scaled'] = scaler_ao.fit_transform(ao_values)

    print(f"海冰 {target_column}: {ice_values.min():.2f} ~ {ice_values.max():.2f} million km²")
    print(f"AO 指数: {ao_values.min():.2f} ~ {ao_values.max():.2f}")

    return df, scaler_ice, scaler_ao


def load_trivariate_data(data_dir, ao_csv_path, sst_csv_path, target_column="area",
                          start_year=None, end_year=None):
    """
    Load sea ice, AO index, and Arctic SST; align by year/month;
    normalize each feature independently.

    Args:
        data_dir: directory with N_*_extent_v4.0.csv files
        ao_csv_path: path to ao_monthly.csv
        sst_csv_path: path to arctic_sst_monthly.csv
        target_column: "area" or "extent"
        start_year, end_year: optional year filters

    Returns:
        df: merged DataFrame with {target_column}_scaled, ao_scaled, sst_scaled
        scaler_ice, scaler_ao, scaler_sst: MinMaxScalers
    """
    # 1. Load sea ice data
    file_pattern = os.path.join(data_dir, "N_*_extent_v4.0.csv")
    files = sorted(glob.glob(file_pattern))
    print(f"Found {len(files)} sea ice files")

    df_list = []
    for file in files:
        filename = os.path.basename(file)
        month = int(filename.split('_')[1])
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df['month'] = month
        df_list.append(df)

    df_ice = pd.concat(df_list, ignore_index=True)
    df_ice = df_ice.sort_values(['year', 'month']).reset_index(drop=True)
    df_ice.columns = df_ice.columns.str.strip()

    if target_column not in df_ice.columns:
        raise KeyError(f"Column '{target_column}' not in ice data: {df_ice.columns.tolist()}")

    df_ice = df_ice[df_ice[target_column] > 0]

    # 2. Load AO
    df_ao = pd.read_csv(ao_csv_path)
    print(f"Found {len(df_ao)} AO records")

    # 3. Load SST
    df_sst = pd.read_csv(sst_csv_path)
    print(f"Found {len(df_sst)} SST records")

    # 4. Three-way inner join
    df = df_ice.merge(df_ao, on=['year', 'month'], how='inner')
    df = df.merge(df_sst, on=['year', 'month'], how='inner')
    print(f"Merged: {len(df)} records ({df['year'].min()}-{df['year'].max()})")

    # 5. Year filter
    if start_year:
        df = df[df['year'] >= start_year]
    if end_year:
        df = df[df['year'] <= end_year]

    # 6. Date column
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    # 7. Independent normalization of all 3 variables
    ice_values = df[target_column].values.reshape(-1, 1)
    scaler_ice = MinMaxScaler(feature_range=(0, 1))
    df[f'{target_column}_scaled'] = scaler_ice.fit_transform(ice_values)

    ao_values = df['ao'].values.reshape(-1, 1)
    scaler_ao = MinMaxScaler(feature_range=(0, 1))
    df['ao_scaled'] = scaler_ao.fit_transform(ao_values)

    sst_values = df['sst'].values.reshape(-1, 1)
    scaler_sst = MinMaxScaler(feature_range=(0, 1))
    df['sst_scaled'] = scaler_sst.fit_transform(sst_values)

    print(f"Sea ice {target_column}: {ice_values.min():.2f} ~ {ice_values.max():.2f} M km²")
    print(f"AO index: {ao_values.min():.2f} ~ {ao_values.max():.2f}")
    print(f"Arctic SST: {sst_values.min():.2f} ~ {sst_values.max():.2f} °C")

    return df, scaler_ice, scaler_ao, scaler_sst


def load_dual_encoder_data(lagged_csv, target_column="area",
                            start_year=None, end_year=None, aux_seq_len=3):
    """
    Load data for SeaIceDualEncoderLSTM.

    Reads lagged_features.csv (built by build_lagged_features.py).
    Normalizes area, AO features, and SST features independently.
    Creates main sequences (12-month sea ice) and aux sequences (last N
    months of lagged AO/SST/SST_mask).

    Args:
        lagged_csv: path to lagged_features.csv
        target_column: "area" or "extent"
        start_year, end_year: optional year filters
        aux_seq_len: number of months for aux encoder context (default 3)

    Returns:
        X_main: (n_samples, 12, 1) — sea ice
        X_aux:  (n_samples, aux_seq_len, 7) — ao, ao_l1, ao_l2, sst, sst_l1, sst_l2, mask
        y:      (n_samples, output_len) — target
        scaler_ice: MinMaxScaler for sea ice
        df: DataFrame (for year-based split)
    """
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv(lagged_csv)
    print(f"Loaded lagged features: {len(df)} records ({df['year'].min()}-{df['year'].max()})")

    # Year filter
    if start_year:
        df = df[df['year'] >= start_year]
    if end_year:
        df = df[df['year'] <= end_year]

    # Independent normalization
    ice_values = df[target_column].values.reshape(-1, 1)
    scaler_ice = MinMaxScaler(feature_range=(0, 1))
    df['ice_scaled'] = scaler_ice.fit_transform(ice_values)

    # AO features: single scaler for (ao, ao_lag1, ao_lag2)
    ao_cols = ['ao', 'ao_lag1', 'ao_lag2']
    ao_values = df[ao_cols].values.reshape(-1, 1)
    scaler_ao = MinMaxScaler(feature_range=(0, 1))
    ao_scaled = scaler_ao.fit_transform(ao_values).reshape(-1, 3)
    for i, col in enumerate(ao_cols):
        df[f'{col}_scaled'] = ao_scaled[:, i]

    # SST features: single scaler for (sst, sst_lag1, sst_lag2) — fit on non-zero only
    sst_cols = ['sst', 'sst_lag1', 'sst_lag2']
    sst_values = df[sst_cols].values.reshape(-1, 1)
    scaler_sst = MinMaxScaler(feature_range=(0, 1))
    scaler_sst.fit(sst_values[sst_values > 0].reshape(-1, 1))  # fit only valid SST
    sst_scaled = scaler_sst.transform(sst_values).reshape(-1, 3)
    for i, col in enumerate(sst_cols):
        df[f'{col}_scaled'] = sst_scaled[:, i]

    # Build aux features array: 7 channels
    aux_features = df[[
        'ao_scaled', 'ao_lag1_scaled', 'ao_lag2_scaled',
        'sst_scaled', 'sst_lag1_scaled', 'sst_lag2_scaled',
        'sst_mask',
    ]].values  # (N, 7)

    # Build sequences
    area_data = df['ice_scaled'].values
    n = len(area_data)
    input_len = 12

    X_main_list, X_aux_list, y_list = [], [], []
    for i in range(n - input_len - aux_seq_len + 1):
        # Main: 12 months of sea ice
        X_main_list.append(area_data[i:i + input_len])
        # Aux: last aux_seq_len months of lagged features
        X_aux_list.append(aux_features[i + input_len - aux_seq_len:i + input_len])
        # y: placeholder — output_len set at call site via separate function
        y_list.append(area_data[i + input_len])  # placeholder

    X_main = np.array(X_main_list).reshape(-1, input_len, 1)
    X_aux = np.array(X_aux_list)  # (samples, aux_seq_len, 7)

    print(f"Dual-encoder sequences: {len(X_main)} samples")
    print(f"  X_main: {X_main.shape}, X_aux: {X_aux.shape}")
    print(f"  SST mask coverage: {df['sst_mask'].sum()}/{len(df)} records")

    return X_main, X_aux, df, scaler_ice


def create_dual_targets(df, input_len, output_len, aux_seq_len, target_column="area"):
    """
    Create target sequences and year-aligned DataFrame for dual-encoder data.

    Must be called after load_dual_encoder_data().
    Returns y and a year array for split masking.
    """
    area_data = df['ice_scaled'].values
    years = df['year'].values
    n = len(area_data)
    total_len = input_len + aux_seq_len  # effective window

    y_list, year_list = [], []
    for i in range(n - total_len - output_len + 1):
        y_list.append(area_data[i + input_len:i + input_len + output_len])
        year_list.append(years[i + input_len])

    y = np.array(y_list)
    years_out = np.array(year_list)
    return y, years_out


# Test code
if __name__ == "__main__":
    # 使用你的实际路径
    data_path = r"C:\Users\86152\PycharmProjects\arctic_seaice_prediction lstm SIE\date\raw"
    df, scaler = load_and_merge_data(data_path, "extent")
    print("\n前10行数据：")
    print(df[['year', 'month', 'extent', 'area', 'date']].head(10))