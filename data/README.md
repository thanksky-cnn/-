# 数据文档

论文"基于LSTM的北极海冰面积预测研究"所用数据说明。

## 数据来源

| 变量 | 来源 | 时间范围 | 分辨率 | 下载链接 |
|------|------|----------|--------|----------|
| 海冰面积/范围 | NSIDC Sea Ice Index v4 | 1978-10 至今 | 月度 | https://climate.nasa.gov/vital-signs/arctic-sea-ice/ |
| 北极涛动 (AO) | NOAA CPC | 1950-01 至今 | 月度 | https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table |
| 海表温度 (SST) | NOAA OI SST V2 | 1981-12 至 2023-01 | 月度, 1°×1° 格点 | https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/sst.mnmean.nc |

## 数据文件说明

```
data/
├── raw/                           # 原始海冰数据
│   ├── N_01_extent_v4.0.csv      # 1月海冰面积/范围
│   ├── N_02_extent_v4.0.csv      # 2月
│   └── ... N_12_extent_v4.0.csv  # 12月
├── NOAA .CPC. AO.monthly.txt     # 原始AO文本数据
├── sst.mnmean.nc                 # 原始SST NetCDF (64MB)
├── ao_monthly.csv                # 处理后的AO (year, month, ao)
├── arctic_sst_monthly.csv        # 处理后的SST (year, month, sst)
└── lagged_features.csv           # 合并后的滞后特征 (11列)
```

## 数据处理流程

1. **AO解析** (`data_parse_ao.py`): 从NOAA文本文件解析 → `ao_monthly.csv`
2. **SST提取** (`data_extract_sst.py`): 从NetCDF提取北半球(30.98°N-90°N)空间平均 → `arctic_sst_monthly.csv`
3. **滞后特征构建** (`data_build_lagged.py`): 合并海冰+AO+SST，计算滞后特征(t, t-1, t-2)和SST掩码 → `lagged_features.csv`

## 训练/验证/测试划分

| 集合 | 年份 | 说明 |
|------|------|------|
| 训练集 | 1979-2010 | ~369个样本 |
| 验证集 | 2011-2015 | ~60个样本 |
| 测试集 | 2016-2025 | ~112个样本 |

## 已知数据局限性

1. **SST时间范围受限**: 仅覆盖1981-12至2023-01。测试集最后近3年(2023-2025)无SST数据，训练集前3年(1979-1981)也无SST数据
2. **SST空间平均粗糙**: 当前对30.98°N以北全部区域进行平均，未区分不同海区
3. **样本量极小**: ~369个训练样本对于深度学习模型偏少，限制了模型复杂度
4. **缺失值处理**: 3个缺失月份(1987-12, 1988-01, 1988-08)被直接丢弃；缺失SST值用0填充
5. **AO/SST滞后期**: 仅使用t, t-1, t-2三步滞后，可能不足以捕捉海洋过程的更长时间尺度影响

## 复现说明

要复现数据处理流程：

```bash
# 1. 解压原始AO文本
python data_parse_ao.py

# 2. 下载并提取SST
# 下载: curl -L -o data/sst.mnmean.nc "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/sst.mnmean.nc"
python data_extract_sst.py

# 3. 构建滞后特征
python data_build_lagged.py
```
