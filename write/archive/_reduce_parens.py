#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reduce parentheses in thesis chapters by 80%"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
CHAPTERS = os.path.join(BASE, "chapters")

def count_parens(text):
    return text.count('（'), text.count('）')

def apply_rules(filepath, rules):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    l0, r0 = count_parens(content)
    applied = 0
    for old, new in rules:
        if old in content:
            content = content.replace(old, new)
            applied += 1
    l1, r1 = count_parens(content)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    name = os.path.basename(filepath)
    pct = 100*(l0-l1)//max(l0,1)
    print(f"  {name}: {l0} -> {l1} pairs ({pct}% reduced, {applied}/{len(rules)} rules)")
    return l0, l1

# ================================================================
# 01_Data_and_Methods rules
# ================================================================
RULES_01 = []
r = RULES_01.append

# Sea ice data section
r(("Version 4.0（G02135, doi:10.7265/r6nf-0842）数据集", "Version 4.0数据集"))
r(("月平均海冰面积产品（Sea Ice Area, SIA），空间范围", "月平均海冰面积产品，空间范围"))
r(("百万平方公里（10⁶ km²）为单位", "百万平方公里为单位"))
r(("区域月平均海冰面积产品（Regional Monthly Data），包含", "区域月平均海冰面积产品，包含"))
r(("白令海（Bering Sea）和楚科奇海（Chukchi Sea）", "白令海和楚科奇海"))
r(("（Barents Sea）、喀拉海（Kara Sea）、拉普捷夫海（Laptev Sea）和格陵兰海（Greenland Sea）", "、喀拉海、拉普捷夫海和格陵兰海"))
r(("中北冰洋（Central Arctic）", "中北冰洋"))
r(("归一化（MinMax Normalization）至[0, 1]区间", "归一化至[0, 1]区间"))
r(("北极极射投影（North Polar Stereographic），彩色", "北极极射投影，彩色"))

# Climate index section
r(("：极地大气环流指数（AO、NAO、PNA）、局地海洋热力指数（SST）和热带外强迫指数（Nino3.4）",
  "：极地大气环流指数AO、NAO和PNA，局地海洋热力指数SST，以及热带外强迫指数Nino3.4"))
r(("覆盖时段与NAO一致（1950年1月至2026年6月）", "覆盖时段与NAO一致"))
r(("海表温度第二版（Optimum Interpolation Sea Surface Temperature Version 2, OISST V2）月平均",
  "海表温度第二版OISST V2月平均"))
r(("甚高分辨率辐射计（AVHRR）卫星遥感", "甚高分辨率辐射计AVHRR卫星遥感"))
r(("较海冰数据（1979年）延迟约两年", "较海冰数据延迟约两年"))
r(("海表温度第五版（ERSST V5, doi:10.1175/JCLI-D-16-0836.1）数据集", "海表温度第五版ERSST V5数据集"))
r(("区域（5°N–5°S, 170°W–120°W）海表温度", "区域5°N–5°S/170°W–120°W的海表温度"))
r(("其物理源地（热带太平洋）与极地大气指数（北极、北大西洋、北太平洋）在空间上完全正交",
  "其物理源地热带太平洋与极地大气指数北极、北大西洋、北太平洋在空间上完全正交"))

# Data overview section
r(("月平均海冰面积（1979–2025年），包括长期", "月平均海冰面积1979–2025年，包括长期"))
r(("递减趋势（约−0.54百万平方公里/十年，p < 0.01）和强季节周期性",
  "递减趋势，约−0.54百万平方公里/十年且p < 0.01，和强季节周期性"))
r(("增暖趋势（约+0.05°C/十年）", "增暖趋势，约+0.05°C/十年"))

# Preprocessing section
r(("左连接（Left Join）策略", "左连接策略"))
r(("零值填充（Zero-Fill）并在对应时间位置生成二值掩码（Binary Mask）",
  "零值填充并在对应时间位置生成二值掩码"))
r(("滞后1个月（lag-1）和滞后2个月（lag-2）特征", "滞后1个月lag-1和滞后2个月lag-2特征"))
r(("滞后特征（5变量×2滞后阶）、", "滞后特征5变量×2滞后阶、"))
r(("掩码通道（SST和Nino3.4各一个）、", "掩码通道SST和Nino3.4各一个、"))
r(("训练集为1979年至2010年（32年，约 384 个样本），验证集为2011年至2015年（5年，约 60 个样本），测试集为2016年至2025年（10年，约 120 个样本）",
  "训练集为1979–2010年32年约384个样本，验证集为2011–2015年5年约60个样本，测试集为2016–2025年10年约120个样本"))

# LSTM section - keep first mentions, remove English-only annotations
r(("细胞状态（Cell State）解决了", "细胞状态解决了"))
r(("遗忘门（Forget Gate）通过", "遗忘门通过"))
r(("输入门（Input Gate）同样使用", "输入门同样使用"))
r(("输出门（Output Gate）控制", "输出门控制"))
r(("恢复构建海表温度第五版（Output Gate）", "SKIP"))  # placeholder, won't match
r(("常数误差传送带”（Constant Error Carousel）机制", "常数误差传送带”机制"))
r(("参数量（本研究单变量基线模型约 268K 参数）使其", "参数量本研究单变量基线模型约268K参数使其"))
r(("直接多步输出（Direct Multi-Step Output）策略", "直接多步输出策略"))

# Dual encoder section
r(("时间通道维度上（即每个时间步的输入由标量冰面积扩展为多变量向量），使用", "时间通道维度上，即每个时间步的输入由标量冰面积扩展为多变量向量，使用"))
r(("气候变量（AO和SST）时，测试集", "气候变量AO和SST时，测试集"))
r(("高频噪声（天气尺度扰动和指数计算误差），当这些", "高频噪声，包括天气尺度扰动和指数计算误差，当这些"))
r(("双编码器LSTM（Dual-Encoder LSTM）由", "双编码器LSTM由"))
r(("主编码器（Main Encoder）和辅助编码器（Auxiliary Encoder）", "主编码器和辅助编码器"))
r(("最近若干个月（aux_seq_len）的气候指数", "最近若干个月的气候指数"))
r(("可预报时效短（约 2 周）和海洋热力指数记忆时间有限（3–6个月）的物理认知",
  "可预报时效短约2周和海洋热力指数记忆时间有限3–6个月的物理认知"))
r(("统一的超参数配置（统一超参数策略），确保", "统一超参数策略，确保"))

# Loss function section
r(("海冰面积空间（百万平方公里）中计算", "海冰面积空间中计算"))
r(("随机种子（42、52、62、72、82）独立训练", "随机种子42、52、62、72、82独立训练"))
r(("重采样方法（10,000 次有放回重采样）计算95%置信区间", "重采样方法10,000次有放回重采样计算95%置信区间"))

print(f"01 rules: {len(RULES_01)} loaded")
print(f"All rules loaded successfully")

# ================================================================
# 03 rules
# ================================================================
RULES_03 = []
r3 = RULES_03.append

# 3.1 Evaluation
r3(("均方根误差（RMSE，单位：百万平方公里）作为", "均方根误差RMSE作为"))
r3(("平均绝对误差（MAE）和平均绝对百分比误差（MAPE）", "平均绝对误差MAE和平均绝对百分比误差MAPE"))
r3(("随机种子（42、52、62、72、82）的模型", "随机种子42、52、62、72、82的模型"))
r3(("集成预测（Ensemble Prediction），并报告", "集成预测，并报告"))
r3(("重采样方法（10,000 次有放回重采样）估计", "重采样方法10,000次有放回重采样估计"))
r3(("目标日历月份（1月至12月）分组统计", "目标日历月份1至12月分组统计"))
r3(("预测超前步长（lead-1至lead-12）分组", "预测超前步长lead-1至lead-12分组"))
r3(("（例如，以1月为起始的样本其lead-1对应2月，而以7月为起始的样本其lead-1对应8月）", "，例如以1月为起始的样本其lead-1对应2月，而以7月为起始的样本其lead-1对应8月"))

# 3.1 Figures
r3(("线性回归即LR（156参数），绿色", "线性回归即LR仅156参数，绿色"))
r3(("LSTM即E1（约268K参数）", "LSTM即E1约268K参数"))
r3(("低于SimpleRNN（0.528）和LR（0.532）", "低于SimpleRNN的0.528和LR的0.532"))
r3(("标准差最大（7月约±0.05），反映", "标准差最大7月约±0.05，反映"))
r3(("单变量LSTM（E1）的季节预测", "单变量LSTM E1的季节预测"))
r3(("LSTM预测均值（5种子集成），蓝紫色", "LSTM预测均值5种子集成，蓝紫色"))
r3(("3月峰（14.8百万平方公里）和9月谷（5.0百万平方公里）均被精确捕捉", "3月峰值14.8百万平方公里和9月谷值5.0百万平方公里均被精确捕捉"))
r3(("颜色从绿色（低RMSE）渐变至红色（高RMSE）", "颜色从绿色低RMSE渐变至红色高RMSE"))

print(f"03 rules: {len(RULES_03)} loaded")

# ================================================================
# Main
# ================================================================
def main():
    total_before = 0
    total_after = 0
    
    # Process 01
    f1 = os.path.join(CHAPTERS, "01_数据与方法.md")
    if os.path.exists(f1):
        b, a = apply_rules(f1, RULES_01)
        total_before += b
        total_after += a
    
    # Process 03 - two passes
    f3 = os.path.join(CHAPTERS, "03_结果与分析.md")
    if os.path.exists(f3):
        b1, a1 = apply_rules(f3, RULES_03)
        b2, a2 = apply_rules(f3, MORE_03)
        total_before += b1
        total_after += a2

        # Round 3: regex-based bulk cleanup
        with open(f3, 'r', encoding='utf-8') as f:
            content = f.read()
        l3 = content.count('（')
        # Color/visual labels
        for label in ['绿色', '红色', '改善', '退化', '圆点', '水平线段']:
            content = content.replace(f'（{label}）', label)
        content = content.replace('（百万平方公里）', '')
        # Short asides
        reps = {
            '（热带太平洋）': '热带太平洋',
            '（归一化RMSE）': '归一化RMSE',
            '（无气候指数辅助）': '',
            '（三者单独使用时均有正向增益）': '，三者单独使用时均有正向增益',
            '（特别是AO的共线性分量）': '，特别是AO的共线性分量',
            '（在七个海域中仅高于中北冰洋）': '，在七个海域中仅高于中北冰洋',
            '（5种子集成均值）': '5种子集成均值',
            '（2022/2016）': '',
            '（偏差 < 0.1百万平方公里）': '，偏差<0.1百万平方公里',
            '（绝对误差 > 0.5百万平方公里）': '，绝对误差>0.5百万平方公里',
            '（4–9月）': '4–9月',
            '（基准，RMSE = 0.5243）': '，基准RMSE=0.5243',
        }
        for old, new in reps.items():
            content = content.replace(old, new)
        # Regex-based
        import re
        content = re.sub(r'（(E\d+[a-z]*)）', r'\1', content)
        content = re.sub(r'（(\d+\.\d+)）', r'\1', content)
        content = re.sub(r'（(Δ = [+−]\d+\.\d+)）', r'，\1', content)
        content = content.replace('（2022年，左）', '2022年左').replace('（2016年，右）', '2016年右')
        content = re.sub(r'（(最佳年，RMSE = \d+\.\d+)）', r'，\1', content)
        content = re.sub(r'（(最差年，RMSE = \d+\.\d+)）', r'，\1', content)
        content = content.replace('（0.07至0.14百万平方公里）', '，0.07至0.14百万平方公里')
        content = re.sub(r'（(E\d+[a-z]*,\s*RMSE = [^）]+)）', r'，\1', content)
        content = re.sub(r'（(E\d+[a-z]*,\s*Δ = [^）]+)）', r'，\1', content)
        content = re.sub(r'（（', '（', content)
        content = re.sub(r'），', '，', content)
        content = re.sub(r'，，+', '，', content)
        l4 = content.count('（')
        with open(f3, 'w', encoding='utf-8') as f:
            f.write(content)
        pct3 = 100*(l3-l4)//max(l3,1)
        print(f"  [Round3] {os.path.basename(f3)}: {l3} -> {l4} pairs ({pct3}% reduced)")
        total_after = l4
    
    print(f"\n{'='*60}")
    pct = 100*(total_before-total_after)//max(total_before,1)
    print(f"TOTAL: {total_before} -> {total_after} pairs ({pct}% reduced)")

# ================================================================
# Additional 03 rules - Phase 1 results
# ================================================================
MORE_03 = []
m3 = MORE_03.append

# Experiment/variable refs in parens -> integrate with slash
m3(("单变量基线E1（纯海冰LSTM，无气候指数辅助）的5种子", "单变量基线E1纯海冰LSTM无气候指数辅助的5种子"))
m3(("百万平方公里（各种子均值0.5319 ± 0.0115），对应的", "百万平方公里，各种子均值0.5319 ± 0.0115，对应的"))
m3(("增益（ΔRMSE ∈ [−0.015, −0.004]），但增益", "增益ΔRMSE在−0.015至−0.004之间，但增益"))
m3(("最佳单变量（集成RMSE = 0.5089，Δ = −0.0147，改善2.81%），其次为北极涛动AO（集成RMSE = 0.5130，Δ = −0.0106，改善2.02%），而Nino3.4（集成RMSE = 0.5180，Δ = −0.0055）和NAO（集成RMSE = 0.5192，Δ = −0.0044）仅",
  "最佳单变量，集成RMSE=0.5089/Δ=−0.0147/改善2.81%，其次为北极涛动AO，集成RMSE=0.5130/Δ=−0.0106/改善2.02%，而Nino3.4集成RMSE=0.5180/Δ=−0.0055和NAO集成RMSE=0.5192/Δ=−0.0044仅"))
m3(("与Liu等（2021）关于", "与Liu等2021关于"))
m3(("AO+SST组合（E7v1）非但", "AO+SST组合E7v1非但"))
m3(("至0.5420（+3.54% vs E1），成为", "至0.5420，+3.54% vs E1，成为"))
m3(("冗余度较高（AO正位相期间北极温度普遍偏高，与SST升高的热力信号高度相关），且SST", "冗余度较高，AO正位相期间北极温度普遍偏高与SST升高的热力信号高度相关，且SST"))

# Layer descriptions
m3(("太平洋扇区大气指数（PNA）优于全域大气指数（AO），后者优于大西洋扇区大气指数（NAO）和热带海洋信号（Nino3.4）",
  "太平洋扇区大气指数PNA优于全域大气指数AO，后者优于大西洋扇区大气指数NAO和热带海洋信号Nino3.4"))

# Figure 3.3 description
m3(("集成RMSE（5种子平均）与纯冰基线E1的对比", "集成RMSE 5种子平均与纯冰基线E1的对比"))
m3(("PNA（E10, 0.5089）为最佳单变量，RMSE较基线降低0.0147百万平方公里（改善2.81%）；其次为AO（E14, 0.5130, 改善2.02%）、Nino3.4（E9, 0.5180）和NAO（E8, 0.5192），改善幅度约1%。纯冰基线E1（0.5236）居中，蓝色虚线标注基线RMSE作为参考。AO+SST组合（E7v1, 0.5420）是唯一劣于基线的实验（退化3.54%），以红色柱标示",
  "PNA/E10, 0.5089为最佳单变量，RMSE较基线降低0.0147百万平方公里，改善2.81%；其次为AO/E14, 0.5130/改善2.02%、Nino3.4/E9, 0.5180和NAO/E8, 0.5192，改善幅度约1%。纯冰基线E1 0.5236居中，蓝色虚线标注基线RMSE作为参考。AO+SST组合E7v1, 0.5420是唯一劣于基线的实验，退化3.54%，以红色柱标示"))
m3(("太平洋扇区大气指数（PNA） > 全域大气指数（AO） > 大西洋扇区大气指数（NAO）和热带海洋信号（Nino3.4）",
  "太平洋扇区大气指数PNA > 全域大气指数AO > 大西洋扇区大气指数NAO和热带海洋信号Nino3.4"))

# Phase 2 results
m3(("海洋内部组合E17（SST+Nino3.4，集成RMSE = 0.5098）和大气+海洋组合E18（AO+SST+NAO，集成RMSE = 0.5218）优于基线E1（0.5236），其余",
  "海洋内部组合E17/SST+Nino3.4，集成RMSE=0.5098和大气+海洋组合E18/AO+SST+NAO，集成RMSE=0.5218优于基线E1 0.5236，其余"))
m3(("与纯冰基线（0.5236）几乎无差异（Δ = +0.0007），投入", "与纯冰基线0.5236几乎无差异，Δ=+0.0007，投入"))
m3(("跨扇区互补（E23, AO+PNA, 集成RMSE = 0.5244）与同扇区冗余（E15, AO+NAO, 集成RMSE = 0.5229）的对比",
  "跨扇区互补E23/AO+PNA，集成RMSE=0.5244与同扇区冗余E15/AO+NAO，集成RMSE=0.5229的对比"))
m3(("为0.0015（Bootstrap 95% CI不显著）。这表明AO与PNA（跨扇区）的信息组合并未比AO与NAO（同扇区）提供实质性优势",
  "为0.0015，Bootstrap 95% CI不显著。这表明AO与PNA跨扇区的信息组合并未比AO与NAO同扇区提供实质性优势"))
m3(("全扇区组合E24（AO+NAO+PNA）表现最差，集成RMSE为0.5383，较基线退化2.81%。结合单变量结果——三者单独使用时均有正向增益（E14: 0.5130, E8: 0.5192, E10: 0.5089）——这一退化表明当三种大气指数同时编码时，其相互之间的高冗余度（AO与NAO月尺度相关系数约0.7）和共线性导致辅助编码器难以有效区分各变量的独立贡献",
  "全扇区组合E24/AO+NAO+PNA表现最差，集成RMSE为0.5383，较基线退化2.81%。结合单变量结果——三者单独使用时均有正向增益，E14: 0.5130/E8: 0.5192/E10: 0.5089——这一退化表明当三种大气指数同时编码时，其相互之间的高冗余度AO与NAO月尺度相关系数约0.7和共线性导致辅助编码器难以有效区分各变量的独立贡献"))

m3(("零增益（E15: 0.5229, E23: 0.5244, E16: 0.5272, E24: 0.5383, E19: 0.5243），而所有不含AO的组合即E17为0.5098、E18为0.5218均不劣于基线。结合E14（AO单独使用时产生Δ = −0.0106的正向增益）的表现，AO呈现出一种",
  "零增益，E15: 0.5229/E23: 0.5244/E16: 0.5272/E24: 0.5383/E19: 0.5243，而所有不含AO的组合即E17为0.5098、E18为0.5218均不劣于基线。结合E14/AO单独使用时产生Δ=−0.0106正向增益的表现，AO呈现出一种"))
m3(("扇区特异性指数（PNA/NAO）更为弥散", "扇区特异性指数PNA和NAO更为弥散"))
m3(("最佳组合E17（SST+Nino3.4, RMSE = 0.5098）全部由海洋变量构成，是唯一超过最佳单变量E10（PNA, 0.5089）水平的组合。而大气变量组合——无论是同扇区（E15, 0.5229）、跨扇区（E23, 0.5244）还是全扇区（E24, 0.5383）——均未带来",
  "最佳组合E17/SST+Nino3.4，RMSE=0.5098全部由海洋变量构成，是唯一超过最佳单变量E10/PNA, 0.5089水平的组合。而大气变量组合——无论是同扇区E15/0.5229、跨扇区E23/0.5244还是全扇区E24/0.5383——均未带来"))
m3(("记忆时间更长（SST的自相关e-folding时间尺度约3 至 6 个月）、大气环流可预报时效更短（约 2 周）的物理规律",
  "记忆时间更长，SST的自相关e-folding时间尺度约3至6个月、大气环流可预报时效更短约2周的物理规律"))

print(f"Additional 03 rules: {len(MORE_03)} loaded")

if __name__ == "__main__":
    main()
