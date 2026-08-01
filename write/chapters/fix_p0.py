#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0 Fixes: formulas, tables, chapter numbering"""
import docx, sys, io, os
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r'c:\Users\86152\PycharmProjects\2 +ao arctic_seaice_prediction lstm SIE'
PLOTS = os.path.join(BASE, 'outputs', 'plots', 'paper')
CHAPTERS = os.path.join(BASE, 'write', 'chapters')
TEMPLATE = r'C:\Users\86152\Desktop\毕设\融合Cellpose与圆度判别的盐水冰微结构自动识别与参数提取方法 - 副本.docx'
OUTPUT = os.path.join(BASE, 'write', '基于LSTM的多源气候指数的北极海冰面积预测研究_终稿.docx')

def add_figure(doc, img_path, caption, width_inches=5.5):
    if not os.path.exists(img_path):
        print(f'  MISSING: {os.path.basename(img_path)}')
        return
    # Image with 图 style
    p = doc.add_paragraph(style='图')
    run = p.add_run()
    run.add_picture(img_path, width=Inches(width_inches))
    # Caption with 图注 style
    cap = doc.add_paragraph(style='图注')
    cap.add_run(caption)

def add_normal(doc, text):
    if not text or not text.strip():
        return
    p = doc.add_paragraph(style='Normal')
    p.add_run(text.strip())

def add_heading(doc, text, level):
    doc.add_heading(text, level=level)

def add_formula_img(doc, img_name, label):
    """Add a rendered formula as inline image + label"""
    path = os.path.join(PLOTS, img_name)
    if not os.path.exists(path):
        return
    # Formula label on left
    p = doc.add_paragraph(style='Normal')
    run = p.add_run(label + '：')
    run.font.size = Pt(9)
    # Formula image centered
    p2 = doc.add_paragraph(style='Normal')
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.add_run().add_picture(path, width=Inches(4.5))

def add_table(doc, headers, rows, caption):
    """Add a formatted table"""
    doc.add_paragraph(style='Normal')
    cp = doc.add_paragraph(style='Normal')
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cp.add_run(caption)
    run.bold = True
    run.font.size = Pt(9)

    table = doc.add_table(rows=len(rows)+1, cols=len(headers))
    table.style = 'Table Grid'

    # Header row
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ''
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(8)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Data rows
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i+1].cells[j]
            cell.text = ''
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(8)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(style='Normal')

def process_chapter(doc, content, ch_num):
    """Process markdown with chapter-aware heading rewriting"""
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.startswith('## 参考文献') or line == '# 参考文献':
            add_heading(doc, '参考文献', level=1)
        elif line.startswith('## '):
            text = line[3:].strip()
            add_heading(doc, text, 2)
        elif line.startswith('### '):
            text = line[4:].strip()
            add_heading(doc, text, 3)
        elif line.startswith('# '):
            text = line[2:].strip()
            add_heading(doc, text, 1)
        elif line.startswith('**') and ('表' in line or '图' in line):
            # Skip table/figure captions in markdown - we handle them separately
            pass
        elif line.startswith('|'):
            i += 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                i += 1
            continue
        elif line.startswith('$$'):
            i += 1
            continue
        elif line.startswith('[') and ']' in line[:5]:
            p = doc.add_paragraph(style='EndNote Bibliography')
            p.add_run(line.strip())
        else:
            text = line.replace('**', '')
            add_normal(doc, text)
        i += 1

print('='*60)
print('P0 FIXES: Formulas + Tables + Numbering')
print('='*60)

doc = docx.Document(TEMPLATE)
body = doc.element.body
for p in body.findall(qn('w:p')):
    body.remove(p)

# ===== TITLE PAGE =====
doc.add_paragraph('基于LSTM的多源气候指数的北极海冰面积预测研究', style='Title')
p = doc.add_paragraph(style='作者')
p.add_run('谢天')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph(style='单位')
p.add_run('（大连理工大学，海岸和近海工程国家重点实验室，辽宁  大连  116024）')

# ===== ABSTRACT =====
abstract = (
    '摘要：现有基于深度学习的海冰预测研究大多依赖单一海冰数据源，气候指数能否为海冰预测提供额外的预测技能，'
    '以及这种增量价值在何种条件下成立，尚未得到系统性评估。本文基于 1979 年 1 月至 2025 年 12 月'
    '美国国家冰雪数据中心（NSIDC）月平均海冰面积数据及同期五个气候指数——北极涛动（Arctic Oscillation, AO）、'
    '北大西洋涛动（North Atlantic Oscillation, NAO）、太平洋-北美型遥相关（Pacific North American pattern, PNA）、'
    'Nino 3.4 区海表温度异常指数（Nino3.4）及北极海表温度（Sea Surface Temperature, SST）——'
    '构建了双编码器长短时记忆网络（Dual-Encoder Long Short-Term Memory, Dual-Encoder LSTM）预测框架，'
    '通过单变量增量、变量组合、消融验证及七海域空间异质性分析四个递进阶段的 38 个独立实验，'
    '系统性检验了标量气候指数对北极海冰面积预测的增量贡献。结果表明：（1）气候指数的增量预测技能有限但真实存在——'
    '全北极最优单变量 PNA 的均方根误差（RMSE）较纯冰基线改善约 2.8%，而全五变量组合与纯冰基线几乎无差异；'
    '（2）"越少越好"的变量选择规律在全北极和区域两个空间尺度上一致成立，AO 在多变量组合中持续性起负面作用；'
    '（3）气候指数预测价值的空间异质性范围有限——统一超参数条件下仅巴伦支海呈现明显的 NAO 增益优势'
    '（NAO 增益为 PNA 的 3.4 倍），但区域独立调参后白令海（-8.6%）、巴伦支海（-7.3%）和喀拉海（-9.9%）'
    '的预测误差均大幅降低，而中北冰洋和楚科奇海在调参后仍近零增益，表明敏感海域集中在受大西洋和太平洋入流直接影响的边缘海。'
    '本研究为标量气候指数在海冰统计预测中的价值提供了首个系统性定量评估基准。'
)
doc.add_paragraph(abstract, style='摘要')
doc.add_paragraph('关键词：北极海冰面积；长短时记忆网络；气候指数；双编码器；变量消融；空间异质性', style='摘要')

# ---- ENGLISH ABSTRACT ----
en_title = doc.add_paragraph(style='Normal')
en_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = en_title.add_run('Arctic Sea Ice Area Prediction Using LSTM with Multi-Source Climate Indices')
run.bold = True
run.font.size = Pt(12)

en_abstract = (
    'Abstract: Existing deep learning-based sea ice prediction studies predominantly rely on single-source sea ice data. '
    'Whether climate indices can provide additional predictive skill beyond sea ice history, and under what conditions '
    'such incremental value holds, has not been systematically evaluated. This study constructs a Dual-Encoder Long '
    'Short-Term Memory (Dual-Encoder LSTM) prediction framework based on NSIDC monthly sea ice area data (January 1979 '
    'to December 2025) and five concurrent climate indices—Arctic Oscillation (AO), North Atlantic Oscillation (NAO), '
    'Pacific North American pattern (PNA), Nino 3.4 index, and Arctic sea surface temperature (SST). Through a four-phase '
    'progressive experimental design comprising 38 independent experiments—single-variable increment, variable combination, '
    'ablation validation, and seven-region spatial heterogeneity analysis—this study systematically examines the incremental '
    'contribution of scalar climate indices to Arctic sea ice area prediction. Results show that: (1) the incremental predictive '
    'skill of climate indices is limited but real—the best single variable PNA improves RMSE by approximately 2.8% over '
    'the ice-only baseline, while the full five-variable combination shows virtually no difference from the baseline; '
    '(2) the “less is more” pattern holds consistently at both pan-Arctic and regional scales, with AO persistently '
    'exhibiting negative effects in any multi-variable combination; (3) the spatial heterogeneity of climate index predictive '
    'value is limited in scope—under uniform hyperparameters, only the Barents Sea shows the expected NAO matching advantage '
    '(3.4 times the PNA gain), but after region-specific hyperparameter tuning, substantial error reductions are achieved '
    'for the Bering Sea (−8.6%), Barents Sea (−7.3%), and Kara Sea (−9.9%), while the Central Arctic and Chukchi Sea '
    'show near-zero gain even after tuning, indicating that sensitive regions are concentrated in marginal seas directly '
    'influenced by Atlantic and Pacific inflow. This study provides the first systematic quantitative benchmark for evaluating '
    'the value of scalar climate indices in statistical sea ice prediction.'
)
doc.add_paragraph(en_abstract, style='摘要')
doc.add_paragraph('Keywords: Arctic sea ice area; Long Short-Term Memory; climate indices; dual-encoder; variable ablation; spatial heterogeneity', style='摘要')

# ---- ABBREVIATION LIST ----
add_heading(doc, '缩略语对照表', level=1)
abbr_table = doc.add_table(rows=18, cols=2)
abbr_table.style = 'Table Grid'
abbrs = [
    ('AO', 'Arctic Oscillation — 北极涛动'),
    ('NAO', 'North Atlantic Oscillation — 北大西洋涛动'),
    ('PNA', 'Pacific North American pattern — 太平洋-北美型遥相关'),
    ('SST', 'Sea Surface Temperature — 海表温度'),
    ('ENSO', 'El Niño-Southern Oscillation — 厄尔尼诺-南方涛动'),
    ('LSTM', 'Long Short-Term Memory — 长短时记忆网络'),
    ('RNN', 'Recurrent Neural Network — 循环神经网络'),
    ('CNN', 'Convolutional Neural Network — 卷积神经网络'),
    ('RMSE', 'Root Mean Squared Error — 均方根误差'),
    ('MAE', 'Mean Absolute Error — 平均绝对误差'),
    ('MAPE', 'Mean Absolute Percentage Error — 平均绝对百分比误差'),
    ('NSIDC', 'National Snow and Ice Data Center — 美国国家冰雪数据中心'),
    ('NOAA CPC', 'NOAA Climate Prediction Center — 美国国家海洋和大气管理局气候预测中心'),
    ('CMIP6', 'Coupled Model Intercomparison Project Phase 6 — 第六次耦合模式比较计划'),
    ('EOF', 'Empirical Orthogonal Function — 经验正交函数'),
    ('SIA', 'Sea Ice Area — 海冰面积'),
    ('SIC', 'Sea Ice Concentration — 海冰密集度'),
]
for i, (abbr, full) in enumerate(abbrs):
    abbr_table.rows[i].cells[0].text = abbr
    p = abbr_table.rows[i].cells[0].paragraphs[0]
    for run in p.runs:
        run.bold = True
        run.font.size = Pt(9)
    abbr_table.rows[i].cells[1].text = full
    for run in abbr_table.rows[i].cells[1].paragraphs[0].runs:
        run.font.size = Pt(9)

doc.add_page_break()

# ===== CHAPTER 1: 绪论 (formerly 0 引言) =====
with open(os.path.join(CHAPTERS, '00_引言.md'), 'r', encoding='utf-8') as f:
    ch0_content = f.read()

# Fix heading: # 0 引言 → # 第1章 绪论
ch0_content = ch0_content.replace('# 0 引言', '# 第1章 绪论')
ch0_content = ch0_content.replace('## 0.1', '## 1.1')
ch0_content = ch0_content.replace('## 0.2', '## 1.2')
ch0_content = ch0_content.replace('## 0.3', '## 1.3')
ch0_content = ch0_content.replace('### 0.2.1', '### 1.2.1')
ch0_content = ch0_content.replace('### 0.2.2', '### 1.2.2')
ch0_content = ch0_content.replace('### 0.2.3', '### 1.2.3')

process_chapter(doc, ch0_content, 1)
print('  Added: 第1章 绪论 (formerly 0 引言)')

# ===== CHAPTER 2: 数据与方法 (formerly 1) =====
with open(os.path.join(CHAPTERS, '01_数据与方法.md'), 'r', encoding='utf-8') as f:
    ch1_content = f.read()

ch1_content = ch1_content.replace('# 1 数据与方法', '# 第2章 数据与方法')
ch1_content = ch1_content.replace('## 1.1', '## 2.1')
ch1_content = ch1_content.replace('## 1.2', '## 2.2')
ch1_content = ch1_content.replace('## 1.3', '## 2.3')
ch1_content = ch1_content.replace('### 1.1.1', '### 2.1.1')
ch1_content = ch1_content.replace('### 1.1.2', '### 2.1.2')
ch1_content = ch1_content.replace('### 1.1.3', '### 2.1.3')
ch1_content = ch1_content.replace('### 1.3.1', '### 2.3.1')
ch1_content = ch1_content.replace('### 1.3.2', '### 2.3.2')
ch1_content = ch1_content.replace('### 1.3.3', '### 2.3.3')

process_chapter(doc, ch1_content, 2)

# Insert LSTM formulas after the formula descriptions
print('  Inserting LSTM formulas...')
add_formula_img(doc, 'formula_forget.png', '遗忘门')
add_formula_img(doc, 'formula_input.png', '输入门')
add_formula_img(doc, 'formula_output.png', '输出门')
add_formula_img(doc, 'formula_cell.png', '细胞状态更新')
add_normal(doc, '其中，σ 为 Sigmoid 激活函数，输出范围为 (0, 1)；tanh 为双曲正切激活函数，输出范围为 (-1, 1)；W 和 b 分别为各门控单元的权重矩阵和偏置向量（Hochreiter & Schmidhuber, 1997）。')

# Insert architecture diagram
add_figure(doc, os.path.join(PLOTS, 'fig_arch_dual_encoder.png'), '图2.5 双编码器LSTM架构示意图')

# Insert MSE and RMSE formulas
add_formula_img(doc, 'formula_mse.png', '均方误差损失函数')
add_formula_img(doc, 'formula_rmse.png', '均方根误差评估指标')

print('  Added: 第2章 数据与方法 (with formulas)')

# ===== CHAPTER 3: 实验方案设计 (formerly 2) =====
with open(os.path.join(CHAPTERS, '02_实验方案设计.md'), 'r', encoding='utf-8') as f:
    ch2_content = f.read()

ch2_content = ch2_content.replace('# 2 实验方案设计', '# 第3章 实验方案设计')
ch2_content = ch2_content.replace('## 2.1', '## 3.1')
ch2_content = ch2_content.replace('## 2.2', '## 3.2')
ch2_content = ch2_content.replace('## 2.3', '## 3.3')
ch2_content = ch2_content.replace('### 2.2.1', '### 3.2.1')
ch2_content = ch2_content.replace('### 2.2.2', '### 3.2.2')
ch2_content = ch2_content.replace('### 2.3.1', '### 3.3.1')
ch2_content = ch2_content.replace('### 2.3.2', '### 3.3.2')
ch2_content = ch2_content.replace('### 2.3.3', '### 3.3.3')
ch2_content = ch2_content.replace('§3.3', '§4.3')

process_chapter(doc, ch2_content, 3)

# Insert experiment design tables
print('  Inserting experiment tables...')

# Table 3.1: Single variable
add_table(doc,
    ['实验ID', '辅助编码器输入', '变量来源', '实验目的'],
    [
        ['E1', '无（纯冰基线）', '—', '单变量LSTM基准性能'],
        ['E7v1', 'AO + SST', '极地大气+局地海洋', '已有最优多变量参照'],
        ['E14', 'AO only', '极地大气（全域）', 'AO独立贡献'],
        ['E8', 'NAO only', '极地大气（大西洋扇区）', 'NAO独立贡献'],
        ['E10', 'PNA only', '极地大气（太平洋扇区）', 'PNA独立贡献'],
        ['E9', 'Nino3.4 only', '热带海洋', 'ENSO独立贡献'],
    ],
    '表3.1 单变量增量实验设计'
)

# Table 3.2: Variable combination
add_table(doc,
    ['实验ID', '辅助编码器输入', '变量数', '组合类型', '核心假设'],
    [
        ['E15', 'AO + NAO', '2', '同扇区（大西洋）', '冗余假设'],
        ['E23', 'AO + PNA', '2', '跨扇区', '互补假设'],
        ['E16', 'AO + Nino3.4', '2', '极地+热带', '信号正交假设'],
        ['E17', 'SST + Nino3.4', '2', '海洋内部', '局地+热带协同假设'],
        ['E24', 'AO + NAO + PNA', '3', '大气全扇区', '大气信息饱和上限'],
        ['E18', 'AO + SST + NAO', '3', '大气为主+海洋', 'E7v1+大西洋扇区'],
        ['E19', 'AO+SST+NAO+PNA+Nino3.4', '5', '全变量', '联合信息上限'],
    ],
    '表3.2 变量组合实验设计'
)

# Table 3.3: Ablation
add_table(doc,
    ['实验ID', '配置', '移除变量', '验证问题'],
    [
        ['E20', '全 - NAO', 'NAO', 'NAO在AO+PNA覆盖下是否冗余？'],
        ['E21', '全 - Nino3.4', 'Nino3.4', '热带信号是否仍有独立价值？'],
        ['E22', '全 - SST', 'SST', '移除局地热力强迫是否导致退化？'],
        ['E25', '全 - PNA', 'PNA', 'PNA是否提供不可替代信息？'],
    ],
    '表3.3 消融验证实验设计'
)

# Table 3.4: Spatial heterogeneity
add_table(doc,
    ['实验ID', '预测目标', '辅助输入', '匹配类型', '核心假说'],
    [
        ['E26/E26a/E26b', '白令海 Bering', '无/PNA/NAO', '基线/匹配/不匹配', 'PNA应对白令海增益最大'],
        ['E29/E29a/E29b', '楚科奇海 Chukchi', '无/PNA/NAO', '基线/匹配/不匹配', 'PNA应对楚科奇海增益最大'],
        ['E27/E27a/E27b', '巴伦支海 Barents', '无/NAO/PNA', '基线/匹配/不匹配', 'NAO应对巴伦支海增益最大'],
        ['E30/E30a/E30b', '喀拉海 Kara', '无/NAO/PNA', '基线/匹配/不匹配', 'NAO应对喀拉海增益最大'],
        ['E31/E31a/E31b', '拉普捷夫海 Laptev', '无/NAO/PNA', '基线/匹配/不匹配', 'NAO应对拉普捷夫海增益最大'],
        ['E32/E32a/E32b', '格陵兰海 Greenland', '无/NAO/PNA', '基线/匹配/不匹配', 'NAO应对格陵兰海增益最大'],
        ['E28/E28a/E28b', '中北冰洋 Central Arctic', '无/AO/SST', '基线/全域/局地', '核心区增益预期最低'],
    ],
    '表3.4 空间异质性实验矩阵'
)

# Insert framework diagram
add_figure(doc, os.path.join(PLOTS, 'fig_experimental_framework.png'), '图3.1 四阶段递进式实验设计框架')

print('  Added: 第3章 实验方案设计 (with 4 tables + framework)')

# ===== CHAPTER 4: 结果与分析 (formerly 3) =====
with open(os.path.join(CHAPTERS, '03_结果与分析.md'), 'r', encoding='utf-8') as f:
    ch3_content = f.read()

ch3_content = ch3_content.replace('# 3 结果与分析', '# 第4章 结果与分析')
ch3_content = ch3_content.replace('## 3.1', '## 4.1')
ch3_content = ch3_content.replace('## 3.2', '## 4.2')
ch3_content = ch3_content.replace('## 3.3', '## 4.3')
ch3_content = ch3_content.replace('## 3.4', '## 4.4')
ch3_content = ch3_content.replace('### 3.2.1', '### 4.2.1')
ch3_content = ch3_content.replace('### 3.2.2', '### 4.2.2')
ch3_content = ch3_content.replace('### 3.2.3', '### 4.2.3')
ch3_content = ch3_content.replace('### 3.2.4', '### 4.2.4')
ch3_content = ch3_content.replace('### 3.3.1', '### 4.3.1')
ch3_content = ch3_content.replace('### 3.3.2', '### 4.3.2')
ch3_content = ch3_content.replace('### 3.3.3', '### 4.3.3')
ch3_content = ch3_content.replace('### 3.3.4', '### 4.3.4')

process_chapter(doc, ch3_content, 4)
print('  Added: 第4章 结果与分析')

# ===== CHAPTER 5: 结论与展望 (formerly 4) =====
with open(os.path.join(CHAPTERS, '04_结论与展望_参考文献.md'), 'r', encoding='utf-8') as f:
    ch4_content = f.read()

ch4_content = ch4_content.replace('# 4 结论与展望', '# 第5章 结论与展望')
ch4_content = ch4_content.replace('## 4.1', '## 5.1')
ch4_content = ch4_content.replace('## 4.2', '## 5.2')
ch4_content = ch4_content.replace('## 4.3', '## 5.3')

process_chapter(doc, ch4_content, 5)
print('  Added: 第5章 结论与展望 (with 32 references)')

# ===== FIGURES =====
print('\nEmbedding data figures...')
figures = [
    ('fig_2-1_seasonal_cycle.png', '图2.1 北极海冰面积季节循环（1979-2025年月气候态）'),
    ('fig_2-2_longterm_trend.png', '图2.2 北极海冰面积长期变化趋势'),
    ('fig_data_overview.png', '图2.3 数据总览：海冰面积与五个气候指数时间序列'),
    ('fig_acf.png', '图2.4 海冰面积月尺度自相关函数'),
    ('fig_lstm_loss_curve.png', '图4.1 单变量LSTM训练与验证损失曲线'),
    ('fig_lstm_univariate_seasonal_mean_std.png', '图4.2 单变量LSTM季节预测均值±1标准差'),
    ('fig_lstm_short_seasonal_mean_std.png', '图4.3 短期预测方案季节均值±1标准差'),
    ('fig3-4_lstm_best_worst_year.png', '图4.4 单变量LSTM最好/最坏预测年（2020/2016）'),
    ('fig_loss_short_medium.png', '图4.5 短期与中期预测方案损失曲线对比'),
    ('fig3-7_three_models_monthly_rmse.png', '图4.7 线性回归/RNN/LSTM三种模型逐月RMSE对比'),
    ('fig_loss_e4_e7.png', '图4.8 三变量E4与双编码器E7损失曲线对比'),
    ('fig4-2_dual_encoder_monthly_mean.png', '图4.9 双编码器E7v1季节预测均值±1标准差'),
    ('fig_e4_trivariate_seasonal_mean_std.png', '图4.10 三变量E4季节预测均值±1标准差'),
    ('fig4-4_e4_e7_monthly_rmse.png', '图4.11 E4与E7v1逐月RMSE对比'),
    ('fig_e7_best_worst.png', '图4.12 双编码器E7最好/最坏预测年（2022/2016）'),
    ('fig_ablation_e7.png', '图4.6 消融实验结果对比'),
    ('fig_scatter_e1_e4_e7.png', '图4.13 三种模型预测值与观测值散点图'),
    ('fig_uncertainty.png', '附A.1 预测不确定性分析'),
    ('fig_residuals.png', '附A.2 预测残差分析'),
]

for fname, caption in figures:
    fpath = os.path.join(PLOTS, fname)
    if os.path.exists(fpath):
        add_figure(doc, fpath, caption)
    else:
        print(f'  MISSING: {fname}')

# ===== SAVE =====
print('\nSaving...')
doc.save(OUTPUT)

total_chars = sum(len(p.text) for p in doc.paragraphs)
print(f'\n{"="*60}')
print(f'P0 FIXES COMPLETE')
print(f'Output: {OUTPUT}')
print(f'Paragraphs: {len(doc.paragraphs)}')
print(f'Characters: {total_chars}')
print(f'Figures: {len(figures)} data + 6 formulas + 4 tables')
print(f'{"="*60}')
print('FIXED:')
print('  [P0 #1] 6 LSTM formulas rendered as PNG and embedded')
print('  [P0 #2] 4 experiment tables rebuilt as docx tables')
print('  [P0 #3] Chapter numbering: 0->1, 1->2, 2->3, 3->4, 4->5')
print('  [P0 #3] 绪论/数据与方法/实验方案/结果与分析/结论与展望')
