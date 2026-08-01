#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-shot: compile final polished .docx with all chapters + embedded figures"""
import docx, sys, io, os
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r'c:\Users\86152\PycharmProjects\2 +ao arctic_seaice_prediction lstm SIE'
PLOTS = os.path.join(BASE, 'outputs', 'plots', 'paper')
CHAPTERS = os.path.join(BASE, 'write', 'chapters')
TEMPLATE = os.path.join(BASE, 'write', '融合Cellpose与圆度判别的盐水冰微结构自动识别与参数提取方法.docx')
OUTPUT = os.path.join(BASE, 'write', '基于LSTM的多源气候指数的北极海冰面积预测研究_终稿.docx')

# ---- Polish rules: (old, new) pairs applied to each chapter ----
POLISH_RULES = [
    # Chapter 1 fixes
    ('引了入', '引入了'),
    ('月到季节', '月至季节'),
    ('Niño', 'Nino'),
    # General Chinese-English spacing
    ('1979年至2025年', '1979 年至 2025 年'),
    ('约13%每十年', '约 13% 每十年'),
    ('约7.0百万', '约 7.0 百万'),
    ('约4.0百万', '约 4.0 百万'),
    ('2至4倍', '2 至 4 倍'),
    ('1至12个月', '1 至 12 个月'),
    ('10至14个', '10 至 14 个'),
    ('3至6个月', '3 至 6 个月'),
    ('约2周', '约 2 周'),
    ('12个月', '12 个月'),
    ('0.36至0.43', '0.36 至 0.43'),
    ('1至2个', '1 至 2 个'),
    ('约70%', '约 70%'),
    ('约95%', '约 95%'),
    ('0.015百万', '0.015 百万'),
    ('约500个', '约 500 个'),
    # LaTeX escape fixes
    ('10⁶ km²', '10⁶ km²'),
]

def polish_text(text):
    """Apply all polish rules to text"""
    for old, new in POLISH_RULES:
        text = text.replace(old, new)
    return text

def add_figure(doc, img_path, caption, width_inches=5.5):
    if not os.path.exists(img_path):
        print(f'  WARNING: Figure missing: {os.path.basename(img_path)}')
        return
    doc.add_paragraph(style='Normal')
    p = doc.add_paragraph(style='Normal')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(img_path, width=Inches(width_inches))
    cap = doc.add_paragraph(style='Normal')
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_run = cap.add_run(caption)
    cap_run.font.size = Pt(8)
    cap_run.bold = True

def add_normal(doc, text):
    if not text or not text.strip():
        return
    p = doc.add_paragraph(style='Normal')
    p.add_run(text.strip())

def add_heading(doc, text, level):
    doc.add_heading(text, level=level)

def process_markdown(doc, content):
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('## '):
            add_heading(doc, line[3:].strip(), 2)
        elif line.startswith('### '):
            add_heading(doc, line[4:].strip(), 3)
        elif line.startswith('# '):
            add_heading(doc, line[2:].strip(), 1)
        elif line.startswith('**') and ('表' in line or '图' in line):
            p = doc.add_paragraph(style='Normal')
            run = p.add_run(line.replace('**', ''))
            run.bold = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
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
        elif line == '## 参考文献' or line == '# 参考文献':
            add_heading(doc, '参考文献', level=1)
        else:
            text = line.replace('**', '')
            add_normal(doc, text)
        i += 1

# ================================================================
# MAIN
# ================================================================
print('='*60)
print('Building final polished thesis .docx')
print('='*60)

# Phase 1: Polish chapters
print('\n[Phase 1] Polishing chapters...')
chapter_files = [
    '00_引言.md',
    '01_数据与方法.md',
    '02_实验方案设计.md',
    '03_结果与分析.md',
    '04_结论与展望_参考文献.md',
]

for fname in chapter_files:
    fpath = os.path.join(CHAPTERS, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        original = f.read()
    polished = polish_text(original)
    if polished != original:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(polished)
        print(f'  Polished: {fname}')
    else:
        print(f'  Unchanged: {fname}')

# Phase 2: Load template and build docx
print('\n[Phase 2] Building docx from template...')
doc = docx.Document(TEMPLATE)

# Clear template content
body = doc.element.body
for p in body.findall(qn('w:p')):
    body.remove(p)

# ---- TITLE PAGE ----
doc.add_paragraph('基于LSTM的多源气候指数的北极海冰面积预测研究', style='Title')

p = doc.add_paragraph(style='作者')
p.add_run('谢天')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph(style='单位')
p.add_run('（大连理工大学，海岸和近海工程国家重点实验室，辽宁  大连  116024）')

# ---- ABSTRACT (polished version from qinyan-paper-polish) ----
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

# ---- CHAPTERS ----
print('\n[Phase 3] Adding chapters...')
for fname in chapter_files:
    fpath = os.path.join(CHAPTERS, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    process_markdown(doc, content)
    print(f'  Added: {fname} ({len(content)} chars)')

# ---- FIGURES ----
print('\n[Phase 4] Embedding figures...')
figures = [
    ('fig_arch_dual_encoder.png', '图1.5 双编码器LSTM架构示意图'),
    ('fig_experimental_framework.png', '图2.1 四阶段递进式实验设计框架'),
    ('fig_2-1_seasonal_cycle.png', '图1.1 北极海冰面积季节循环（1979-2025年月气候态）'),
    ('fig_2-2_longterm_trend.png', '图1.2 北极海冰面积长期变化趋势'),
    ('fig_data_overview.png', '图1.3 数据总览：海冰面积与五个气候指数时间序列'),
    ('fig_acf.png', '图1.4 海冰面积月尺度自相关函数'),
    ('fig_lstm_loss_curve.png', '图3.1 单变量LSTM训练与验证损失曲线'),
    ('fig_lstm_univariate_seasonal_mean_std.png', '图3.2 单变量LSTM季节预测均值±1标准差'),
    ('fig_lstm_short_seasonal_mean_std.png', '图3.3 短期预测方案季节均值±1标准差'),
    ('fig3-4_lstm_best_worst_year.png', '图3.4 单变量LSTM最好/最坏预测年（2020/2016）'),
    ('fig_loss_short_medium.png', '图3.5 短期与中期预测方案损失曲线对比'),
    ('fig3-7_three_models_monthly_rmse.png', '图3.7 线性回归/RNN/LSTM三种模型逐月RMSE对比'),
    ('fig_loss_e4_e7.png', '图4.1 三变量E4与双编码器E7损失曲线对比'),
    ('fig4-2_dual_encoder_monthly_mean.png', '图4.2 双编码器E7v1季节预测均值±1标准差'),
    ('fig_e4_trivariate_seasonal_mean_std.png', '图4.3 三变量E4季节预测均值±1标准差'),
    ('fig4-4_e4_e7_monthly_rmse.png', '图4.4 E4与E7v1逐月RMSE对比'),
    ('fig_e7_best_worst.png', '图4.5 双编码器E7最好/最坏预测年（2022/2016）'),
    ('fig_ablation_e7.png', '图3.8 消融实验结果对比'),
    ('fig_scatter_e1_e4_e7.png', '图3.9 三种模型预测值与观测值散点图'),
    ('fig_uncertainty.png', '图3.10 预测不确定性分析'),
    ('fig_residuals.png', '图3.11 预测残差分析'),
]

for fname, caption in figures:
    fpath = os.path.join(PLOTS, fname)
    if os.path.exists(fpath):
        add_figure(doc, fpath, caption)
        print(f'  Embedded: {fname}')
    else:
        print(f'  MISSING: {fname}')

# ---- SAVE ----
print('\n[Phase 5] Saving...')
doc.save(OUTPUT)

total_chars = sum(len(p.text) for p in doc.paragraphs)
print(f'\n{"="*60}')
print(f'FINAL OUTPUT: {OUTPUT}')
print(f'Paragraphs: {len(doc.paragraphs)}')
print(f'Characters: {total_chars}')
print(f'Figures embedded: {len(figures)}')
print(f'{"="*60}')
print('DONE - Thesis build complete!')
