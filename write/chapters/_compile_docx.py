#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compile all chapters into final .docx using template formatting"""
import docx, sys, io, os, re
from docx.shared import Pt, Inches, Cm, Emu, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r'c:\Users\86152\PycharmProjects\2 +ao arctic_seaice_prediction lstm SIE'

# Load template for style reference
template_path = os.path.join(BASE, 'write', '融合Cellpose与圆度判别的盐水冰微结构自动识别与参数提取方法.docx')
target_path = os.path.join(BASE, 'write', '基于LSTM的多源气候指数的北极海冰面积预测研究 (已自动恢复).docx')
output_path = os.path.join(BASE, 'write', '基于LSTM的多源气候指数的北极海冰面积预测研究_完整版.docx')

# Use the template as base document (preserves styles)
doc = docx.Document(template_path)

# Clear all existing paragraphs except title/author/institution/abstract styles
# Strategy: remove all content, then re-add using template styles
# Keep the document but clear body
for p in doc.paragraphs:
    p.clear()

# Also clear element body
body = doc.element.body
for p in body.findall(qn('w:p')):
    body.remove(p)

# Now write all content
# Helper functions
def add_paragraph_with_style(doc, text, style_name, bold=False, size=None, alignment=None):
    """Add a paragraph with given style"""
    p = doc.add_paragraph(style=style_name)
    if text:
        run = p.add_run(text)
        if bold:
            run.bold = True
        if size:
            run.font.size = size
    if alignment is not None:
        p.alignment = alignment
    return p

def add_normal_para(doc, text):
    """Add a normal body paragraph with first-line indent"""
    p = doc.add_paragraph(style='Normal')
    if text:
        p.add_run(text)
    return p

def add_heading1(doc, text):
    """Add Heading 1"""
    return doc.add_heading(text, level=1)

def add_heading2(doc, text):
    """Add Heading 2"""
    return doc.add_heading(text, level=2)

def add_heading3(doc, text):
    """Add Heading 3"""
    return doc.add_heading(text, level=3)

# ====== TITLE PAGE ======
doc.add_paragraph('基于LSTM的多源气候指数的北极海冰面积预测研究', style='Title')

p = doc.add_paragraph(style='作者')
p.add_run('谢天')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph(style='单位')
p.add_run('（大连理工大学，海岸与海洋工程全国重点实验室，辽宁  大连  116024）')

# ====== ABSTRACT ======
abstract_text = (
    '摘要：现有基于深度学习的海冰预测研究多依赖单一海冰数据源，气候指数能否提供额外预测技能及其增量价值在何种条件下成立，尚未得到系统性评估。'
    '本文基于1979-2025年NSIDC月平均海冰面积及同期五个气候指数（北极涛动AO、北大西洋涛动NAO、太平洋-北美型遥相关PNA、'
    'Niño 3.4区海表温度异常指数Nino3.4及北极海表温度SST），构建双编码器长短时记忆网络（Dual-Encoder LSTM）框架，'
    '通过单变量增量、变量组合、消融验证及七海域空间异质性分析四阶段递进实验（共38个独立实验），系统性检验了标量气候指数对北极海冰面积预测的增量贡献。'
    '结果表明：（1）气候指数增量技能有限但真实存在，全北极最优单变量PNA改善约2.8%，全变量组合与纯冰基线几乎无差异；'
    '（2）"越少越好"的变量选择规律在全北极和区域尺度一致成立，AO在组合中持续起负面作用；'
    '（3）气候指数预测价值的空间异质性范围有限，统一超参数下仅巴伦支海呈现明显NAO增益优势（3.4倍于PNA），'
    '但区域独立调参后白令海（−8.6%）、巴伦支海（−7.3%）、喀拉海（−9.9%）预测误差大幅降低，'
    '而中北冰洋和楚科奇海仍近零增益，表明敏感海域集中在受海洋入流直接影响的边缘海。'
    '本研究为标量气候指数在海冰统计预测中的价值提供了首个系统性定量评估基准。'
)
doc.add_paragraph(abstract_text, style='摘要')

kw_text = '关键词：北极海冰面积；长短时记忆网络；气候指数；双编码器；变量消融；空间异质性'
doc.add_paragraph(kw_text, style='摘要')

# ====== Read chapters ======
chapters_dir = os.path.join(BASE, 'write', 'chapters')
chapter_files = [
    ('00_引言.md', '0 引言', 1),
    ('01_数据与方法.md', '1 数据与方法', 1),
    ('02_实验方案设计.md', '2 实验方案设计', 1),
    ('03_结果与分析.md', '3 结果与分析', 1),
    ('04_结论与展望_参考文献.md', '4 结论与展望', 1),
]

for fname, default_h1, _ in chapter_files:
    fpath = os.path.join(chapters_dir, fname)
    if not os.path.exists(fpath):
        print(f'WARNING: {fname} not found, skipping')
        continue

    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse markdown into paragraphs
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Heading detection
        if line.startswith('## '):
            text = line[3:].strip()
            doc.add_heading(text, level=2)
        elif line.startswith('### '):
            text = line[4:].strip()
            doc.add_heading(text, level=3)
        elif line.startswith('# '):
            text = line[2:].strip()
            doc.add_heading(text, level=1)
        elif line.startswith('**表'):
            # Table caption - add as centered bold
            p = doc.add_paragraph(style='Normal')
            run = p.add_run(line.replace('**', ''))
            run.bold = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith('|'):
            # Skip markdown tables (they're not easily convertible to docx tables here)
            # Just note them
            i += 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                i += 1
            continue
        elif line.startswith('$$'):
            # Skip LaTeX math blocks
            i += 1
            continue
        elif line.startswith('[') and line.strip().endswith(']'):
            # Reference line
            p = doc.add_paragraph(style='EndNote Bibliography')
            p.add_run(line.strip())
        elif line == '## 参考文献' or line == '# 参考文献':
            doc.add_heading('参考文献', level=1)
        else:
            # Normal paragraph
            p = doc.add_paragraph(style='Normal')
            # Handle inline math $...$
            text = line
            # Remove bold markers for simplicity
            text = text.replace('**', '')
            p.add_run(text)

        i += 1

    print(f'  Added: {fname}')

# Save
doc.save(output_path)
print(f'\nSaved to: {output_path}')

# Count stats
total_chars = sum(len(p.text) for p in doc.paragraphs)
print(f'Total paragraphs: {len(doc.paragraphs)}')
print(f'Total characters: {total_chars}')
print(f'Estimated Chinese chars: ~{total_chars}')
