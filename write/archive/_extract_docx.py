#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提取用户修改过的 v4.docx 中 Ch2-Ch3 文本，与 .md 源文件对比"""
import os, sys, re
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
from docx import Document

BASE = os.path.dirname(os.path.abspath(__file__))
doc = Document(os.path.join(BASE, '基于LSTM的多源气候指数的北极海冰面积预测研究_填入终稿_v4.docx'))

# 提取所有非空段落
doc_paras = [(i, p.text.strip(), p.style.name) for i, p in enumerate(doc.paragraphs) if p.text.strip()]
print(f'Docx: {len(doc.paragraphs)} total paras, {len(doc_paras)} non-empty')

# 找章节边界
ch2_start = ch4_start = None
for i, (idx, text, style) in enumerate(doc_paras):
    if '数据与方法' in text and ch2_start is None:
        ch2_start = i
    if '结论与展望' in text and ch4_start is None:
        ch4_start = i

print(f'Ch2边界: doc_para[{ch2_start}], Ch4边界: doc_para[{ch4_start}]')
print(f'Ch2-Ch3 共 {ch4_start - ch2_start} 段\n')

# 输出 Ch2-Ch4 之间的所有段落（用户修改区域）
print('=' * 70)
print('用户修改区域 (Ch2 数据与方法 → Ch4 结论与展望):')
print('=' * 70)
for idx, text, style in doc_paras[ch2_start:ch4_start]:
    # 截断长段落
    display = text[:120].replace('\n', ' ')
    if len(text) > 120:
        display += ' [...]'
    print(f'[{idx}] [{style}] {display}')
    print()

# 保存完整文本到文件供对比
output_path = os.path.join(BASE, '_docx_ch2ch3_text.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for idx, text, style in doc_paras[ch2_start:ch4_start]:
        f.write(f'[{idx}] [{style}]\n{text}\n\n')
print(f'\n完整文本已保存到: {output_path}')
