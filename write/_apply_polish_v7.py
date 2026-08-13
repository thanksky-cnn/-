"""将 _v7_polished.txt 的润色内容应用到 v7.docx → v8.docx"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from docx import Document
from docx.shared import Pt

# 1. 读取润色后的段落
with open('write/_v7_polished.txt', 'r', encoding='utf-8') as f:
    polished_lines = f.read().split('\n')

# 过滤出正文段落（跳过表格区域和纯空行）
polished_paras = []
in_table_section = False
for line in polished_lines:
    stripped = line.strip()
    if stripped.startswith('表1 (') or stripped.startswith('表2 (') or stripped.startswith('表3 (') or stripped.startswith('表4 ('):
        in_table_section = True
    if in_table_section:
        continue
    if stripped == '':
        polished_paras.append('')  # 保留空段落
    else:
        polished_paras.append(stripped)

# 2. 打开 v7.docx
doc = Document('write/基于LSTM的多源气候指数的北极海冰面积预测研究_填入终稿_v7.docx')
doc_paras = doc.paragraphs

# 3. 逐段替换（按索引对齐）
replaced = 0
skipped = 0
pi = 0  # polished index

for dp in doc_paras:
    if pi >= len(polished_paras):
        break

    new_text = polished_paras[pi]
    old_text = dp.text.strip()

    # 跳过表格内段落（docx中表格也有paragraphs，但我们已经单独处理）
    # 跳过公式占位符、图表标签等不需要替换的内容
    if old_text == new_text:
        pi += 1
        skipped += 1
        continue

    if new_text == '' and old_text == '':
        pi += 1
        skipped += 1
        continue

    # 替换非空段落
    if new_text:
        # 保留原有格式，只替换文本
        for run in dp.runs:
            run.text = ''
        if dp.runs:
            dp.runs[0].text = new_text
        else:
            dp.text = new_text
        replaced += 1

    pi += 1

# 4. 保存 v8
doc.save('write/基于LSTM的多源气候指数的北极海冰面积预测研究_填入终稿_v8.docx')
print(f'Done. Replaced: {replaced}, Skipped: {skipped}, Total polished: {len(polished_paras)}, Total docx: {len(doc_paras)}')
