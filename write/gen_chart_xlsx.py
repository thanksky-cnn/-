#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, re, json

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'write', '论文图表汇总.xlsx')
DATA_FILE = os.path.join(BASE, 'write', 'chart_data.json')

wb = Workbook()

# === Styles ===
font_title   = Font(name='SimHei', size=12, bold=True)
font_header  = Font(name='SimHei', size=10.5, bold=True)
font_body    = Font(name='SimSun', size=10.5)
font_en      = Font(name='Times New Roman', size=10.5)
font_note    = Font(name='SimSun', size=9, italic=True)

medium = Side(style='medium', color='000000')
thin   = Side(style='thin', color='000000')

align_c = Alignment(horizontal='center', vertical='center', wrap_text=True)
align_l = Alignment(horizontal='left', vertical='center', wrap_text=True)
align_r = Alignment(horizontal='right', vertical='center', wrap_text=True)

header_fill = PatternFill(start_color='F2F2F2', end_color='F2F2F2', fill_type='solid')

def apply_row_style(ws, row, ncols, font, alignments=None):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = font
        al = alignments[c - 1] if alignments else align_c
        cell.alignment = al
        val = str(cell.value) if cell.value is not None else ''
        if val and re.match(r'[\d.\-+%]+$', val.strip()):
            cell.font = font_en

def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = font_header
        cell.alignment = align_c
        cell.fill = header_fill
        cell.border = Border(bottom=thin)

def add_three_line(ws, title_row, header_row, data_start, data_end, ncols):
    """标准三线表：仅顶线(粗)、表头线(细)、底线(粗)三条横线"""
    for c in range(1, ncols + 1):
        # Row 1 (title): thick top border only
        ws.cell(row=title_row, column=c).border = Border(top=medium)
        # Row 2 (header): thin bottom border only (table header line)
        ws.cell(row=header_row, column=c).border = Border(bottom=thin)
    for c in range(1, ncols + 1):
        # Last data row: thick bottom border only (table bottom line)
        ws.cell(row=data_end, column=c).border = Border(bottom=medium)

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ============ Sheet 1: 图清单 ============
ws1 = wb.active
ws1.title = '图清单'

fig_headers = ['序号', '图号', '标题', '所在章节', '类型', '面板数', '建议宽度', '对应SVG文件', '图表描述']
ncols_fig = len(fig_headers)

ws1.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols_fig)
ws1.cell(row=1, column=1, value='附表1  论文插图清单').font = font_title
ws1.cell(row=1, column=1).alignment = align_c

for c, h in enumerate(fig_headers, 1):
    ws1.cell(row=2, column=c, value=h)
style_header(ws1, 2, ncols_fig)

for i, fig in enumerate(data['figures']):
    row = i + 3
    vals = [i + 1, fig['id'], fig['title'], fig['chapter'], fig['type'],
            fig['panels'], fig['width'], fig['file'], fig['desc']]
    for c, v in enumerate(vals, 1):
        ws1.cell(row=row, column=c, value=v)
    alignments = [align_c, align_c, align_l, align_c, align_c, align_c, align_c, align_c, align_l]
    apply_row_style(ws1, row, ncols_fig, font_body, alignments)

add_three_line(ws1, 1, 2, 3, 2 + len(data['figures']), ncols_fig)

col_widths_fig = [5, 9, 30, 14, 12, 10, 9, 28, 60]
for c, w in enumerate(col_widths_fig, 1):
    ws1.column_dimensions[get_column_letter(c)].width = w

ws1.row_dimensions[1].height = 25
ws1.row_dimensions[2].height = 22
for r in range(3, 3 + len(data['figures'])):
    ws1.row_dimensions[r].height = 80

ws1.sheet_properties.pageSetUpPr = None
ws1.page_setup.orientation = 'landscape'
ws1.print_title_rows = '1:2'

print('Sheet 1 (图清单) done')

# ============ Sheet 2: 表清单 ============
ws2 = wb.create_sheet('表清单')

tbl_headers = ['序号', '表号', '标题', '所在章节', '列数', '数据行数', '备注']
ncols_tbl = len(tbl_headers)

ws2.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols_tbl)
ws2.cell(row=1, column=1, value='附表2  论文表格清单').font = font_title
ws2.cell(row=1, column=1).alignment = align_c

for c, h in enumerate(tbl_headers, 1):
    ws2.cell(row=2, column=c, value=h)
style_header(ws2, 2, ncols_tbl)

for i, tm in enumerate(data['tables_meta']):
    row = i + 3
    nrows = len(tm['rows']) + 1
    vals = [i + 1, tm['id'], tm['title'], tm['chapter'],
            len(tm['headers']), nrows, tm['note']]
    for c, v in enumerate(vals, 1):
        ws2.cell(row=row, column=c, value=v)
    alignments = [align_c, align_c, align_l, align_c, align_c, align_c, align_l]
    apply_row_style(ws2, row, ncols_tbl, font_body, alignments)

add_three_line(ws2, 1, 2, 3, 2 + len(data['tables_meta']), ncols_tbl)

col_widths_tbl = [5, 9, 28, 14, 6, 6, 60]
for c, w in enumerate(col_widths_tbl, 1):
    ws2.column_dimensions[get_column_letter(c)].width = w

ws2.row_dimensions[1].height = 25
ws2.row_dimensions[2].height = 22
for r in range(3, 3 + len(data['tables_meta'])):
    ws2.row_dimensions[r].height = 45

ws2.sheet_properties.pageSetUpPr = None
ws2.page_setup.orientation = 'landscape'
ws2.print_title_rows = '1:2'

print('Sheet 2 (表清单) done')

# ============ Sheets 3-6: Individual tables ============
for tm in data['tables_meta']:
    ws = wb.create_sheet(tm['id'])

    headers = tm['headers']
    rows_data = tm['rows']
    ncols = len(headers)

    title_text = tm['id'] + '  ' + tm['title']
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    ws.cell(row=1, column=1, value=title_text).font = font_title
    ws.cell(row=1, column=1).alignment = align_c

    for c, h in enumerate(headers, 1):
        ws.cell(row=2, column=c, value=h)
    style_header(ws, 2, ncols)

    for i, row_data in enumerate(rows_data):
        row = i + 3
        for c, v in enumerate(row_data, 1):
            ws.cell(row=row, column=c, value=v)
        alignments = [align_c] * ncols
        apply_row_style(ws, row, ncols, font_body, alignments)

    note_row = 3 + len(rows_data)
    ws.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=ncols)
    ws.cell(row=note_row, column=1, value=tm['note']).font = font_note
    ws.cell(row=note_row, column=1).alignment = align_l

    add_three_line(ws, 1, 2, 3, 2 + len(rows_data), ncols)

    col_w = [14, 30, 18, 42, 50][:ncols]
    for c, w in enumerate(col_w, 1):
        ws.column_dimensions[get_column_letter(c)].width = w

    ws.row_dimensions[1].height = 25
    ws.row_dimensions[2].height = 22
    for r in range(3, 3 + len(rows_data)):
        ws.row_dimensions[r].height = 30
    ws.row_dimensions[note_row].height = 40

    ws.print_title_rows = '1:2'

    print(f'Sheet {tm["id"]} done')

# Save
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
wb.save(OUTPUT)
print(f'\nDone! Output: {OUTPUT}')
print(f'Sheets: {wb.sheetnames}')
