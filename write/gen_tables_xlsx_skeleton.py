
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter

BASE = r'C:\Users\86152\PycharmProjects\2 +ao arctic_seaice_prediction lstm SIE'
OUTPUT = os.path.join(BASE, 'write', '论文图表汇总.xlsx')

font_cn = Font(name='宋体', size=10.5)
font_en = Font(name='Times New Roman', size=10.5)
font_header = Font(name='黑体', size=10.5, bold=True)
font_title = Font(name='黑体', size=12, bold=True)

thin = Side(style='thin', color='000000')
medium = Side(style='medium', color='000000')
border_top = Border(top=medium)
border_header = Border(bottom=thin)
border_bottom = Border(bottom=medium)

align_center = Alignment(horizontal='center', vertical='center', wrap_text=True)
align_left = Alignment(horizontal='left', vertical='center', wrap_text=True)
align_right = Alignment(horizontal='right', vertical='center', wrap_text=True)

print('Script skeleton written')
