#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final compilation: polish text + generate architecture diagram + embed figures → polished .docx"""
import docx, sys, io, os, re
from docx.shared import Pt, Inches, Cm, Emu, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r'c:\Users\86152\PycharmProjects\2 +ao arctic_seaice_prediction lstm SIE'
PLOTS = os.path.join(BASE, 'outputs', 'plots', 'paper')

# ============================================================
# Step 1: Generate Dual-Encoder LSTM architecture diagram
# ============================================================
print("Generating architecture diagram...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Register Chinese font
import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
    plt.rcParams["font.sans-serif"] = ["SimHei"]
except:
    pass
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

# Color scheme
c_main = '#2166ac'    # blue - main encoder
c_aux = '#1b7837'     # green - aux encoder
c_output = '#d73027'  # red - output
c_box = '#f7f7f7'
c_arrow = '#555555'

def draw_box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    """Draw a rounded box with text"""
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333333', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=weight, color='white' if color != c_box else '#333333')

def draw_arrow(ax, x1, y1, x2, y2, color=c_arrow, lw=1.5, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# === Input section ===
# Sea ice input
draw_box(ax, 1.5, 5.5, 2.0, 0.7, '海冰面积历史\n12个月输入', c_main, 8, True)
ax.text(1.5, 6.2, '主编码器输入', ha='center', fontsize=7, color=c_main, fontweight='bold')

# Climate index input
draw_box(ax, 1.5, 3.5, 2.0, 0.7, '气候指数滞后特征\n6个月 × 19通道', c_aux, 8, True)
ax.text(1.5, 4.2, '辅助编码器输入', ha='center', fontsize=7, color=c_aux, fontweight='bold')

# LSTM encoders
draw_box(ax, 5.0, 5.5, 2.0, 1.0, '主编码器 LSTM\nmain_hidden=256\ndropout=0.1', c_main, 8)
draw_box(ax, 5.0, 3.5, 2.0, 1.0, '辅助编码器 LSTM\naux_hidden=64\ndropout=0.6', c_aux, 8)

# Hidden states
draw_box(ax, 8.0, 5.5, 1.8, 0.7, '主隐藏状态\nh_main (256维)', '#5a9bd4', 8)
draw_box(ax, 8.0, 3.5, 1.8, 0.7, '辅助隐藏状态\nh_aux (64维)', '#5aab6e', 8)

# Concatenation
draw_box(ax, 10.5, 4.5, 1.8, 0.8, '特征拼接\nConcat (320维)', '#f0ad4e', 8)

# Output
draw_box(ax, 12.8, 4.5, 2.0, 0.8, '全连接层\n12个月预测输出', c_output, 8, True)

# Arrows: input → encoder
draw_arrow(ax, 2.5, 5.5, 4.0, 5.5, c_main, 2)
draw_arrow(ax, 2.5, 3.5, 4.0, 3.5, c_aux, 2)

# Encoder → hidden
draw_arrow(ax, 6.0, 5.5, 7.1, 5.5, c_main, 2)
draw_arrow(ax, 6.0, 3.5, 7.1, 3.5, c_aux, 2)

# Hidden → concat
draw_arrow(ax, 8.9, 5.3, 9.6, 4.7, c_arrow, 1.5)
draw_arrow(ax, 8.9, 3.7, 9.6, 4.5, c_arrow, 1.5)

# Concat → output
draw_arrow(ax, 11.4, 4.5, 11.8, 4.5, c_output, 2)

# Labels
ax.text(5.0, 6.6, '不对称设计: 4:1 隐藏维度比', ha='center', fontsize=7, color='#888888', style='italic')
ax.text(5.0, 2.8, '强正则化: 6× Dropout率', ha='center', fontsize=7, color='#888888', style='italic')

# Legend-like annotations at bottom
ax.text(1.0, 1.5, '图1.5 双编码器LSTM架构示意图', fontsize=11, fontweight='bold', color='#333333')
ax.text(1.0, 1.0, '注：主编码器处理海冰面积历史序列，辅助编码器处理多变量气候指数滞后特征。', fontsize=7, color='#666666')
ax.text(1.0, 0.7, '两者隐藏状态拼接后经全连接层一次性输出12个月预测值。', fontsize=7, color='#666666')

# Add subfigure labels
ax.text(0.3, 6.8, '(a)', fontsize=14, fontweight='bold')

# Save
arch_path = os.path.join(PLOTS, 'fig_arch_dual_encoder.png')
plt.tight_layout()
plt.savefig(arch_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Architecture diagram saved: {arch_path}")

# ============================================================
# Step 2: Generate experimental design framework diagram
# ============================================================
print("Generating experimental framework diagram...")
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')

phases = [
    (1.5, 3.5, '阶段一\n单变量增量\n(6实验)', '每个气候指数独立\n加入辅助编码器', c_main),
    (4.5, 3.5, '阶段二\n变量组合\n(7实验)', '同扇区→跨扇区\n→全变量递进', '#d73027'),
    (7.5, 3.5, '阶段三\n消融验证\n(4实验)', '全变量逐个移除\n反向确认必要性', '#1b7837'),
    (10.5, 3.5, '阶段四\n空间异质性\n(21+7实验)', '七海域匹配检验\n+独立调参', '#f0ad4e'),
]

for x, y, title, desc, color in phases:
    draw_box(ax, x, y, 2.5, 1.5, title, color, 10, True)
    ax.text(x, y-1.0, desc, ha='center', fontsize=7, color='#555555')

# Arrows between phases
for i in range(len(phases)-1):
    x1 = phases[i][0] + 1.25
    x2 = phases[i+1][0] - 1.25
    y = phases[i][1]
    draw_arrow(ax, x1, y, x2, y, '#333333', 2.5)

# Down arrow annotation
ax.text(7.0, 2.0, '递进式实验设计：变量维度递增 → 空间维度扩展 → 参数维度优化', 
        ha='center', fontsize=8, fontweight='bold', color='#333333')

ax.text(1.0, 0.5, '图2.1 实验设计框架示意图', fontsize=11, fontweight='bold', color='#333333')

frame_path = os.path.join(PLOTS, 'fig_experimental_framework.png')
plt.tight_layout()
plt.savefig(frame_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Framework diagram saved: {frame_path}")

print("\nDiagrams generated successfully!")
