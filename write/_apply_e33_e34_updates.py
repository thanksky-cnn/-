#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直接在 v5 docx 中应用 E33/E34 集成修改"""
import copy
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = r"write\基于LSTM的多源气候指数的北极海冰面积预测研究_填入终稿_v5.docx"
DST = r"write\基于LSTM的多源气候指数的北极海冰面积预测研究_填入终稿_v6.docx"

doc = Document(SRC)
changes = 0

# ── helpers ──
def find_para(doc, needle):
    for i, p in enumerate(doc.paragraphs):
        if needle in p.text:
            return i
    return None

def replace_in_para(p, old, new):
    full = p.text
    for run in p.runs:
        if old in run.text:
            run.text = run.text.replace(old, new)
            return True
    if old in full:
        p.runs[0].text = full.replace(old, new)
        for r in p.runs[1:]:
            r.text = ''
        return True
    p.text = full.replace(old, new)  # fallback
    return True

def set_para_text(p, text):
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ''
    else:
        p.text = text

def insert_para_after(doc, after_idx, text):
    ref_p = doc.paragraphs[after_idx]
    ref_elem = ref_p._element
    new_p = OxmlElement('w:p')
    r = OxmlElement('w:r')
    t = OxmlElement('w:t')
    t.text = text
    t.set(qn('xml:space'), 'preserve')
    r.append(t)
    new_p.append(r)
    ref_elem.addnext(new_p)

def insert_table_row(table, after_row_idx, cells_text):
    ref_row = table.rows[after_row_idx]
    ref_elem = ref_row._tr
    new_tr = OxmlElement('w:tr')
    for ci, ct in enumerate(cells_text):
        tc = OxmlElement('w:tc')
        ref_tc = ref_row.cells[min(ci, len(ref_row.cells) - 1)]
        ref_tcPr = ref_tc._tc.find(qn('w:tcPr'))
        if ref_tcPr is not None:
            tc.append(copy.deepcopy(ref_tcPr))
        p = OxmlElement('w:p')
        r = OxmlElement('w:r')
        t = OxmlElement('w:t')
        t.text = ct
        t.set(qn('xml:space'), 'preserve')
        r.append(t)
        p.append(r)
        tc.append(p)
        new_tr.append(tc)
    ref_elem.addnext(new_tr)

def bulk_replace(doc, old, new):
    """替换全文中所有出现的 old -> new"""
    cnt = 0
    for p in doc.paragraphs:
        if old in p.text:
            replace_in_para(p, old, new)
            cnt += 1
    return cnt

# ── 执行 ──

# 1. 全局计数替换
changes += bulk_replace(doc, '38个', '40个')       # 38个 → 40个
changes += bulk_replace(doc, '17个实验', '19个实验')  # 17个实验 → 19个实验
changes += bulk_replace(doc, '7个组合实验', '8个组合实验')  # 7个组合实验 → 8个组合实验
changes += bulk_replace(doc, '7个变量组合实验', '8个变量组合实验')  # 7个变量组合实验 → 8个
print(f'全局替换完成')

# 2. "其余5个组合" → "其余6个组合"
changes += bulk_replace(doc, '其余5个组合', '其余6个组合')

# 3. 摘要 (P3) - 更新 AO 结论
idx = find_para(doc, '四阶段递进实验（共')  # 四阶段递进实验（共
if idx is not None:
    replace_in_para(doc.paragraphs[idx],
        'AO在组合中持续起负面作用',  # AO在组合中持续起负面作用
        'AO呈现独有的“单独有用、组合有害”矛盾模式——消融实验中移除AO是改善幅度最大的单一操作（E33, ΔRMSE=−0.0104），而AO单独加入时产生2.0%的正向增益，其双重性已被双向证据完整确认')
    changes += 1
    print('P3 摘要更新')

# 4. P77 - 组合实验框架描述
idx = find_para(doc, '同扇区冗余→跨扇区互补→全变量信息上限')  # 同扇区冗余→跨扇区互补→全变量信息上限
if idx is not None:
    p = doc.paragraphs[idx]
    replace_in_para(p, '同扇区冗余→跨扇区互补→全变量信息上限',
                    '同扇区冗余→跨扇区互补→纯跨扇区互补检验→全变量信息上限')
    changes += 1
    print(f'P{idx} 组合框架描述更新')

# 5. P84 后插入 E34 设计描述
idx = find_para(doc, '两者差异不显著甚至反转')  # 两者差异不显著甚至反转
if idx is not None:
    e34_design = (
        'E34/PNA+NAO（跨太平洋-大西洋扇区、不含AO）'
        '是对跨扇区互补假说最纯粹的检验。'
        '此前E23/AO+PNA虽为跨扇区配置，但AO作为全域指数'
        '同时参与两个扇区的大气环流表征，其全域覆盖效应'
        '使PNA和NAO各自的独立贡献无法被分离量化。'
        'E34仅使用太平洋扇区的PNA和大西洋扇区的NAO，'
        '完全排除AO的全域覆盖混淆，是唯一不含任何全域'
        '大气指数的纯粹跨扇区双变量配置。'
        '若PNA+NAO组合显著优于各自单独使用（E10/PNA=0.5089, '
        'E8/NAO=0.5192），则跨扇区互补假说成立；'
        '若组合性能不升反降，则表明PNA与NAO的'
        '大气遥相关信号存在跨洋盆信息冗余——'
        '尽管两者分属不同地理扇区，但可能通过行星波列的'
        '半球尺度耦合而共享大量方差。')
    insert_para_after(doc, idx, e34_design)
    changes += 1
    print(f'P{idx}后插入E34设计描述')

# 6. P87 - 消融验证描述
idx = find_para(doc, '消融验证采用反向逻辑')  # 消融验证采用反向逻辑
if idx is not None:
    p = doc.paragraphs[idx]
    replace_in_para(p,
        '实验设计见表3.3。',  # 实验设计见表3.3。
        '共设计5个实验，覆盖全部五个指数的逐一移除（表3.3）。'
        '其中E33（All−AO）为补做实验，使消融矩阵实现完整的五变量全覆盖。')
    changes += 1
    print(f'P{idx} 消融设计更新')

# 7. P91 - 交叉验证示例 (PNA→AO)
idx = find_para(doc, '消融验证与单变量增量实验构成互补的交叉验证逻辑链条')  # 消融验证与单变量增量实验构成互补的交叉验证逻辑链条
if idx is not None:
    new_text = (
        '消融验证与单变量增量实验构成互补的交叉验证逻辑链条。'
        '以AO为例：若E14/AO单独加入产生正向增益（Δ=−0.0106），'
        '但E33/从全变量中移除AO后性能同样改善（Δ=−0.0104）'
        '且幅度近乎相等，则表明AO所承载的海冰相关信息可被'
        '其他四个指数（SST、NAO、PNA、Nino3.4）完全等效替代——'
        'AO的半球尺度覆盖虽使其单独使用时成为有效的信息浓缩通道，'
        '但在多变量环境中其信号被分解至空间覆盖更精确、'
        '物理机制更专一的其他指数，自身蜕变为纯冗余噪声。')
    set_para_text(doc.paragraphs[idx], new_text)
    changes += 1
    print(f'P{idx} 交叉验证更新 (PNA→AO)')

# 8. P115后插入 E34 结果讨论 (找"海洋热力变量记忆时间更长"段落后)
idx = find_para(doc, '海洋热力变量记忆时间更长')  # 海洋热力变量记忆时间更长
if idx is not None:
    e34_result = (
        'E34/PNA+NAO跨扇区纯组合是全部8个组合实验中表现最差的配置之一'
        '（集成RMSE=0.5370, Δ=+0.0134 vs E1），仅略优于全大气三变量组合E24（0.5383），'
        '远劣于PNA单独使用（0.5089, 改善2.81%）和NAO单独使用（0.5192, 改善0.84%）。'
        '5种子RMSE范围为0.5205–0.5747，标准差达0.0197，为全部Phase 2实验中方差最大者，'
        '表明PNA+NAO组合存在严重的训练不稳定性。'
        '这一结果构成对“跨扇区互补”假说的直接否定——'
        '当排除AO的全域覆盖效应后，仅使用纯粹的太平洋和大西洋扇区'
        '大气遥相关型不仅无法产生互补增益，反而引入了严重的冗余噪声。'
        'PNA与NAO虽分属不同洋盆扇区，但其大气环流变率在半球尺度上通过'
        '行星波列耦合而高度相关——两者均受中纬度西风带强度和极涡位置的'
        '共同调控，在月尺度上的信息独立性远低于扇区地理分隔所暗示的水平。')
    insert_para_after(doc, idx, e34_result)
    changes += 1
    print(f'P{idx}后插入E34结果讨论')
else:
    print('⚠ 未找到“海洋热力变量记忆时间更长”')

# 9. P124 - 消融实验结果
idx = find_para(doc, '结果呈现出清晰的不对称模式')  # 结果呈现出清晰的不对称模式
if idx is not None:
    new_text = (
        '消融验证实验从全五变量组合（E19, RMSE=0.5243）中逐个移除单一指数，'
        '反向确认各变量的边际必要性。'
        '五个消融实验中，四个移除操作改善了性能，仅移除Nino3.4导致退化。'
        '其中移除AO（E33, RMSE=0.5138, Δ=−0.0104）改善幅度位居首位，'
        '其次为移除PNA（E25, 0.5154, Δ=−0.0089）、移除NAO（E20, 0.5161, Δ=−0.0081）'
        '和移除SST（E22, 0.5209, Δ=−0.0034）。'
        'AO是五个指数中在全变量环境里冗余度最高的变量——'
        '其单独加入时产生Δ=−0.0106的正向增益（E14），'
        '从全变量中移除时同样产生Δ=−0.0104的改善，两值近乎相等。'
        '这一对称性表明：AO携带的海冰相关信息可被其他四个指数完全等效替代——'
        'NAO和PNA分别在大西洋和太平洋扇区提供AO所覆盖的大气环流信息，'
        'SST和Nino3.4则覆盖AO与海冰耦合中涉及的海洋热力成分。')
    set_para_text(doc.paragraphs[idx], new_text)
    changes += 1
    print(f'P{idx} 消融结果重写 (E33首位)')

# 10. P126 - 消融交叉验证结论
idx = find_para(doc, 'PNA在单独加入时是最佳单变量')  # PNA在单独加入时是最佳单变量
if idx is not None:
    new_text = (
        '五种指数的“单独增量/消融移除”双向表现如下。'
        'AO呈现最典型的冗余模式——单独增益排名第二（E14, Δ=−0.0106）、'
        '移除改善排名第一（E33, Δ=−0.0104），'
        '两个方向的信号量值几乎对称，AO是全变量组合中最强的噪声源。'
        'PNA呈现次强冗余——单独增益排名第一（E10, Δ=−0.0147）、'
        '移除改善排名第二（E25, Δ=−0.0089），'
        '单变量最优但组合中独立信号被稀释。'
        'NAO冗余度与PNA接近——单独增益第四（E8, Δ=−0.0044）、'
        '移除改善第三（E20, Δ=−0.0081）。'
        'SST的边际效应最小——移除改善仅Δ=−0.0034，在全变量组合中冗余与信息近乎平衡。'
        'Nino3.4是五个指数中唯一提供不可替代独特信息的变量——'
        '单独增益第三（E9, Δ=−0.0055）、'
        '移除后唯一导致退化（E21, Δ=+0.0095），'
        '其热带太平洋的物理源地与所有极地指数在空间上完全正交，'
        '通过Rossby波列遥相关路径向北极传输独立外强迫信号。')
    set_para_text(doc.paragraphs[idx], new_text)
    changes += 1
    print(f'P{idx} 消融交叉验证重写')

# 11. P127 图3.5 caption
idx = find_para(doc, 'Phase 3消融验证ΔRMSE瀑布图')  # Phase 3消融验证ΔRMSE瀑布图
if idx is not None:
    p = doc.paragraphs[idx]
    if 'E33' not in p.text:
        replace_in_para(p,
            '向下延伸为改善绿色，向上为退化红色。',
            '向下延伸为改善（绿色），向上为退化（红色）。'
            '五个消融实验中，移除AO（E33）改善幅度最大。')
    changes += 1
    print(f'P{idx} 图3.5 caption更新')

# 12. P130 - 全北极汇总
idx = find_para(doc, '综合全北极三个阶段的')  # 综合全北极三个阶段的
if idx is not None:
    new_text = (
        '综合全北极三个阶段的19个实验，以集成RMSE排序，'
        '排名前五的实验为：E10（PNA only, 0.5089）、'
        'E17（SST+Nino3.4, 0.5098）、E14（AO only, 0.5130）、'
        'E33（All−AO, 0.5138）和E25（All−PNA, 0.5154）。'
        '排名前五中四个为单变量或双变量配置，'
        'E33（All−AO）是唯一进入前五的多变量（四变量）配置——'
        '但其本质是对AO冗余信息的针对性剔除（从五变量中移除最冗余者），'
        '而非多变量互补的正面证据。'
        '纯冰基线E1（0.5236）排名第13位，全五变量组合E19（0.5243）排名第14位，'
        '最低的两个实验为E34/PNA+NAO（0.5370）和E24/AO+NAO+PNA（0.5383）——'
        '全大气信息堆叠反而导致最大退化。'
        '这一全局排名格局表明，对全北极总海冰面积的12个月预测任务而言，'
        '最优策略是仅引入1至2个经过精心选择的气候指数，而非堆叠全部可得变量。')
    set_para_text(doc.paragraphs[idx], new_text)
    changes += 1
    print(f'P{idx} 全北极汇总重写')

# 13. P131 - 图3.6 标题
idx = find_para(doc, '全北极')  # 多匹配，需要更精确
if idx is not None:
    # Find more specific one
    for i, p in enumerate(doc.paragraphs):
        if 'Bootstrap森林图' in p.text and '17' in p.text:  # Bootstrap森林图 + 17
            replace_in_para(p, '17', '19')
            changes += 1
            print(f'P{i} 图3.6标题 17→19')
            break

# 14. P132 - 图3.6 caption details
for i, p in enumerate(doc.paragraphs):
    if '全部17个实验的集成RMSE' in p.text:  # 全部17个实验的集成RMSE
        replace_in_para(p, '全部17个实验', '全部19个实验')
        replace_in_para(p,
            '前五实验（E10, E17, E14, E25, E20）',
            '前五实验（E10, E17, E14, E33, E25）')
        changes += 1
        print(f'P{i} 图3.6 caption更新')
        break

# 15. P161 - AO悖论增强
for i, p in enumerate(doc.paragraphs):
    if '独立加入时产生正向增益、组合使用时系统性退化' in p.text:  # 单独加入时产生正向增益、组合使用时系统性退化
        marker = '对这种现象的一个'  # 对这种现象的一个
        evidence = (
            '反向证据更具有决定意义——从全五变量组合中移除AO'
            '（E33, 0.5138, Δ=−0.0104）是五个消融实验中改善幅度最大的操作，'
            '且移除AO后剩余四变量（SST+NAO+PNA+Nino3.4）的集成RMSE（0.5138）'
            '几乎等于AO单独使用时的水平（0.5130），表明其他四个指数联合'
            '可近乎完美等效替代AO的全部辅助预测信息。'
            '与此同时，跨扇区纯组合E34/PNA+NAO（0.5370）提供了另一角度的证据——'
            '即使两个最优单变量（PNA: 0.5089, NAO: 0.5192），在排除AO全域覆盖效应后'
            '直接组合，亦因跨洋盆大气环流的半球尺度耦联而产生严重退化，'
            '这表明“越少越好”规律的根本原因不在于特定变量（如AO）的独特性，'
            '而在于大气遥相关型之间固有的信息冗余——'
            '无论变量的地理扇区归属如何，堆叠多个大气环流指数均不能带来增量增益。')
        full = p.text
        pos = full.find(marker)
        if pos >= 0:
            new_full = full[:pos] + evidence + full[pos:]
            set_para_text(p, new_full)
            changes += 1
            print(f'P{i} AO悖论增强 (E33+E34证据)')
        break

# 16. P165 - 结论
for i, p in enumerate(doc.paragraphs):
    if 'AO在任何组合中起负面作用' in p.text:  # AO在任何组合中起负面作用
        replace_in_para(p, 'AO在任何组合中起负面作用',
            'AO呈现独特的“单独有用、组合有害”悖论：单独加入时产生2.0%的正向增益（E14），'
            '但从全变量中移除AO是改善幅度最大的消融操作（E33, Δ=−0.0104），'
            '且移除后剩余四变量的联合性能（0.5138）几乎等同于AO单独使用水平（0.5130），'
            '证实AO信息可被其他指数完全等效替代；'
            'PNA+NAO跨扇区纯组合（E34, 0.5370）更是直接否定跨扇区互补假说——'
            '两个分属不同洋盆扇区的指数组合后性能远劣于各自单独使用，'
            '表明大气遥相关型的信息冗余是跨扇区的半球尺度现象，而非局限于同扇区内部')
        changes += 1
        print(f'P{i} 结论(2)重写')
        break

# 17. 表格修改
# Table 3.2 (docx.tables[1]): insert E34 after E17 row
tbl2 = doc.tables[1]
e17_row_idx = None
for ri, row in enumerate(tbl2.rows):
    if row.cells[0].text.strip() == 'E17':
        e17_row_idx = ri
        break
if e17_row_idx is not None:
    has_e34 = any(row.cells[0].text.strip() == 'E34' for row in tbl2.rows)
    if not has_e34:
        insert_table_row(tbl2, e17_row_idx, [
            'E34', 'PNA + NAO', '跨太平洋-大西洋扇区（纯）',  # 跨太平洋-大西洋扇区（纯）
            '不含AO的纯粹跨扇区互补检验：若PNA与NAO各自独立信息正交，组合应优于单独使用'  # 不含AO的纯粹跨扇区互补检验
        ])
        changes += 1
        print('表3.2: 新增E34行')

# Table 3.3 (docx.tables[2]): append E33 after last row
tbl3 = doc.tables[2]
last_row = len(tbl3.rows) - 1
if not any(row.cells[0].text.strip() == 'E33' for row in tbl3.rows):
    insert_table_row(tbl3, last_row, ['E33', '全 − AO', 'AO'])
    changes += 1
    print('表3.3: 新增E33行')

# ── 保存 ──
doc.save(DST)

# 验证
doc2 = Document(DST)
e33_n = sum(1 for p in doc2.paragraphs if 'E33' in p.text)
e34_n = sum(1 for p in doc2.paragraphs if 'E34' in p.text)
n_40 = sum(1 for p in doc2.paragraphs if '40个' in p.text)
n_19 = sum(1 for p in doc2.paragraphs if '19个实验' in p.text)

print(f'\n{"="*60}')
print(f'✅ 修改完成，共 {changes} 处修改')
print(f'   输出: {DST}')
print(f'   验证: E33 出现 {e33_n} 次, E34 出现 {e34_n} 次')
print(f'   40个实验 出现 {n_40} 次, 19个实验 出现 {n_19} 次')
