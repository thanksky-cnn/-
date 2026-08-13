// 生成 §3.1 评估指标与分析方法 独立 Word 文档
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType,
  HeadingLevel
} = require("docx");

const BASE = "C:\\Users\\86152\\PycharmProjects\\2 +ao arctic_seaice_prediction lstm SIE\\write";
const OUTPUT = BASE + "\\§3.1_评估指标与分析方法.docx";

const FONT_BODY = "SimSun";
const FONT_MATH = "Cambria Math";
const SIZE_BODY = 24;   // 小四号 12pt
const SIZE_MATH = 22;   // 公式略小

// 正文段落
const P = (text) => new Paragraph({
  spacing: { after: 120, line: 360 },
  children: [new TextRun({ text, font: FONT_BODY, size: SIZE_BODY })],
});

// 正文+加粗混排: segments = [{t, b}] 或 纯字符串
const Pb = (segments) => new Paragraph({
  spacing: { after: 120, line: 360 },
  children: (Array.isArray(segments) ? segments : [{ t: segments }]).map(s =>
    new TextRun({ text: typeof s === "string" ? s : s.t, font: FONT_BODY, size: SIZE_BODY, bold: s.b || false })
  ),
});

// 居中公式块
const Eq = (latex) => new Paragraph({
  spacing: { before: 100, after: 100, line: 340 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: latex, font: FONT_MATH, size: SIZE_MATH, italics: true })],
});

// 公式编号
const EqNum = (latex, num) => new Paragraph({
  spacing: { before: 60, after: 100, line: 340 },
  alignment: AlignmentType.CENTER,
  children: [
    new TextRun({ text: latex, font: FONT_MATH, size: SIZE_MATH, italics: true }),
    new TextRun({ text: `    (${num})`, font: FONT_BODY, size: 20 }),
  ],
});

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: FONT_BODY, size: SIZE_BODY },
        paragraph: { spacing: { line: 360 } },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "SimHei" },
        paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children: [
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3.1 评估指标与分析方法")] }),

      // ===== 评估指标定义 =====
      P("本研究以均方根误差（Root Mean Squared Error, RMSE）作为主要评估指标，辅以平均绝对误差（Mean Absolute Error, MAE）和平均绝对百分比误差（Mean Absolute Percentage Error, MAPE）。各指标定义如下："),

      EqNum("RMSE = √((1/n) Σᵢ₌₁ⁿ (yᵢ − ŷᵢ)²)", "1"),
      Eq("MAE = (1/n) Σᵢ₌₁ⁿ |yᵢ − ŷᵢ|"),
      Eq("MAPE = (1/n) Σᵢ₌₁ⁿ |(yᵢ − ŷᵢ)/yᵢ| × 100%"),

      Pb([
        { t: "其中 yᵢ", b: true },
        " 为第 i 个样本的观测海冰面积，",
        { t: "ŷᵢ", b: true },
        " 为对应的模型预测值，",
        { t: "n", b: true },
        " 为测试集样本总数。所有指标均在归一化逆变换后的原始海冰面积空间（百万平方公里）中计算，直接反映预测误差的物理量级，避免归一化空间中的无量纲数值缺乏直观可解释性的问题。",
      ]),

      // ===== 集成预测 =====
      P("为降低随机初始化种子对模型性能评估的偶然性干扰，每个实验配置独立训练5个不同随机种子（42、52、62、72、82）的模型，取5个种子预测值的算术平均构建集成预测："),

      EqNum("ŷᵉⁿˢ = (1/S) Σₛ₌₁ˢ ŷ⁽ˢ⁾,  S = 5", "2"),

      Pb(["对集成预测结果计算RMSE，下文统称为", { t: "集成RMSE", b: true }, "："]),

      Eq("RMSEᵉⁿˢ = √((1/n) Σᵢ₌₁ⁿ (yᵢ − ŷᵢᵉⁿˢ)²)"),

      Pb([
        "同时报告各独立种子RMSE的均值 ",
        { t: "μʀᵁˡᵉ = (1/S)ΣRMSE⁽ˢ⁾", b: true },
        " 与标准差 ",
        { t: "σʀᵁˡᵉ = √((1/S)Σ(RMSE⁽ˢ⁾ − μʀᵁˡᵉ)²)", b: true },
        "，以反映模型性能对随机初始化的敏感度。",
      ]),

      // ===== 小增益+小样本 困境 =====
      P("本论文面临一个统计推断上的结构性困境：气候指数带来的增量预测增益极为有限——全北极最优单变量ΔRMSE仅约0.015百万平方公里（相对改善~3%），而测试集样本量受限于按年划分的严格约束仅约120个（2016–2025年）。在这一“小增益 + 小样本”的双重约束下，经典的参数化显著性检验（如配对t检验）面临方法论层面的根本困难。配对t检验的检验统计量为："),

      EqNum("t = d̄ / (sᵈ / √n),  dᵢ = RMSEᵢ⁽ᴀ⁾ − RMSEᵢ⁽ᴁ⁾", "3"),

      Pb([
        "其中 ",
        { t: "d̄", b: true },
        " 为两种模型RMSE差异的样本均值，",
        { t: "sᵈ", b: true },
        " 为差异的样本标准差。t检验依赖 d̄ 近似服从正态分布的渐近假设，而 n ≈ 120 的有限样本规模与 |d̄| ≈ 0.015 量级的微弱效应量叠加，使得该假设的可靠性存疑——检验统计量对样本分布尾部行为的敏感度超过了对真实效应存在性的判别能力。换言之，在小样本条件下，经典检验的统计功效（statistical power）可能不足以将真实的微弱信号从采样噪声中分离出来，导致假阴性风险（即气候指数确有贡献但被误判为不显著）显著升高。",
      ]),

      // ===== Bootstrap =====
      P("为应对这一困境，本研究采用Bootstrap非参数重采样方法（Efron & Tibshirani, 1994）评估RMSE差异的统计显著性。Bootstrap的核心思想是利用样本自身的经验分布 F̂ 作为总体分布 F 的近似，通过从原始样本中有放回地重复抽取 B = 10,000 个等规模的Bootstrap样本："),

      Eq("D*ᵇ = {(yᵢ*ᵇ, ŷᵢ*ᵇ)}ᵢ₌₁ⁿ,  b = 1, 2, …, B"),

      P("对每个Bootstrap样本计算RMSE差异："),

      Eq("ΔRMSE*ᵇ = RMSEₘₒᵈᵉₗ*ᵇ − RMSEᵇᵃˢᵉₗᵢₙᵉ*ᵇ"),

      P("由此构建 ΔRMSE 的经验抽样分布，进而通过分位数法估计95%置信区间："),

      EqNum("CI₉₅% = [ΔRMSE*(ᴵ·α/2), ΔRMSE*(ᴵ·(1−α/2))],  α = 0.05", "4"),

      P("该方法的关键优势在于：不预设任何参数化分布形式（如正态性），完全由数据驱动，在小样本条件下仍能为效应量的不确定性提供稳健的量化表征。对于本研究而言，Bootstrap实质上是将“ΔRMSE = −0.015是否为偶然波动”这一核心追问，转化为一个可直接从数据中回答的非参数推断问题——若10,000次重采样生成的95%置信区间不包含零，则即使在增益幅度极为微弱的物理背景下，仍可以为气候指数的增量预测技能具有非偶然性。从更广泛的视角来看，Bootstrap在本论文“小增益、小样本”场景下扮演的角色已超越单纯的显著性检验工具——它是确保全部实验结论的统计可信度不被有限测试样本量所动摇的方法论基础。"),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUTPUT, buf);
  console.log("OK: " + OUTPUT);
  console.log("Size: " + (buf.length / 1024).toFixed(1) + " KB");
});
