# Paper Figures

论文图表生成脚本集。所有脚本输出到 `outputs/plots/paper/`。

## 运行环境

```bash
PYTHONIOENCODING=utf-8 E:/anaconda/envs/pytorch/python.exe <script>.py
```

## 图号对照

| 脚本 | 论文图号 | 内容 | 使用模型 |
|------|:--------:|------|----------|
| `gen_fig_lstm_seasonal.py` | 图3.2 | 单变量LSTM多年逐月平均 ±1σ | `best_model_area.pth` |
| `gen_fig_lstm_short_seasonal.py` | 图3.3 | 短期预测逐月平均 ±1σ | (实时训练) |
| `gen_fig34.py` | 图3.4 | 单变量LSTM最好/最坏预测年 (2020/2016) | `best_model_area.pth` |
| `gen_fig37.py` | 图3.7 | 三种模型逐月RMSE (LR/RNN/LSTM) | `best_model_area.pth` + 实时训练 |
| `gen_fig42.py` | 图4.2 | 双编码器E7v1多年逐月平均 ±1σ | `best_model_de.pth` |
| `gen_fig_e4_seasonal.py` | 图4.3 | 三变量E4多年逐月平均 ±1σ | `paper_e4.pth` |
| `gen_fig_e4_e7_rmse.py` | 图4.4 | E4 vs E7v1 逐月RMSE对比 | `paper_e4.pth` + `best_model_de.pth` |
| `gen_fig_e7_bestworst.py` | 图4.5 | E7双编码器最好/最坏预测年 (2022/2016) | `best_model_de.pth` |
| `gen_fig_lstm_loss.py` | 图3.1 | 单变量LSTM训练损失曲线 | (实时训练) |
| `gen_loss_curves.py` | 图4.1 | E4 vs E7 训练损失曲线（含局部放大） | (实时训练) |
| `gen_fig_loss_schemes.py` | 图3.5 | 短期+中期预测损失曲线 | (实时训练) |
| `gen_data_overview.py` | 图2.1 | 数据概览（时间序列+季节循环+AO/SST） | 无 |
| `gen_acf.py` | 图2.2 | 海冰面积自相关函数 | 无 |
| `gen_scheme_comparison.py` | 图3.6 | 三种预测方案RMSE柱状对比 | CSV结果 |
| `gen_ablation_e7.py` | 图4.6 | E7消融实验（四种设计独立贡献） | 需训练(5变体×5种子) |
| `gen_uncertainty.py` | 图4.7 | 5-seed集成预测不确定性区间 | `best_model_de_s*.pth` |
| `gen_residuals.py` | 图4.8 | 残差分析（QQ图+分布+逐月偏差） | `best_model_de.pth` |

## 绘图规范

所有脚本遵循统一规范（详见根目录 `CLAUDE.md` — Paper Plotting Conventions）：

- **字体:** SimHei（中文），matplotlib Agg 后端
- **标签:** `中文名称 (单位)` 格式，无标题
- **子图标注:** 左上角 `(a)` `(b)`，fontsize=16 bold
- **图例:** 左下角，framealpha=0.9
- **输出:** PNG，DPI=200
- **实测值:** 红色实线 (`#d73027` / `#e74c3c`)
- **预测值:** 蓝/绿虚线（模型而异）
- **日历月分组:** 必须基于 `df_months` 映射，禁止 `mean(axis=0)`
