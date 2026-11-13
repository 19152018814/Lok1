# Prism 9 拟合 IC50 操作单（SM2 + ENR 三块板）

适用：桌面照片上那台装有 GraphPad Prism 9 的电脑。数据来自你仓库 / 原始 Excel 的实测 OD，不是新编的。

## 文件（都在 `data/prism-ic50/`）

| 文件 | 板子 | 条件 |
|---|---|---|
| `SM2-2026-10-31-prism.csv` | SM2，2026-10-31 | 包被 SM2-BSA 0.03 μg/mL，抗体 0.03 μg/mL，50 ng/mL 起二倍稀释 |
| `ENR-2026-11-09-prism.csv` | ENR，2026-11-09 | 板条件未记在文件里，只可从浓度列推断顶标 20 ng/mL |
| `ENR-2026-11-10-prism.csv` | ENR，2026-11-10 | 包被 ENR-BSA 1 μg/mL，抗体 1 μg/mL，20 ng/mL 起二倍稀释 |

每份 CSV 的列：浓度 (ng/mL)、三个复孔 OD、均值、SD、按 IR = 1 − 均值 OD / B0 算的平均抑制率。B0 已在文件头注明，三块板分别是 0.7001 / 1.1902 / 0.8282，和原 Excel 对得上。

## 操作（每块板重复一遍，约 5 分钟）

1. Prism 9 → New Data Table → XY → 选 "Enter replicate values in side-by-side subcolumns"，3 个复孔。
2. X 列粘贴浓度，Y 列粘贴三个复孔 OD（CSV 里第 2–4 列）。**0 ng/mL 那行只作 B0 参考**：log(0) 不存在，拟合时把它排除或留空，否则 Prism 会报错 / 丢点。
3. Analyze → Transform → Transform X values → X = log(X)，底数 10。
4. Analyze → Nonlinear regression (curve fit) → Dose-response – Inhibition → **log(inhibitor) vs. response – Variable slope (four parameters)**。
5. 第一次先不约束 Top / Bottom，让 Prism 自由拟合；看一眼 HillSlope 应为负值（竞争法 OD 随浓度下降），Top 应接近 B0。
6. 结果里读 LogIC50，IC50 = 10^LogIC50；同时记下 R² 和 95% CI，论文图就用这套数。

## 对照：Excel 直线拟合的旧数（用来核对，不用来代替 Prism 结果）

| 板子 | Excel IC50 | 备注 |
|---|---|---|
| SM2 2026-10-31 | 1.47 ng/mL | 中 4 点拟合，R² = 0.9704 |
| ENR 2026-11-09 | 1.08 ng/mL | 中 4 点拟合；**原文件里的 IC50 格 2.53 ng/mL 是错的**，不要抄 |
| ENR 2026-11-10 | 5.00 ng/mL | 中 4 点拟合，R² = 0.9720；两点插值 4.64 ng/mL |

Prism 4PL 和 Excel 直线拟合的数会不一样，这是正常的：4PL 用全部 8 个浓度点、拟合整条曲线，直线法只用了中间 4 个点。论文里报 Prism 的 4PL 结果，并注明方法即可。

## 三个坑，先标出来

- **ENR 11-09 板阴性组漏加了 PBS**，B0 的可信度本身有问题；Prism 拟合得再漂亮也不能把这块板洗白，报数前仍以肖老师的决定为准（这个任务你 11-12 已结束，不再追问）。
- **离群复孔**：ENR 11-09 板 2.5 ng/mL 复孔 2 是 0.4606（该行 CV 14.17%）；ENR 11-10 板 1.25 ng/mL 复孔 3 是 0.6237（该行 CV 12.99%）。建议各拟合两次——一次全数据、一次去掉该复孔——两个 IC50 都记下来，不要悄悄删点只报好看的那个。
- **ENR 11-10 板低浓度端基本没信号**：0.3125 与 0.625 ng/mL 两行 IR 约为 0，B0 只有 0.8282，比 11-09 板的 1.1902 低很多；这是两块板 IC50 差近 5 倍的背景，拟合时不要强行把两条曲线画成一条。
