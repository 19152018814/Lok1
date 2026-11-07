# IMAR 论文资料 — Transforming microdroplets into intelligent robots for autonomous colorimetric sensing

整理时间：2026-11-07；整理：Teng Xiaomi（Lok1 的助手）

## 论文信息（已核实）

- 标题：Transforming microdroplets into intelligent robots for autonomous colorimetric sensing
- 期刊：Nature Sensors, 1(11), pp. 1033–1045 (2026)
- 在线发表：2026-10-05
- DOI：10.1038/s44460-026-00133-0
- 原文：https://www.nature.com/articles/s44460-026-00133-0
- 第一作者：Zongliang Guo（郭宗良），北京理工大学集成电路与电子学院
- 通讯作者：Rongxin Fu（符荣鑫，北理工医学科学与工程学院）、Shuailong Zhang（张帅龙，北理工集成电路与电子学院）
- 合作单位：西湖大学、香港中文大学（深圳）、山东大学、北京大学、剑桥大学、格拉斯哥大学等
- 相关评述：Zhang, Y., Wang, L. "AI transforms colorimetry from craft to intelligence", Nature Sensors (2026), DOI: 10.1038/s44460-026-00140-1，2026-10-05 发表

## 这篇论文是做什么的（依据论文官网公开的扩展数据说明、代码仓库 README）

- 系统名 IMAR（Intelligent Microdroplet Analytical Robotics）：数字微流控（DMF）芯片 + 机器视觉 + AI 闭环反馈，把传统比色检测变成自主、自适应的传感系统。
- 流程：背景扣除图像增强 → YOLO 检测液滴 → CNN 回归推断浓度；手机 App 拍照 → 电脑视觉定位液滴、规划无碰撞路径 → 控制器驱动液滴。
- 芯片结构：玻璃基底 + TFT 像素电极 + 绝缘层 + 疏水层。
- 宣称试剂用量比传统方法少约 100,000 倍（据 bioengineer.org 报道）。
- 验证过的检测：葡萄糖（Trinder 法，1–8 mmol/L）、H2O2、Fe³⁺（R²=0.963）、pH（R²=0.990，分类准确率 97.6%）、蛋白（Coomassie 亮蓝，R²=0.980）、水中铜离子（瓶装水 R²=0.956、自来水 0.949、湖水 0.911）；迁移学习可跨比色体系（葡萄糖模型从 H2O2、铜、pH、亚硝酸盐、钙等迁移后，R² 均高于从头训练的 0.898）。

## 拿不到全文的说明

- 摘要和正文在 Nature 付费墙后，本次只能看到页面公开部分（作者、参考文献、扩展数据图注、数据/代码声明）。全文需机构订阅或购买（39.95 欧元），或联系通讯作者索取。

## 本仓库里的内容

- `papers/IMAR_system/`：论文开源代码仓库 https://github.com/Aneage/IMAR_system 的完整拉取（2026-11-07，含代码、模型权重、演示数据），内容包括：
  - `infer.py` / `server.py` / `batch_test.py`：推理、HTTP 服务、批量测试
  - `training/`：YOLO 训练、CNN 训练、迁移学习、图像处理管线
  - `path planning/`：液滴路径规划（SIPP 等）
  - `models/`：YOLO 权重 best2.pt（5.4 MB）、Trinder-1 CNN 权重 best_model.pth（14 MB）
  - `demo_data/`：演示用原图 + 背景图各一张
  - `expected_output/`、`batch_results/`：参考输出与批量测试结果
  - `requirements.txt`：Python 3.10–3.12，PyTorch（默认 CPU 版）
- 跑演示：`python infer.py --original demo_data/original.jpg --background demo_data/background.jpg`（详见 `papers/IMAR_system/README.md`）
- 版权说明：代码版权归原作者所有，本仓库仅作学习备份；原仓库地址见上。

## 和 Lok1 实验的关联（推断，非论文原文）

- 都是比色法定量 + 图像分析：IMAR 用 CNN 回归读液滴颜色浓度；Lok1 的 LFIA 用 RGB/灰度读 T 线。IMAR 的自适应掩膜、多样本策略、迁移学习思路值得参考。
