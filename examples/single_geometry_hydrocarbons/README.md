# 甲烷和乙烷单结构训练与评估

每个分子独立训练一套 Orbformer-SE 参数，核坐标在训练和评估中固定。每套算例使用单卡，从随机初始化训练 50,000 步，seed 42。训练完成后自动固定最终 checkpoint，以 seed 20261119 初始化新采样链并评估 1,000 步。训练和评估每步均为 1 个结构和 1,024 个电子构型。

| 分子 | 固定结构 | 全电子参考能量 |
| --- | --- | ---: |
| CH₄，10 电子 | [OneQMC 平衡构型](../../data/CH4/CH4.yaml) | −40.51454 Ha |
| C₂H₆，18 电子 | [乙烷 BBMEP 的第 0 帧](../../data/C2H6_bbmep_00/00.yaml) | −79.82484961293652 Ha |

甲烷参考值来自 Psiformer 论文。乙烷坐标取自 BBMEP 的 `00.yaml`，参考值来自同一结构的单点 Psiformer 计算，统计误差约 1.64 mHa。解读乙烷的绝对误差时应同时参考该不确定度。[Psiformer 论文](https://arxiv.org/html/2211.13672) · [Orbformer 补充表 7](https://arxiv.org/html/2506.19960v2#A9.T7)

## 运行

在仓库根目录激活已安装的 OneQMC 环境，再执行下列命令。每条命令都会完成训练、独立评估和曲线导出。每次运行使用新的输出目录。首次检查可在命令末尾加 `--smoke`，各执行 2 步训练和评估。

```bash
source examples/ethane_scratch/env.sh
export CUDA_VISIBLE_DEVICES=0

bash examples/single_geometry_hydrocarbons/run.sh ch4 runs/ch4_single_50k
bash examples/single_geometry_hydrocarbons/run.sh c2h6 runs/c2h6_single_50k
```

## 输出

| 相对运行目录的路径 | 内容 |
| --- | --- |
| `fit/oneqmc_train.log` | 训练日志，包含加载的结构数和更新进度 |
| `fit/training/chkpt-50000.pt` | 最终训练 checkpoint |
| `fit/analysis/` | 每分钟更新的能量图、绝对误差图和逐步 CSV |
| `evaluation/oneqmc_train.log` | 固定模型的采样平衡和评估进度 |
| `evaluation/analysis/evaluation.json` | 独立评估的总能量（Ha）和绝对误差（mHa） |
| `training_wall_seconds.txt`、`evaluation_wall_seconds.txt` | 训练和评估的进程耗时（秒），各自包含初始化、采样平衡和编译 |

训练曲线使用局部能量的指数加权平均值。最终评估对 1,000 步的批平均局部能量取 Huber 稳健均值（`delta=1 Ha`），再计算与参考值的绝对误差。

## 已归档结果

甲烷在单张 A100 80 GB 上完成 50,000 步训练，耗时 **7 小时 27 分 26 秒**。随后评估 1,000 步，耗时 **7 分 26 秒**，评估能量为 **−40.514028 Ha**，绝对误差为 **0.512 mHa**。[训练 CSV 与曲线](../../training_records/ch4_single_scratch_1gpu_50k_s42/results/fit/analysis) · [评估结果](../../training_records/ch4_single_scratch_1gpu_50k_s42/results/evaluation/analysis/evaluation.json) · [完整归档](../../training_records/ch4_single_scratch_1gpu_50k_s42)

乙烷在单张 A100 80 GB 上完成 50,000 步训练，耗时 **13 小时 46 分 14 秒**。随后评估 1,000 步，耗时 **15 分 15 秒**，评估能量为 **−79.823266 Ha**，相对参考值的绝对误差为 **1.583 mHa**。[训练 CSV 与曲线](../../training_records/c2h6_single_scratch_1gpu_50k_s42/results/fit/analysis) · [评估结果](../../training_records/c2h6_single_scratch_1gpu_50k_s42/results/evaluation/analysis/evaluation.json) · [完整归档](../../training_records/c2h6_single_scratch_1gpu_50k_s42)

同样的单卡和 batch 设置还用于短程 profiling。统计覆盖第 33–44 步，提供各训练阶段、GPU kernel 和可归属的 XLA 算子耗时。[A100 算子与阶段统计](../../training_records/orbformer_a100_profiling/README.md)
