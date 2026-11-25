# Orbformer 多结构 A100 算子与阶段耗时

三组算例沿用原训练的 GPU 数、batch、MCMC 采样和 K-FAC 设置，从随机初始化开始做短程 profiling。LiCN 和 H₄ 分别使用本机单张 A100 80 GB，乙烷 BBMEP 使用远程单节点四张 A100 80 GB。JAX 版本为 0.4.34。

| 算例 | GPU 数 | 数据集结构数 | 每步结构数 × 每结构电子构型数 | Seed |
| --- | ---: | ---: | ---: | ---: |
| LiCN | 1 | 5 | 1 × 2,048 | 42 |
| H₄ | 1 | 6 | 6 × 256 | 42 |
| 乙烷 BBMEP | 4 | 20 | 4 × 256 | 219 |

初始采样平衡后热身 35 步，用 JAX profiler 和 CUPTI 分两个窗口采集第 **36–40 步**和第 **41–45 步**。统计读取完整 XPlane 记录，并从训练 HDF5 的 `metrics/mol_idx` 核对逐步 geometry ID。LiCN 和乙烷在两个窗口内覆盖了所有结构，每个结构出现 1–3 次。H₄ 的 6 个结构每步都参与。结果按实际采样窗口汇总。[原训练设置](../../docs/TRAINING_EXAMPLES_zh.md)

## 训练阶段耗时

两张表均以窗口内**各 GPU 执行的全部 kernel 的累计耗时**为分母，包含 NCCL kernel。数据拷贝与 memset 归入数据移动，主机计时单独列出。乙烷的总耗时是四卡的 kernel 时间之和，逐卡计时表同时保留每张卡的累计耗时与去重后的 GPU 活跃时间。

| 算例 | GPU kernel 累计耗时（ms/步） | MCMC 采样 | 局部能量（含拉普拉斯） | K-FAC 更新 | 采样器刷新 | 其他工作 / 未归属 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LiCN | 894.2 | 33.11% | 41.11% | 23.47% | 2.04% | 0.27% |
| H₄ | 277.6 | 41.50% | 8.10% | 48.95% | 0.66% | 0.78% |
| 乙烷 BBMEP | 1,183.2（四卡合计） | 39.91% | 27.46% | 25.32% | 6.43% | 0.88% |

LiCN 的局部能量计算耗时最高，为 **367.6 ms/步**。H₄ 的 K-FAC 更新耗时最高，为 **135.9 ms/步**。四卡乙烷的 MCMC 采样耗时最高，合计 **472.3 ms/步**，各卡全部 kernel 累计耗时在 **280.3–302.0 ms/步**之间。阶段 CSV 分别保存其他训练工作和未归属耗时，也保留主机侧阶段标记的时间跨度。[阶段汇总 CSV](phase_timing_comparison.csv) · [逐卡计时 CSV](device_timing.csv)

阶段归属优先用 CUPTI 关联 ID 和 kernel 名称匹配主线程的阶段标记。缺少直接匹配时，使用可明确对应训练阶段的 XLA 模块标注。对仍缺少标签的 kernel，只有在同一 CUDA stream 上被同一步、同一阶段的两个已归属 kernel 夹住时才推断归属。三组分别保留 **0.17%、0.47%、0.79%** 的 GPU 耗时为未归属。各算例的 `summary.json` 记录直接匹配、XLA 模块匹配和推断覆盖情况，以及将部分已知标签隐藏后的验证结果。[训练阶段入口](../../src/oneqmc/fit.py)

## 算子类别耗时

| 算例 | GEMM | LU 分解 | 三角求解 | 边注意力 MHSEA | NCCL | 其他 kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LiCN | 32.72% | 7.51% | 2.11% | 6.43% | 0.00% | 51.24% |
| H₄ | 25.05% | 19.76% | 5.93% | 10.33% | 0.00% | 38.94% |
| 乙烷 BBMEP | 21.76% | 5.99% | 1.56% | 9.20% | 11.29% | 50.21% |

算子类别按 GPU kernel 名称划分。乙烷 NCCL kernel 四卡累计 **133.6 ms/步**，其中 **66.5 ms/步**归入 MCMC 采样，**66.9 ms/步**归入 K-FAC。逐算子明细保留 trace 提供的 HLO 模块与算子名称，可用于查看编译后 kernel 与 XLA 算子的对应关系。[算子汇总 CSV](kernel_family_comparison.csv) · [乙烷阶段 × 算子类别 CSV](ethane_bbmep/phase_kernel_family.csv)

## 逐算例明细

| 算例 | GPU kernel 与 XLA 算子 | 阶段与结构覆盖 | 采集摘要 |
| --- | --- | --- | --- |
| LiCN | [kernel](licn/kernel_totals.csv) · [HLO](licn/xla_op_totals.csv) · [kernel–HLO 对应](licn/kernel_xla_ops.csv) | [逐步阶段](licn/phase_per_update_by_device.csv) · [阶段 × 算子类别](licn/phase_kernel_family.csv) · [结构覆盖](licn/geometry_coverage.csv) | [配置](licn/protocol.json) · [摘要](licn/summary.json) |
| H₄ | [kernel](h4/kernel_totals.csv) · [HLO](h4/xla_op_totals.csv) · [kernel–HLO 对应](h4/kernel_xla_ops.csv) | [逐步阶段](h4/phase_per_update_by_device.csv) · [阶段 × 算子类别](h4/phase_kernel_family.csv) · [结构覆盖](h4/geometry_coverage.csv) | [配置](h4/protocol.json) · [摘要](h4/summary.json) |
| 乙烷 BBMEP | [kernel](ethane_bbmep/kernel_totals.csv) · [HLO](ethane_bbmep/xla_op_totals.csv) · [kernel–HLO 对应](ethane_bbmep/kernel_xla_ops.csv) · [逐卡 kernel](ethane_bbmep/kernel_by_device.csv) | [逐卡阶段](ethane_bbmep/phase_by_device.csv) · [逐步逐卡阶段](ethane_bbmep/phase_per_update_by_device.csv) · [结构覆盖](ethane_bbmep/geometry_coverage.csv) | [配置](ethane_bbmep/protocol.json) · [摘要](ethane_bbmep/summary.json) |

[H₂、B、CH₄ 和 C₂H₆ 单结构 profiling](../orbformer_a100_profiling/README.md)
