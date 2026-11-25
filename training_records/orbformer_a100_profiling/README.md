# Orbformer 单卡 A100 算子与阶段耗时

H₂、B、CH₄ 和 C₂H₆ 各用单张 A100 80 GB，从随机初始化开始运行 Orbformer-SE 和 K-FAC。每步处理 **1 个固定结构 × 1,024 个电子构型**，与各自的正式训练一致。JAX 版本为 0.4.34，seed 为 42。完成初始采样平衡和 32 步热身后，用 JAX profiler 和 CUPTI 采集第 **33–44 步**的 GPU 活动。统计来自完整的 XPlane 记录，每组都核对了 12 步的阶段标记和 GPU kernel 事件。[训练阶段入口](../../src/oneqmc/fit.py)

## 算子类别耗时

占比以采样窗口内**全部 GPU 计算 kernel 的累计执行时间**为分母。数据拷贝和 memset 计入数据移动，主机计时另列。平均每步累计时间由 12 步的 kernel 耗时之和除以 12。

| 算例 | GPU kernel 平均累计耗时（ms/步） | GEMM | LU 分解 | 三角求解 | 边注意力 MHSEA | 其他 kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H₂ | 204.4 | 23.27% | 26.72% | 8.05% | 8.83% | 33.13% |
| B | 236.1 | 25.02% | 23.48% | 7.02% | 8.68% | 35.80% |
| CH₄ | 362.8 | 29.45% | 15.89% | 4.67% | 6.66% | 43.33% |
| C₂H₆ | 816.1 | 26.16% | 8.00% | 2.26% | 11.56% | 52.01% |

类别按 GPU kernel 名称匹配 `gemm`、`getrf`、`trsm` 或 `triangular_solve`，以及 `mhsea` 前缀。乙烷的 MHSEA forward Laplacian kernel 单独占全部 GPU 计算时间的 **7.61%**。XLA 可能把多个操作融合进同一个 kernel，逐算子文件保留 trace 可用的 HLO 模块和算子名称。[类别汇总 CSV](kernel_family_shares.csv) · [分类后的逐 kernel 明细 CSV](kernel_family_members.csv)

## 训练阶段耗时

每格依次列出平均 GPU kernel 累计耗时（ms/步）和占比。阶段占比沿用上表的分母。

| 算例 | MCMC 采样 | 局部能量（含拉普拉斯） | K-FAC 参数更新 | 采样器刷新 | 其他工作 / 未归属 |
| --- | ---: | ---: | ---: | ---: | ---: |
| H₂ | 74.0 / 36.20% | 6.6 / 3.21% | 120.6 / 59.01% | 1.2 / 0.57% | 2.1 / 1.01% |
| B | 86.6 / 36.69% | 19.7 / 8.32% | 126.5 / 53.56% | 1.3 / 0.55% | 2.1 / 0.88% |
| CH₄ | 128.1 / 35.31% | 73.2 / 20.18% | 157.5 / 43.40% | 2.1 / 0.57% | 2.0 / 0.55% |
| C₂H₆ | 255.3 / 31.28% | 335.7 / 41.14% | 217.6 / 26.66% | 4.2 / 0.51% | 3.4 / 0.41% |

H₂、B、CH₄ 的 GPU 耗时主要集中在 K-FAC 更新。乙烷的局部能量计算耗时最高，平均每步 **335.7 ms**，占 **41.14%**。[阶段汇总 CSV](phase_timing_summary.csv)

阶段归属先用 CUPTI 关联 ID 和 kernel 名称匹配主线程的阶段标记。对于未直接匹配的事件，只有在同一 CUDA stream 上被同一步、同一阶段的两个已匹配 kernel 夹住时才推断归属。直接匹配覆盖各组 **69.87%–85.53%** 的 GPU 计算时间，推断补齐 **13.84%–29.81%**，仍有 **0.32%–0.63%** 保留为未归属。表中最后一列合并其他训练工作与未归属时间，CSV 分别记录。每组的归属校验还记录了将部分已匹配事件暂时隐藏后的抽样检查结果。

## 逐算例文件

| 算例 | GPU kernel / XLA 明细 | 逐步阶段耗时 | 阶段归属校验 |
| --- | --- | --- | --- |
| H₂ | [kernel](h2/kernel_totals.csv) · [HLO](h2/xla_op_totals.csv) · [kernel–HLO 对应](h2/kernel_xla_ops.csv) | [CSV](h2/phase_per_update.csv) | [JSON](h2/phase_attribution_validation.json) |
| B | [kernel](b/kernel_totals.csv) · [HLO](b/xla_op_totals.csv) · [kernel–HLO 对应](b/kernel_xla_ops.csv) | [CSV](b/phase_per_update.csv) | [JSON](b/phase_attribution_validation.json) |
| CH₄ | [kernel](ch4/kernel_totals.csv) · [HLO](ch4/xla_op_totals.csv) · [kernel–HLO 对应](ch4/kernel_xla_ops.csv) | [CSV](ch4/phase_per_update.csv) | [JSON](ch4/phase_attribution_validation.json) |
| C₂H₆ | [kernel](c2h6/kernel_totals.csv) · [HLO](c2h6/xla_op_totals.csv) · [kernel–HLO 对应](c2h6/kernel_xla_ops.csv) | [CSV](c2h6/phase_per_update.csv) | [JSON](c2h6/phase_attribution_validation.json) |

各目录也保存运行参数 `protocol.json`、逐步 GPU 计时 `per_update.csv`、阶段平均值 `phase_timing.csv` 和采集摘要 `summary.json`。`host_scope_ms` 是主机侧阶段标记的时间跨度，反映主机调度和等待等因素。这一列独立于 GPU kernel 累计时间。
