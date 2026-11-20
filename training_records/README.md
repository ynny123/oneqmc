# 已完成算例的训练记录

这里收录已完成的从头训练记录。LiCN、H₄ 和乙烷目录包含日志、指标、曲线和运行配置。H₂ 和硼原子 B 目录保存了完整的能量和绝对误差 CSV、PNG 曲线及校验清单。

| 算例 | seed / GPU | 完成步数 | 训练日志 | 完整曲线 CSV | 曲线 PNG |
| --- | --- | --- | --- | --- | --- |
| LiCN，5 geometries | 42 / 1×A100 80GB | 20,000 | [日志](licn_default_1gpu_20k_s42/results/oneqmc_train.log) | [MARE](licn_default_1gpu_20k_s42/results/analysis/mare_online.csv) | [MARE 图](licn_default_1gpu_20k_s42/results/analysis/mare_online.png) |
| H₄ rectangles，6 geometries | 42 / 1×A100 80GB | 16,000 | [日志](h4_globe6_scratch_1gpu_16k_s42/results/oneqmc_train.log) | [绝对能量](h4_globe6_scratch_1gpu_16k_s42/results/analysis/h4_convergence.csv) | [能量图](h4_globe6_scratch_1gpu_16k_s42/results/analysis/h4_convergence.png) |
| 乙烷 BBMEP，20 geometries | 219 / 4×A100 80GB | 32,000 | [日志](ethane_scratch_4gpu_32k_s219/results/oneqmc_train.log) | [MARE](ethane_scratch_4gpu_32k_s219/results/analysis/mare_online.csv) | [MARE 图](ethane_scratch_4gpu_32k_s219/results/analysis/mare_online.png) |
| H₂，1 geometry | 42 / 1×A100 80GB | 10,000 | — | [能量与绝对误差](h2_equilibrium_scratch_1gpu_10k_s42/results/analysis/absolute_energy_error_online.csv) | [能量](h2_equilibrium_scratch_1gpu_10k_s42/results/analysis/energy_online.png) · [绝对误差](h2_equilibrium_scratch_1gpu_10k_s42/results/analysis/absolute_energy_error_online.png) |
| 硼原子 B，1 geometry | 42 / 1×A100 80GB | 10,000 | — | [能量与绝对误差](b_atom_scratch_1gpu_10k_s42/results/analysis/absolute_energy_error_online.csv) | [能量](b_atom_scratch_1gpu_10k_s42/results/analysis/energy_online.png) · [绝对误差](b_atom_scratch_1gpu_10k_s42/results/analysis/absolute_energy_error_online.png) |

LiCN 最终在线 EWM MARE 为 **0.11681 kcal/mol**，乙烷为 **0.59324 kcal/mol**。H₄ 最后 200 步的六结构平均绝对能量为 **−2.01615644 Ha**，不减最低能量。MARE 是训练过程的在线统计，不是固定 checkpoint 的独立采样评估。

## 文件内容

每个算例目录中的 `results/analysis/` 保存完整曲线和逐结构能量；`results/training/config.yaml` 是实际训练配置，`results/runtime_status.json` 记录完成状态与检查结论，`logs/` 或 `results/stdout_stderr.log` 保存标准输出。根目录还保留当时的运行协议、环境版本、冻结源码哈希和退出码（以各目录实际文件为准）。

[1/2/4 小时指标 CSV](time_budget_comparison/time_budget_metrics.csv)、[指标来源与计时口径](time_budget_comparison/time_budget_metrics_metadata.json)，以及逐步时间映射均在 `time_budget_comparison/`。时间包含进程启动后的初始化、平衡和编译，不含调度排队；H₄ 不足 2 小时完成，2、4 小时栏使用其最终结果。

`time_budget_comparison/export_time_budgets.py` 保留了生成时间截面的脚本。重新执行需要本机 `runs/<算例>/` 中的原始 HDF5 和 TensorBoard 文件；仅克隆仓库即可查看已导出的指标和时间映射，无需执行该脚本。

## 校验与未上传内容

每个算例的 `manifest.json` 记录发布文件的大小、SHA-256，以及是否替换过机器路径。仓库的 [SHA256SUMS](SHA256SUMS) 可校验所有已发布训练记录；从仓库根目录执行：

```bash
(cd training_records && sha256sum -c SHA256SUMS)
```

公开文本将机器绝对路径替换成 `${RUN_DIR}`、`${FROZEN_CODE_DIR}`、`${REPO_DIR}`、`${PROJECT_DIR}` 占位符，指标数值保持不变。日志中提及 checkpoint 文件名或重载验证结论，只说明原训练的行为，不表示权重已上传。

**不上传 checkpoint、模型权重（含 LAC）、HDF5（`.h5` / `.hdf5`）、TensorBoard 事件文件、包含这些文件的压缩包、平台回执或凭据。** 原始大文件和 checkpoint 仍保留在本机；公开记录可用于检查训练过程与指标，不能用于恢复模型参数。
