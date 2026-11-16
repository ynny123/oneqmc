# 甲酰胺四卡从头训练示例

任务为 Formamide 的 N–H 键均裂：`HCONH2 → HCONH· + H·`。输入 `data/bbmep/Formamide/00.yaml` 至 `19.yaml` 是路径上的 20 个固定核构型，每个包含 6 个原子、24 个电子（12 up / 12 down）。核坐标单位为 Å。

正式配置为单节点 4 张 A100 80GB、单 Python 进程、JAX `pmap`。全局每步选取 4 个 geometry，每卡 1 个；每个 geometry 采样 256 个完整电子构型，全局 batch 为 1024。全部 20 个结构共享一套 Orbformer-SE 参数。训练目标 32,000 次全局 K-FAC 更新，seed 219；随机初始化、不加载预训练权重，HF 预训练 0 步。

本次先运行 32k 步较短基线；论文列出的甲酰胺多结构实验训练长度为 100k 步。

完整配置在 [setting.json](setting.json)：基础学习率 0.05、damping 0.001、norm constraint 0.001、16 个行列式、每核 8 个包络；stack + Metropolis、去相关 60 步、正式初始平衡上限 500 步。快速 checkpoint 间隔 101 步，阶段 checkpoint 间隔 8000 步。

环境安装沿用[乙烷运行说明](../../docs/ETHANE_SCRATCH_RUNBOOK_zh.md#3-环境准备)的 Python 3.11 / JAX 0.4.34 环境。在仓库根目录、激活该环境后运行：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export ORBFORMER_PICKLE_LOADING=1
mkdir -p logs

# 先预览，既不启动计算也不创建输出目录。
python examples/formamide_scratch/run.py --mode train --devices 4 \
  --output runs/formamide_example_new/results --dry-run

# 单卡运行检查：默认 2 步，只使用 00.yaml。
CUDA_VISIBLE_DEVICES=0 python examples/formamide_scratch/run.py \
  --mode smoke --output runs/formamide_smoke/results \
  > logs/formamide_smoke.log 2>&1

# 正式四卡训练；必须使用新的输出目录。
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/formamide_scratch/run.py \
  --mode train --devices 4 --output runs/formamide_example_new/results \
  > logs/formamide_example_new.log 2>&1
```

调度器已设置 `CUDA_VISIBLE_DEVICES` 时保留分配值。不需要 `torchrun` 或四个 Python 进程。运行检查只验证程序、有限能量、参数更新和检查点重载；不能用它判断训练精度。已有运行以同一配置增加 `--resume` 续训到总计 32,000 步。

主要观察整条 20 点路径的 **MARE 曲线**。参考能量保存在 `data/references/formamide_bbmep_reference_energy.csv`，只用于后处理，不参与训练损失。可在另一终端启动自动导出：

```bash
python examples/formamide_scratch/mare.py \
  --run-dir runs/formamide_example_new/results --watch
```

每分钟刷新 `results/analysis/mare_online.csv`、`mare_online_latest.json` 和 `energy_online_latest.csv`。同一输出目录只开一个导出器；不带 `--watch` 时导出一次快照。等全部 20 个 geometry 均有有效能量后，才能计算完整路径 MARE。

绘图时执行：

```bash
python examples/formamide_scratch/plot_mare.py \
  --run-dir runs/formamide_example_new/results
```

得到 `results/analysis/mare_online.png` 和 `.pdf`。图片是快照，训练继续后重新执行绘图命令。MARE 单位为 kcal/mol，定义为去除整体能量偏移后的相对误差；在线 EWM 曲线用于观察训练，不是独立固定 checkpoint 评估。

若需固定最终模型进行独立评估，使用新的输出目录重新平衡采样器；以下为额外评估成本，本次提交任务只执行训练和在线导出：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/formamide_scratch/run.py \
  --mode evaluate --devices 4 --steps 30000 \
  --checkpoint runs/formamide_example_new/results/training/chkpt-32000.pt \
  --output runs/formamide_evaluation_new/results \
  > logs/formamide_evaluation_new.log 2>&1
python examples/formamide_scratch/mare.py \
  --run-dir runs/formamide_evaluation_new/results --kind evaluation
```

30,000 个四卡全局评估步骤平均对应每个结构约 6000 次访问，实际次数以日志为准。入口检查模型参数与输入检查点保持不变；后处理生成 `analysis/mare_evaluation.json` 和 `energies_evaluation.csv`，使用 Huber 稳健能量估计后计算完整路径 MARE。

当前提交任务的输出位置、阶段状态及其他算例对照见[训练示例总说明](../../docs/TRAINING_EXAMPLES_zh.md)。GitHub 只包含代码、说明、结构和参考输入；checkpoint、训练日志及生成结果保留在运行机器。
