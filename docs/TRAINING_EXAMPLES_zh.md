# Orbformer 训练示例：LiCN 单卡、H₄ 单卡与乙烷四卡

本文按 **LiCN → H₄ → 乙烷** 的顺序介绍三套实验的物理体系、训练设置、输出位置和主要指标。前两套作为单卡示例，乙烷作为单节点四卡示例。本文中的 `runs/...` 路径均相对于仓库根目录。这些输出用于说明原机器的实验布局；GitHub 交付只包含代码、结构和参考能量输入，不包含 checkpoint、日志、评估结果或运行目录。

三套实验都使用完整的 **Orbformer-SE 从头训练**：随机初始化、不加载 LAC 或其他外部 checkpoint、Hartree–Fock（HF）预训练为 0 步。训练通过变分蒙特卡洛（VMC）最小化电子能量；LiCN 和乙烷的参考能量仅用于统计误差，不作为监督标签。

## 1. 总览与共同设置

| 项目 | LiCN：单卡示例 | H₄(6)：单卡示例 | 乙烷：四卡示例 |
| --- | --- | --- | --- |
| GPU | 1 × A100 80GB | 1 × A100 80GB | 4 × A100 80GB，单节点 |
| 训练 geometry 总数 | 5 | 6 | 20 |
| 每个 geometry 的原子 / 电子数 | 3 / 16 | 4 / 4 | 8 / 18 |
| 上 / 下自旋电子数 | 8 / 8 | 2 / 2 | 9 / 9 |
| 每步全局 geometry batch | **1** | **6** | **4**，每卡 1 个 |
| 每个 geometry 的电子构型数 | **2048** | **256** | **256** |
| 每步全局电子构型数 | 2048 | 1536 | 1024 |
| 目标参数更新数 | **20,000** | **16,000** | **32,000** |
| seed | 42 | 219 | 219 |
| 模型参数量 | 3,030,618 | 3,026,730 | 3,029,650 |
| 阶段 checkpoint 间隔 | 10,000 步 | 2000 步 | 8000 步 |
| 主要观察指标 | **MARE 曲线** | **能量收敛曲线** | **MARE 曲线** |
| 编写时状态（2026-11-16） | 已启动，训练中 | 已完成 16k 步 | 已完成 32k 步 |

**geometry 是固定的一组核坐标；电子构型是一组完整的电子坐标。** 例如 LiCN 的 `1×2048` 表示每次更新选取 1 个核构型，对该结构采样 2048 组、每组含 16 个电子的坐标。数据集仍有 5 个 geometry，并非只训练其中一个。本文步数均为全局优化器更新次数，四卡的 32k 步不再乘以 4。

共同训练设置如下，三套实验的完整实际参数以各自 `results/training/config.yaml` 为准：

| 设置 | 三套实验共同取值 |
| --- | --- |
| 模型 / 参数共享模式 | `orbformer-se` / `chem-pretrain` |
| 初始化 | 随机参数；`chkpt=None`；`pretrain_steps=0` |
| 优化器 | K-FAC；基础 learning rate 0.05；damping 0.001；norm constraint 0.001 |
| 波函数 | 16 个行列式；每个原子核 8 个包络 |
| 采样器 | `stack` 多体系采样器 + Metropolis |
| 采样设置 | `decorr_steps=60`；正式训练初始平衡最多 500 步；`repeated_sampling_len=1` |
| 能量处理 | `clip_width=5.0`；`det_penalty_weight=0.001` |
| 数值设置 | float32；matmul precision `high`；Flash Attention 开启；forward Laplacian |
| 快速 checkpoint 间隔 | 101 步 |
| 几何划分 | 各自所有 geometry 全部参与训练，共享一套模型参数；未设置未见 geometry 测试集 |

`chem-pretrain` 是上游共享 orbital generator 的模式名称，本身不表示加载了预训练权重。K-FAC 的 0.05 是基础学习率；实际使用 `InverseScheduleLinearRamp(0.05, train_steps // 2, 250)`，有预热和衰减。因此缩短训练步数也会改变学习率计划，短跑检查不能直接接作正式训练的前缀。

源码基线为 `46ddc5a239650672a3b9eb276b292d3326c79ad5`。已运行环境为 Linux、Python 3.11、JAX 0.4.34、A100 80GB；安装步骤和依赖版本见[环境准备说明](ETHANE_SCRATCH_RUNBOOK_zh.md#3-环境准备)及[依赖文件](../requirements/ethane-a100-py311.txt)，同一环境可运行三个示例。

## 2. LiCN：单卡、多几何、默认 batch 的从头训练

### 2.1 算例内容

输入为 [`data/licn/0.yaml` 至 `4.yaml`](../data/licn/0.yaml)。每个结构由 C、Li、N 三个原子组成，为中性、16 电子体系。三个原子共线：C 位于原点，N 位于 `(1.16213, 0, 0)` Å，Li 位于 `(-R, 0, 0)` Å。

| geometry 文件 | Li–C 距离 R（Å） | 参考能量（Hartree） |
| --- | --- | --- |
| `0.yaml` | 1.9 | -100.3792953491211 |
| `1.yaml` | 2.0 | -100.37855529785156 |
| `2.yaml` | 2.1 | -100.37504577636719 |
| `3.yaml` | 2.3 | -100.36514282226562 |
| `4.yaml` | 2.5 | -100.35298156738281 |

这五个点用于学习 Li–C 距离变化时的能量曲线。参考值来自上游 `experiment_results/04_licn/energy.csv` 中 `dataset == Reference` 的五行；交付仓库将其保存在 [`data/references/licn_reference_energy.csv`](../data/references/licn_reference_energy.csv)，每行附有上游固定版本来源。当前任务根目录也保留了相同的 `reference_energy.csv`。这里沿用官方参考值，不将其称为精确基态能量。

### 2.2 训练设置与启动

单卡运行 **1 个 geometry × 2048 个电子构型**，K-FAC 训练 **20,000 步**，seed **42**。五个 geometry 轮换参与训练，每个 geometry 平均参与约 4000 次更新。此任务采用上游默认 batch；此前验证过的 `5×256` 是另一种配置，不是本任务的设置。

当前任务开始前另做了 25 步短跑检查，验证五个结构均被访问、能量有限、参数更新及 checkpoint 重载。短跑仅将初始平衡上限改为 1 步；正式训练保留 500 步上限，并重新随机初始化。

在已安装的环境中，从仓库根目录启动一套同参数的新训练：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export ORBFORMER_PICKLE_LOADING=1
mkdir -p logs

# 使用新的输出目录，避免上游 autoresume 自动读取已有检查点。
CUDA_VISIBLE_DEVICES=0 python scripts/transferable.py \
  -d licn --mol-batch-size 1 --electron-batch-size 2048 \
  --train-steps 20000 --seed 42 --pretrain-steps 0 \
  -w runs/licn_example_new/results > logs/licn_example_new.log 2>&1
```

其余设置沿用上游默认值。此原生命令启动训练；MARE 导出属于独立后处理，当前已提交任务的包装脚本额外配置了自动导出，见下节。当前机器若已通过调度器分配单卡，应保留调度器设置的 `CUDA_VISIBLE_DEVICES`。

### 2.3 当前输出位置与应看指标

实验根目录：`runs/licn_default_1gpu_20k_s42/`（`runs/licn_default_1gpu_20k_s42/`，原机器输出）。

| 文件或目录（相对于该根目录） | 内容与用途 |
| --- | --- |
| `protocol.json`、`source_sha256.json`、`code/` | 固定的实验协议、文件哈希、实际运行的源码和输入副本 |
| `reference_energy.csv` | 上表五个参考能量及其来源 |
| `results/oneqmc_train.log` | 当前 IDE 打开的训练日志；查看初始化、平衡和 `Progress` |
| `logs/stdout_stderr.log` | 标准输出、错误和编译信息 |
| `results/runtime_status.json` | 运行阶段、设备及目标步数；完成时记录重载验证结果 |
| `results/training/config.yaml` | 实际解析的全部上游配置 |
| `results/training/result.h5` | 逐次更新的 geometry 索引、能量及其他已记录指标 |
| `results/training/events.out.tfevents.*` | TensorBoard 标量，包括本任务原生记录的参考误差指标 |
| `results/training/chkpt-10000.pt`、`chkpt-20000.pt` | 达到对应步数后生成的阶段 / 最终 checkpoint |
| `results/analysis/mare_online.csv` | **主要指标数据：MARE 随训练步数变化** |
| `results/analysis/mare_online_latest.json` | 最近导出的步数、MARE 和导出时间 |
| `results/analysis/energy_online_latest.csv` | 每个 geometry 最近的 EWM 能量、更新时间步和参考能量 |
| `logs/mare_export.log` | 自动导出器日志；MARE 不刷新时检查此处 |
| `exitcode_remote.txt` | 训练进程结束后生成，正常退出为 0 |

**主要看 MARE 是否随训练降低，以及后期是否进入稳定区间。** `mare_online.csv` 的横轴列为 `completed_updates`，纵轴列为 `mare_kcal_mol`；`step` 是从 0 开始的原生索引，比已完成更新数小 1。计算方法见第 5 节。

当前任务每 **60 秒**自动刷新一次 MARE CSV 和最新摘要，训练结束后还会导出一次。因此 CSV 可能比 HDF5 落后约一分钟。曲线须等五个 geometry 均有有效能量后才开始。原始 `data/licn` 没有参考能量元数据；本任务只在冻结的数据副本中补入参考值，用于原生日志统计。仓库中的 `examples/licn_scratch/mare.py` 直接读取参考 CSV，因此对新训练无需修改 geometry YAML。

交付仓库提供可移植后处理入口。新训练可在训练中另开终端运行，或结束后运行；`--run-dir` 指向传给训练入口 `-w` 的目录：

```bash
# 对新训练每分钟导出一次；不需要原机器的冻结目录。
python examples/licn_scratch/mare.py \
  --run-dir runs/licn_example_new/results --watch

# 不带 --watch 时，只导出一次快照。
# 当前任务已有自动导出器；停掉自动导出器后，才对同一目录手动刷新。
python examples/licn_scratch/mare.py \
  --run-dir runs/licn_default_1gpu_20k_s42/results
```

脚本读取 `RUN_DIR/training/result.h5` 和仓库中的五点参考能量，导出到 `RUN_DIR/analysis/`，不会更新模型参数。每个输出目录只运行一个导出器，避免与现有自动导出器同时写入。

## 3. H₄(6)：单卡、六构型的能量收敛示例

### 3.1 算例内容

输入为 [`data/h4_globe6/00.yaml` 至 `05.yaml`](../data/h4_globe6/00.yaml)，对应 *Generalizing Neural Wave Functions* 的六个氢矩形构型。每个结构有四个 H 原子，为中性、4 电子体系，上下自旋各 2 个。

四个原子位于 `(±x, ±y, 0)`，其中：

```text
x = R cos(θ/2)，y = R sin(θ/2)
R = 3.2843 bohr
θ = 85°、86°、87°、88°、89°、89.999°（依次对应 00–05）
```

R 是中心到原子的距离，不能当作矩形边长。此数据集由 [`examples/h4_rectangle/generate.py`](../examples/h4_rectangle/generate.py) 生成，定义来源固定在 Globe 仓库版本 `942feba753a0ab912c23fdebc873ff9aefeb49f6`。具体来源和运行说明见 [H₄ 示例说明](../examples/h4_rectangle/README.md)。

### 3.2 训练设置与启动

单卡每次更新处理全部 **6 个 geometry × 每个 256 个电子构型**，全局为 **1536** 个构型，K-FAC 训练 **16,000 步**，seed **219**。六个 geometry 每步都参与，不需要六张卡。当前完整运行耗时约 **2.03 小时**，已保存并验证 `chkpt-16000.pt`。

在与第 2 节相同的环境和环境变量下，可启动新实验：

```bash
python examples/h4_rectangle/generate.py
CUDA_VISIBLE_DEVICES=0 python examples/h4_rectangle/run.py \
  --mode train --output runs/h4_example_new/results \
  > logs/h4_example_new.log 2>&1
```

固定参数见 [`examples/h4_rectangle/setting.json`](../examples/h4_rectangle/setting.json)。`--mode smoke` 只更新两步，用于运行检查；正式训练另用新目录。原论文的模型、预训练、采样和优化设置与本实验不同；这里使用相同六个几何结构训练 Orbformer，不把结果表述为原论文曲线的完整复现。

### 3.3 当前输出位置与应看指标

实验根目录：`runs/h4_globe6_scratch_1gpu_16k_s219/`（`runs/h4_globe6_scratch_1gpu_16k_s219/`，原机器输出）。

| 文件或目录（相对于该根目录） | 内容与用途 |
| --- | --- |
| `results/oneqmc_train.log`、`logs/stdout_stderr.log` | 上游训练日志、标准输出及错误 |
| `results/runtime_status.json` | 当前为 `COMPLETE`；完成 16,000 步，checkpoint 重载通过 |
| `results/training/config.yaml`、`result.h5` | 实际配置与逐步原始指标 |
| `results/training/chkpt-2000.pt` 等 | 每 2000 步的阶段 checkpoint；最终为 `chkpt-16000.pt` |
| `results/analysis/h4_convergence.png`、`.pdf` | **主要收敛图：六点平均的绝对训练能量，经 200 步平滑，单位 Ha** |
| `results/analysis/h4_convergence_fullrange.png`、`.pdf` | 同时显示未平滑与平滑后的绝对能量，包含最初几步 |
| `results/analysis/h4_convergence.csv` | 每步六点平均绝对能量、200 步平滑绝对能量，单位 Ha |
| `results/analysis/energy_by_geometry.csv` | 六个 geometry 各自的逐步能量，可排查某一结构是否异常 |
| `results/analysis/h4_convergence_metadata.json` | 平滑窗口、能量单位、最终能量和指标定义 |
| `exitcode_remote.txt` | 已记录正常退出码 0 |

**主要看绝对能量是否下降并进入平台，以及每个 geometry 的能量是否稳定。** 曲线按以下步骤生成：

1. 从 HDF5 的 `metrics/E_loc/mean_elec` 读取每步六个结构的电子采样平均能量，并按 geometry 索引排序。
2. 六个 geometry 等权平均，得到每一步的平均训练能量。
3. 对平均能量做向后 **200 步**滑动平均，记作 `E_smooth(t)`。
4. 直接绘制 `E_smooth(t)`，纵轴为绝对能量，单位 Hartree（Ha / `E_h`），不减去最低能量，不作能量平移。

前 199 步没有完整平滑窗口，因此平滑曲线留空，原始数据仍保留。主图采用线性纵轴、自动覆盖平滑能量的范围；`h4_convergence_fullrange.png` 额外显示从第 1 步开始的未平滑六点平均能量。两张图都显示绝对能量，坐标轴关闭 offset，直接标注实际数值。

当前最终平滑平均能量约为 **−2.01608152 Ha**。CSV 的 `mean_training_energy_Ha` 是每步六点平均值，`mean_energy_trailing_200_Ha` 是对应的平滑值；查看单个结构则使用 `energy_by_geometry.csv`。这些值是六个不同 geometry 的训练能量统计，不能视为某一个 H₄ 结构的精确基态能量；收敛判断应结合逐结构能量和后期波动。

## 4. 乙烷：单节点四卡、20 点路径的从头训练

### 4.1 算例内容

输入为 [`data/bbmep/Ethane/00.yaml` 至 `19.yaml`](../data/bbmep/Ethane/00.yaml)，是乙烷 C–H 键均裂路径上的 20 个固定几何结构：

```text
CH₃CH₃ → CH₃CH₂· + H·
```

每个结构有 2 个 C 和 6 个 H，为中性、18 电子体系，上下自旋各 9 个。所有结构共享一套模型参数，学习整条路径的能量。参考能量摘自上游 BBMEP 参考表，保存在 [`data/references/ethane_bbmep_reference_energy.csv`](../data/references/ethane_bbmep_reference_energy.csv) 的 `MEP_system == Ethane`、`Image_ID=0..19` 的 20 行。

### 4.2 训练设置与启动

使用 **单 Python 进程、JAX `pmap`、4 张 A100 80GB**。全局每步处理 **4 个 geometry，每卡 1 个，每个 256 个电子构型**，全局为 **1024** 个构型。K-FAC 训练 **32,000 次全局更新**，seed **219**。共享参数在卡间同步；不是四个相互独立的训练任务。

约 5 次更新覆盖一轮 20 个结构，每个 geometry 平均参与约 6400 次更新。当前完整运行耗时约 **4.40 小时**，已保存并验证 `chkpt-32000.pt`。

在相同环境中启动新实验：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/ethane_scratch/run.py \
  --mode train --devices 4 --output runs/ethane_example_new/results \
  > logs/ethane_example_new.log 2>&1
```

固定参数见 [`examples/ethane_scratch/setting.json`](../examples/ethane_scratch/setting.json)。不需要 `torchrun` 或为每张卡各启动一次 Python。环境安装、检查、续训和独立评估的完整步骤见[乙烷运行说明](ETHANE_SCRATCH_RUNBOOK_zh.md)。

### 4.3 当前输出位置与应看指标

实验根目录：`runs/ethane_scratch_4gpu_32k_s219/`（`runs/ethane_scratch_4gpu_32k_s219/`，原机器输出）。

| 文件或目录（相对于该根目录） | 内容与用途 |
| --- | --- |
| `results/oneqmc_train.log` | 上游训练日志 |
| `results/stdout_stderr.log` | 本实验的标准输出及错误；位置与两个单卡任务不同 |
| `results/runtime_status.json` | 当前为 `COMPLETE`；完成 32,000 步，四卡通信与 checkpoint 重载通过 |
| `results/training/config.yaml`、`result.h5` | 实际配置与逐步原始指标 |
| `results/training/chkpt-8000.pt` 等 | 每 8000 步的阶段 checkpoint；最终为 `chkpt-32000.pt` |
| `results/analysis/mare_online.png`、`.pdf` | **主要指标图：整条 20 点路径的在线 MARE** |
| `results/analysis/mare_online.csv` | MARE 随更新次数变化的数据 |
| `results/analysis/mare_online_latest.json` | 导出覆盖步数、最新 MARE、参考来源和导出时间 |
| `results/analysis/energy_online_latest.csv` | 每个 geometry 的训练 EWM 能量、EWM 误差估计、参考能量及居中相对误差 |
| `results/exitcode.txt` | 正常退出码 0；位置与两个单卡任务不同 |

**主要看 MARE 曲线是否下降、后期是否稳定，以及是否接近或低于图中 1 kcal/mol 的参考线。** 当前完整 32k 步的最终在线 MARE 约为 **0.59324 kcal/mol**。这是训练 EWM 指标；若要报告固定模型的最终精度，应按乙烷运行说明额外进行固定 checkpoint 的独立采样评估。

该 CSV 的横轴列为 `optimizer_updates`，纵轴列为 `MARE_online_EWM_kcal_per_mol`，与 LiCN 的列名不同。当前乙烷 CSV 已刷新至完整 32,000 步，是已完成任务的快照；默认后处理不会自动常驻刷新。需要重新导出时，从仓库根目录执行：

```bash
python examples/ethane_scratch/mare.py \
  --run-dir runs/ethane_scratch_4gpu_32k_s219/results --kind online
```

## 5. LiCN 与乙烷的 MARE 应如何理解

两个算例采用相同的参考居中定义。设路径有 N 个 geometry，`E_i(t)` 为训练到 t 时各结构最近的能量指数加权滑动平均（EWM），`E_ref_i` 为对应参考能量：

```text
MARE(t) = 627.5 / N × Σ_i |[E_i(t) − mean(E(t))] − [E_ref_i − mean(E_ref)]|
```

输入能量单位为 Hartree，627.5 将误差转换为 kcal/mol。**MARE 是去掉整体能量偏移后、相对能量曲线的平均绝对误差，不是百分比。** 因此还应结合 `energy_online_latest.csv` 查看每个点的能量与相对误差；MARE 小并不单独保证绝对总能量准确。

LiCN 每步更新 1/5 的结构，乙烷每步更新 4/20 的结构。导出器按 geometry 索引维护最近有效 EWM，覆盖全部结构后才计算 MARE；同一行 MARE 中各 geometry 的最近更新时间可能不同。两套曲线均用于观察训练趋势，包含滑动平均和采样噪声，不是固定模型的独立评估，也不是未见 geometry 的泛化误差。

查看曲线时，应关注持续下降、后期平台和反复波动，不能仅凭某一个较低的点判断收敛。图中的 1 kcal/mol 是便于阅读的参考线；在线平均误差低于该线并不保证每个结构误差都小于 1 kcal/mol。

## 6. 曲线查看与重新生成

### 6.1 LiCN：绘制当前 MARE 快照

当前任务自动刷新 CSV / JSON，PNG / PDF 使用以下命令手动生成或刷新。该命令只读取 CSV，不改训练配置或模型：

```bash
python - <<'PY'
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output = Path('runs/licn_default_1gpu_20k_s42/results/analysis')
with (output / 'mare_online.csv').open() as stream:
    rows = list(csv.DictReader(stream))
steps = [int(row['completed_updates']) for row in rows]
mare = [float(row['mare_kcal_mol']) for row in rows]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(steps, mare, label='Orbformer-SE, scratch, seed 42')
ax.axhline(1, color='gray', linestyle='--', label='1 kcal/mol')
ax.set(xlabel='Optimizer updates', ylabel='Online EWM MARE (kcal/mol)',
       title=f'LiCN: 5 geometries, snapshot through {steps[-1]} updates')
if min(mare) > 0:
    ax.set_yscale('log')
ax.grid(alpha=0.25)
ax.legend()
fig.text(0.5, 0.01, 'Training diagnostic; not independent evaluation.',
         ha='center', fontsize=8)
fig.tight_layout(rect=[0, 0.04, 1, 1])
for suffix in ('png', 'pdf'):
    fig.savefig(output / f'mare_online.{suffix}', dpi=200)
plt.close(fig)
print(f'Exported {output}/mare_online.png and .pdf through {steps[-1]} updates')
PY
```

曲线文件为 `LiCN MARE PNG`（`runs/licn_default_1gpu_20k_s42/results/analysis/mare_online.png`，原机器输出）。训练继续时，需重跑以上绘图命令才能更新图片；以 CSV 的最新步数为准，不能将早期图片当成完整 20k 结果。

### 6.2 H₄ 与乙烷：重新生成已有图

两套任务均已完成，可执行：

```bash
python examples/plot_training_curves.py --window 200
```

该脚本要求两套运行状态均为 `COMPLETE`，且乙烷 MARE CSV 已覆盖全部训练步数；必要时先执行第 4 节的 MARE 刷新命令。默认写入：

- H₄ 收敛图（`runs/h4_globe6_scratch_1gpu_16k_s219/results/analysis/h4_convergence.png`，原机器输出）和全范围图（`runs/h4_globe6_scratch_1gpu_16k_s219/results/analysis/h4_convergence_fullrange.png`，原机器输出）。
- 乙烷 MARE 图（`runs/ethane_scratch_4gpu_32k_s219/results/analysis/mare_online.png`，原机器输出）。
- 乙烷 / H₄ 并排图（`runs/analysis/ethane_h4_training_curves.png`，原机器输出），以及同目录 PDF 和 `training_curves_metadata.json`。

若对新输出目录重绘，可传 `--ethane-run runs/ethane_example_new/results --h4-run runs/h4_example_new/results --output-dir runs/analysis_new`，并先为新乙烷运行导出 MARE。

## 7. 实验目录、进度和完成检查

当前仓库位于 `/project_c20250506/251105015/oneqmc`。三个 `runs/...` 都是指向项目盘实验目录的软链接，训练工作目录是各自的 **`results/`**：

| 仓库内实验根目录 | 实际目录（前缀均为 `/project_c20250506/251105015/oneqmc_runs/`） |
| --- | --- |
| `runs/licn_default_1gpu_20k_s42` | `licn_default_1gpu_g1_e2048_20k_s42_20261116_212025` |
| `runs/h4_globe6_scratch_1gpu_16k_s219` | `h4_globe6_scratch_1gpu_g6_e256_16k_s219_20261116_174502` |
| `runs/ethane_scratch_4gpu_32k_s219` | `ethane_scratch_4gpu_g4_e256_32k_s219_20261116_153050` |

实验根目录的 `code/`、`protocol.json`、`source_sha256.json` 供核对实际输入和源码；`results/training/config.yaml` 是已解析的训练参数，修改仓库中的 `setting.json` 不会改变已启动的冻结任务。`runtime_status.json` 主要记录阶段，训练时不会逐步刷新已完成更新数。

跟踪当前 LiCN 日志：

```bash
tail -f runs/licn_default_1gpu_20k_s42/results/oneqmc_train.log
```

查看三个任务已写入的更新数（HDF5 只读 SWMR，不打断训练）：

```bash
python - <<'PY'
from pathlib import Path
import h5py
for name in ('licn_default_1gpu_20k_s42',
             'h4_globe6_scratch_1gpu_16k_s219',
             'ethane_scratch_4gpu_32k_s219'):
    with h5py.File(Path('runs') / name / 'results/training/result.h5',
                   'r', swmr=True) as stream:
        indices = stream['metrics/mol_idx']
        indices.refresh()
        print(name, 'logged updates:', len(indices))
PY
```

查看 TensorBoard 时，对需要的运行指定 `results/training`，例如：

```bash
tensorboard --logdir runs/licn_default_1gpu_20k_s42/results/training \
  --host 127.0.0.1 --port 6006
```

默认 HDF5 只保存筛选后的逐 geometry 指标；耗时等标量主要在 TensorBoard 中，不能假定所有标量都在 `result.h5`。LiCN 原生参考误差标量采用 Hartree，而本文导出的 MARE CSV 采用 kcal/mol，比较时需要统一单位。

训练完成应同时核对目标步数、`runtime_status.json` 的 `COMPLETE`、正常退出码 0，以及目标步数 checkpoint 的重载验证。`chkpt--1.pt` 是滚动保存的快速 checkpoint，文件名不表示训练只有 1 步；其实际步数保存在文件内部。

以上路径描述当前机器的已有实验；`runs/` 被 Git 忽略，克隆代码库不会自动得到这些日志、图片和 checkpoint。迁移时应单独复制所需实验目录，或用新输出目录重新运行。三套训练与后处理之间的指标口径，以本文第 3、5 节和对应元数据文件为准。
