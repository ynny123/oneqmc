# Orbformer 乙烷多几何从头训练运行说明

本文以当前乙烷 BBMEP 实验为例，说明如何从源码准备环境、验证运行、启动训练、查看输出并计算 MARE。接收方只需要代码库和可写的工作目录，不需要原机器的绝对路径、调度平台账号或 LAC 预训练权重。

基线源码版本为 `46ddc5a239650672a3b9eb276b292d3326c79ad5`。本文的环境版本对应已运行的 Linux x86_64、Python 3.11、4 张 A100 80GB 配置。通用入口是仓库中的 `scripts/transferable.py`，配套启动脚本通过该入口运行原有训练代码。

这是交付时的 A100 运行基线；昇腾设备上的安装与运行尚未实测。配套的[交付验证记录](ETHANE_HANDOFF_VALIDATION_zh.md)说明实际完成的检查和验证范围。

## 1 实验任务和固定配置

任务是计算乙烷 C–H 键均裂路径上的能量：

```text
CH3CH3 → CH3CH2· + H·
```

数据集包含预先生成好的 20 个分子几何结构，训练不会重新优化核坐标。每个结构均为 8 个原子、18 个电子，上下自旋各 9 个。20 个 geometry 全部参与训练，共享一套模型参数；本实验没有额外划分未见 geometry 测试集。

| 项目 | 本次配置 |
| --- | --- |
| 数据集 | `data/bbmep/Ethane/00.yaml` 至 `19.yaml` |
| 模型 | `orbformer-se`，3,029,650 个参数 |
| 初始状态 | 随机初始化，seed 219 |
| 外部 checkpoint | 不加载 |
| Hartree–Fock 预训练 | 0 步 |
| 参数共享方式 | `--orb-parameter-mode chem-pretrain` |
| 运行方式 | 单节点、单 Python 进程，JAX `pmap` 使用 4 张卡 |
| 全局 geometry batch | 每次参数更新共 4 个 geometry，每张卡 1 个 |
| 电子构型 batch | 每个 geometry 256 个完整电子构型，全局共 1024 个 |
| 训练长度 | 32,000 次全局参数更新 |
| 优化器 | K-FAC，learning rate 0.05，damping 0.001，norm constraint 0.001 |
| 行列式及包络数 | 16 个行列式，每个原子核 8 个包络 |
| 采样 | stack 多体系采样器，Metropolis，去相关步数 60，初始平衡最多 500 步 |
| 其他设置 | `repeated-sampling-len=1`，`clip-width=5.0`，`det-penalty-weight=0.001` |
| 计算设置 | float32，matmul precision 为 `high`，Flash Attention 开启，forward Laplacian |
| checkpoint | 每 101 步保存快速检查点，每 8000 步保存阶段检查点 |

0.05 是 K-FAC 的基础学习率。上游使用 `InverseScheduleLinearRamp(0.05, 16000, 250)`：先预热，再按该调度函数衰减，并非整个过程固定为 0.05。

**`chem-pretrain` 是上游对共享 orbital generator 模式的命名。是否加载预训练权重由 `-c/--chkpt` 决定。本实验没有该参数，且 `--pretrain-steps 0`，因此是从头训练。**

256 表示 256 组电子坐标，每组有 18 个电子，并不是把电子数改成 256。全局逻辑布局可理解为 `[4 张卡, 每卡 1 个 geometry, 256 个构型, 18 个电子, 3 个坐标]`；代码中上下自旋电子分别存储。

在默认每步更换 geometry batch 的设置下，约 5 次更新覆盖 20 个结构一轮。32k 步结束时，每个 geometry 平均参与 6400 次更新。

## 2 需要交付的文件

```text
oneqmc/
├── README.md
├── docs/ETHANE_SCRATCH_RUNBOOK_zh.md       本说明
├── requirements/ethane-a100-py311.txt    已运行环境的依赖版本
├── examples/ethane_scratch/
│   ├── env.sh                          环境变量及缓存路径
│   ├── setup_env.sh                    创建并安装 Python 环境
│   ├── setting.json                    固定的训练参数
│   ├── run.py                          检查、训练、续训和评估入口
│   └── mare.py                         MARE 后处理
├── scripts/transferable.py              上游训练入口
├── scripts/args.py                      上游命令行参数
├── src/oneqmc/                          模型、采样、能量和优化器
├── data/bbmep/Ethane/                   20 个结构
├── data/references/
│   └── ethane_bbmep_reference_energy.csv  MARE 所用参考能量
└── tests/                              上游测试
```

数据已经随仓库提供，不需要另外下载 TinyMol 或模型权重。参考能量 CSV 只用于运行后的误差计算，训练通过 VMC 能量最小化进行，不将这些能量作为监督标签。

结构 YAML 中的核坐标单位为 Å（`unit: angstrom`），加载时由上游转换为内部原子单位。不要手工再次转换坐标。

`runs/ethane_scratch_4gpu_32k_s219` 是本机已完成的 seed 219 乙烷实验入口；模板保持相同 seed，本轮不重新提交乙烷任务。重新从头训练时，请使用新的输出目录，避免覆盖已有结果。接收方按本文的 `examples/ethane_scratch/run.py` 启动新实验，它调用当前仓库的 `scripts/transferable.py`，不依赖原机器的软链接。

## 3 环境准备

### 3.1 版本基线

| 组件 | 版本 |
| --- | --- |
| Python | 3.11；参考运行使用 3.11.18 |
| JAX / jaxlib / CUDA plugin | 0.4.34 |
| CUDA 运行库 / cuDNN | 12.2 / 9.2.1.18 |
| dm-haiku | 0.0.16 |
| folx | 0.2.19 |
| kfac-jax | 0.0.6 |
| optax / flax | 0.2.5 / 0.10.4 |
| NumPy / SciPy | 1.26.4 / 1.11.3 |
| PySCF / h5py | 2.11.0 / 3.15.1 |

完整依赖固定在 [`requirements/ethane-a100-py311.txt`](../requirements/ethane-a100-py311.txt)。安装前准备好可用的 Python 3.11，包含 `venv` 和 pip。A100 基线还需要 Linux x86_64、可用的 NVIDIA 驱动和设备分配；用户态 CUDA 库由依赖文件安装。所有命令均从仓库根目录开始。

单卡运行检查使用 A100 80GB，正式配置使用单节点四卡 A100 80GB。参考机器驱动版本为 `535.129.03`，可先运行 `nvidia-smi` 确认设备分配和驱动状态。安装 Python 依赖不会安装系统驱动。

### 3.2 创建环境

```bash
cd /实际路径/oneqmc

# 可选：将运行环境和缓存放到指定的可写项目盘。
# 不设置时，默认使用当前仓库的 .runtime/ethane/。
# export ONEQMC_WORK_ROOT=/data/oneqmc_ethane_runtime

source examples/ethane_scratch/env.sh
bash examples/ethane_scratch/setup_env.sh
source "$ONEQMC_VENV/bin/activate"

python --version
python -m pip check
python -c 'import jax; print("JAX:", jax.__version__); print("Devices:", jax.devices())'
```

`setup_env.sh` 默认使用 `python3.11`，可通过 `PYTHON_BIN` 指定解释器。默认从官方 PyPI 安装；需要镜像时，在安装前设置 `ONEQMC_PIP_INDEX_URL`，例如：

```bash
export ONEQMC_PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
bash examples/ethane_scratch/setup_env.sh
```

有完整离线 wheel 目录时可设置：

```bash
export ONEQMC_WHEELHOUSE=/实际路径/wheelhouse
bash examples/ethane_scratch/setup_env.sh
```

离线目录需包含锁定文件中所有依赖的 Python 3.11、Linux x86_64 安装包；仅复制 `requirements` 文件不能代替安装包。

`env.sh` 将 pip、Conda、临时文件、Python 字节码和编译缓存统一指向工作目录，不修改 `HOME`，也不向 `/root` 下载文件。默认下载量为数 GB，解压后的环境及缓存需要额外空间；建议为工作目录预留至少 20 GB，并为长训练日志和 checkpoint 保留余量。

新终端中重新进入环境：

```bash
cd /实际路径/oneqmc
# 如上次使用了自定义 ONEQMC_WORK_ROOT，这里设置相同的值。
source examples/ethane_scratch/env.sh
source "$ONEQMC_VENV/bin/activate"
```

## 4 先执行单卡两步检查

此检查使用 `00.yaml`、同一完整模型、每卡真实 256 个构型和 K-FAC，随机初始化后更新两步。初始采样器仅平衡 1 步，以缩短检查时间；该设置只用于验证运行，不能用其能量判断精度。

```bash
mkdir -p "$ONEQMC_WORK_ROOT/logs"
export SMOKE_DIR="$ONEQMC_WORK_ROOT/runs/ethane_smoke"

CUDA_VISIBLE_DEVICES=0 python examples/ethane_scratch/run.py \
  --mode smoke \
  --output "$SMOKE_DIR" \
  > "$ONEQMC_WORK_ROOT/logs/ethane_smoke.log" 2>&1

cat "$SMOKE_DIR/runtime_status.json"
```

成功时应出现：

```json
{
  "stage": "COMPLETE",
  "completed_global_steps": 2,
  "checkpoint_reload": "PASSED",
  "changed_parameter_tensors": 88
}
```

`changed_parameter_tensors` 以实际结果为准，要求大于 0，不要求逐台机器恰好等于示例中的 88。脚本同时检查参数有限、发生更新，以及保存后的 checkpoint 可以严格读取。

每次从头检查使用新的输出目录，例如 `ethane_smoke_2`。第一次运行包含 JIT 编译和 K-FAC 初始化，可能耗时数分钟。

## 5 启动正式四卡训练

先预览并检查命令，不初始化设备、不启动训练，也不创建实验输出目录：

```bash
export TRAIN_DIR="$ONEQMC_WORK_ROOT/runs/ethane_scratch_4gpu_32k_s219"

python examples/ethane_scratch/run.py \
  --mode train --devices 4 --output "$TRAIN_DIR" --dry-run
```

确认有 4 张可见设备后，运行：

```bash
mkdir -p "$ONEQMC_WORK_ROOT/logs"

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/ethane_scratch/run.py \
  --mode train \
  --devices 4 \
  --output "$TRAIN_DIR" \
  > "$ONEQMC_WORK_ROOT/logs/ethane_train.log" 2>&1
```

这是**单进程四卡**启动，不需要 `torchrun`、`mpirun` 或每张卡分别启动 Python。脚本会验证可见设备数量以及跨设备求和通信，再进入训练。如果调度器已经设置了 `CUDA_VISIBLE_DEVICES`，保留调度器给定的值，直接运行上面的 Python 命令即可。

首次运行要求输出目录不存在或为空，以避免误读其他实验的 checkpoint。日志重定向到独立的 `logs/` 目录，不能在首次启动前把日志文件写入 `TRAIN_DIR`。

需要脱离终端运行时，可使用系统的 `screen`，或在上述命令前使用 `nohup` 并在末尾加 `&`。需要与当前远程任务相同的进程时限时，在 Python 命令前增加：

```bash
timeout --signal=TERM --kill-after=120s 8h
```

`run.py` 本身不设置超时。8 小时是旧包装层使用过的保护值；自行运行时应按实际目标和机器设置。实际耗时请在目标机器完成编译后，按 TensorBoard 中测得的步速估算，另计初始化和平衡时间。

## 6 与当前配置对应的原生训练命令

需要直接使用上游入口时，以下命令与正式训练的核心参数一致。保持 `TRAIN_DIR` 为新目录：

```bash
python scripts/transferable.py \
  -d bbmep/Ethane \
  -a orbformer-se \
  --orb-parameter-mode chem-pretrain \
  --mol-batch-size 4 \
  --electron-batch-size 256 \
  --train-steps 32000 \
  --seed 219 \
  --pretrain-steps 0 \
  --optimizer kfac \
  --learning-rate 0.05 \
  --damping 0.001 \
  --norm-constraint 0.001 \
  --n-determinants 16 \
  --n-envelopes-per-nucleus 8 \
  --jax-matmul-precision high \
  --multi-system-sampler stack \
  --repeated-sampling-len 1 \
  --decorr-steps 60 \
  --max-eq-steps 500 \
  --max-restarts 1 \
  --chkpts-fast-interval 101 \
  --chkpts-slow-interval 8000 \
  -w "$TRAIN_DIR"
```

该命令依赖仓库根目录作为工作目录，并要求 `PYTHONPATH` 包含 `src/`。第 3 节的 `env.sh` 已设置该路径。所有其他未显式列出的上游参数均保存在启动后的 `training/config.yaml` 中；通用脚本还会在 `launch.json` 中保存完整解析结果。

## 7 输出文件与进度查看

```text
$ONEQMC_WORK_ROOT/
├── logs/ethane_train.log          标准输出、错误和编译日志
└── runs/ethane_scratch_4gpu_32k_s219/
    ├── launch.json               实际启动命令、全部参数和数据哈希
    ├── runtime_status.json       初始化、运行、完成或失败阶段
    ├── oneqmc_train.log          上游训练日志
    ├── training/
    │   ├── config.yaml           上游运行配置和模型尺寸
    │   ├── result.h5             逐步采样、能量、优化器和耗时指标
    │   ├── events.out.tfevents.*  TensorBoard 数据
    │   ├── chkpt--1.pt           最新快速 checkpoint
    │   ├── chkpt--2.pt           前一个快速 checkpoint
    │   ├── chkpt-8000.pt         阶段 checkpoint，到对应步数后生成
    │   └── chkpt-32000.pt        正式训练结束时的 checkpoint
    └── analysis/                 手动运行 MARE 后处理时生成
```

查看日志：

```bash
tail -f "$ONEQMC_WORK_ROOT/logs/ethane_train.log"
```

`runtime_status.json` 用于查看运行阶段，训练过程中不会逐步更新步数。可以通过 HDF5 查询已写入的全局更新数：

```bash
python - "$TRAIN_DIR" <<'PY'
import sys
from pathlib import Path
import h5py
with h5py.File(Path(sys.argv[1]) / 'training/result.h5', 'r', swmr=True) as f:
    print('Logged optimizer updates:', f['metrics/mol_idx'].shape[0])
PY
```

TensorBoard：

```bash
tensorboard --logdir "$TRAIN_DIR/training" --host 127.0.0.1 --port 6006
```

正常训练结束应同时满足：日志包含 `Progress: 32000/32000` 和 `The training has been completed!`；状态文件为 `COMPLETE`；最终 checkpoint 记录 32000 次更新，并通过脚本的有限性及重载检查。这些条件表示程序完整运行，不自动等同于达到化学精度。

## 8 断点续训

使用原输出目录和相同的目标步数：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/ethane_scratch/run.py \
  --mode train --devices 4 --output "$TRAIN_DIR" --resume \
  >> "$ONEQMC_WORK_ROOT/logs/ethane_train.log" 2>&1
```

上游会读取该目录最新的可用 checkpoint，恢复模型参数、优化器和采样器状态，继续至总共 32000 步；不是在已有进度上再增加 32000 步。重新从头训练时使用新的输出目录，并省略 `--resume`。

启动脚本会对照 `training/config.yaml` 检查续训参数。续训时保持设备数、总目标步数及训练设置一致；`--steps` 改变上游学习率计划，不能将短跑检查直接当作正式训练的前缀。

读取 checkpoint 需要 `ORBFORMER_PICKLE_LOADING=1`，`env.sh` 已设置。仅加载本项目自行生成或来源可信的 pickle checkpoint。续训按上游规则重建随机流，不承诺与完全不中断的运行逐位一致。

## 9 查看训练过程中的在线 MARE

当前结构文件没有配置 `reference_energy`，因此原生训练日志不会直接产生 MARE。配套后处理脚本读取已保存的每个 geometry 的训练能量滑动平均值，与官方 BBMEP 参考能量对照：

```bash
python examples/ethane_scratch/mare.py --run-dir "$TRAIN_DIR" --kind online
```

生成：

```text
$TRAIN_DIR/analysis/mare_online.csv
$TRAIN_DIR/analysis/mare_online_latest.json
$TRAIN_DIR/analysis/energy_online_latest.csv
```

公式为：

```text
MARE = 627.5 × mean_i |(E_i − mean(E)) − (E_ref_i − mean(E_ref))|
```

能量输入单位为 Hartree，输出为 kcal/mol，与论文绘图脚本的换算系数一致。MARE 是去除整体能量偏移后的平均绝对误差，不是百分比。

`energy_online_latest.csv` 包含 20 个结构的训练能量滑动平均、对应 EWM 误差估计、参考能量和去除整体偏移后的相对误差；该误差估计不是独立评估的置信区间。

在线数值用于观察训练趋势，包含滑动平均、采样噪声和不同 geometry 更新时间的影响；不能直接当作固定模型的独立评估结果。每次命令导出当前快照，需要刷新时重新执行。单卡单 geometry 的两步检查不覆盖全部 20 个点，不能计算完整路径 MARE。

## 10 固定 checkpoint 后独立评估

训练完成后，使用固定的 `chkpt-32000.pt`，在新的输出目录重新平衡 MCMC，然后评估能量。评估不更新模型参数。

以下示例平均为每个结构设置约 6000 个评估步骤。每个全局步骤处理 4 个 geometry，`6000 × 20 / 4 = 30000` 个全局步骤；每次仍采样 256 个电子构型。初始化也会消耗 geometry batch，日志不一定从轮换边界开始，各结构实际访问次数以导出的统计为准：

这是额外评估成本，不包含在训练 32k 步内。

```bash
export EVAL_DIR="$ONEQMC_WORK_ROOT/runs/ethane_eval_ckpt32000"

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/ethane_scratch/run.py \
  --mode evaluate \
  --devices 4 \
  --steps 30000 \
  --checkpoint "$TRAIN_DIR/training/chkpt-32000.pt" \
  --output "$EVAL_DIR" \
  > "$ONEQMC_WORK_ROOT/logs/ethane_eval.log" 2>&1

python examples/ethane_scratch/mare.py --run-dir "$EVAL_DIR" --kind evaluation
```

评估模式向上游传递 `--test --discard-sampler-state --no-autoresume`。这样会丢弃原采样器和优化器状态、从评估第 0 步开始，使用新 MCMC 样本计算同一套固定参数的能量。启动脚本在结束后检查模型参数与输入 checkpoint 保持逐元素一致。

后处理使用与上游 `robust_mean()` 相同的 Huber 目标函数，`delta=1 Ha`，分别估计 20 个结构的能量，再计算 MARE。输出为：

```text
$EVAL_DIR/analysis/mare_evaluation.json
$EVAL_DIR/analysis/energies_evaluation.csv
```

若改变 geometry batch 或设备数，需要相应调整全局评估步数。仅评估一部分结构得到的误差，不代表整条 20 点路径的 MARE。

只检查评估流程时，可改用 `--steps 10`（四卡）或 `--devices 1 --steps 40`（单卡），覆盖至少一轮完整的 20 个结构。初始化也会消耗 geometry batch，日志可能从一轮的中间开始，因此仅运行 5 个四卡步骤或 20 个单卡步骤不保证覆盖全部结构。如此短的评估只用于检查输出和参数冻结，不能用于报告最终精度。配套入口要求设备数能整除 20。

## 11 常见运行问题

| 现象 | 检查方法 |
| --- | --- |
| `No module named oneqmc` | 在当前终端重新 `source examples/ethane_scratch/env.sh`，并激活安装好的环境。 |
| 检测到的设备数与 4 不符 | 检查调度器分配和 `CUDA_VISIBLE_DEVICES`；确认只启动一个 Python 进程。 |
| `Data missing` | 确认完整保留 `data/bbmep/Ethane/00.yaml` 至 `19.yaml`。直接运行上游命令时从仓库根目录启动。 |
| 输出目录非空 | 从头运行改用新目录；恢复已有训练使用 `--resume`。不要删除正在运行任务的输出。 |
| 开始后几分钟还没有更新步数 | 查看日志是否处于参数初始化、采样器平衡或 JIT 编译阶段；首步耗时不能作为稳态耗时。 |
| 最初几步能量列表有 `nan` | 可能是尚未访问的 geometry 的滑动统计未初始化；在覆盖全部 20 个结构后检查。真实数值异常会伴随错误日志或失败状态。 |
| MARE 文件没有随训练更新 | 后处理导出的是快照，重新执行 `mare.py` 刷新。 |
| 显存不足 | 先用单卡检查定位；减小 batch 或开启上游 `--local-energy-chunk-size` 都会改变运行配置，比较结果时记录修改项。 |

## 12 交接和验收清单

1. 完整保留代码、20 个结构、参考能量 CSV、锁定依赖文件和本文的配套脚本。
2. 环境中 `python -m pip check` 通过，设备枚举和通信检查通过。
3. 单卡两步检查完成，参数有限、发生更新，checkpoint 可重载。
4. 四卡启动后确认全局 geometry batch 为 4、每个 geometry 的构型 batch 为 256，且从随机参数开始。
5. 正式运行至 32000 步并保存最终 checkpoint；精度比较使用第 10 节的独立评估。

若需将当前工作树打包，可从仓库根目录运行以下命令。输出放在仓库外，且不包含本机运行目录和软链接：

```bash
tar --exclude='./.git' --exclude='./runs' --exclude='./.runtime' \
  --exclude='__pycache__' --exclude='.pytest_cache' \
  -czf ../oneqmc_ethane_handoff.tar.gz .
```

参考资料：仓库 [README](../README.md)、[上游训练入口](../scripts/transferable.py)、[命令行参数](../scripts/args.py)、[论文绘图 notebook](../notebooks/02_bbmep/create_figures.ipynb)，以及 [Orbformer 论文](https://arxiv.org/abs/2506.19960)。
