# Orbformer 独立评估命令

在 OneQMC 仓库根目录、已安装依赖的环境中执行。将 `/path/to/...` 换成实际 checkpoint 路径，保留它同目录的训练 `config.yaml`。每次评估使用新的 `-w` 目录，GPU 编号按实际分配调整。

~~~bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export AMLT_DATA_DIR="$PWD/data"
export ORBFORMER_PICKLE_LOADING=1
export JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# H₄ 单卡，6 个结构每步一起评估，每个结构 6,000 步
CUDA_VISIBLE_DEVICES=0 python scripts/transferable.py \
  -d h4_globe6 --test --discard-sampler-state --no-autoresume \
  -c /path/to/h4/training/chkpt-16000.pt \
  --seed 20261119 --mol-batch-size 6 --electron-batch-size 256 \
  -n 6000 --chkpts-fast-interval 6000 --chkpts-slow-interval 6000 \
  -w runs/repro_h4_eval

# LiCN 单卡，5 个结构平均各评估 6,000 步
CUDA_VISIBLE_DEVICES=0 python scripts/transferable.py \
  -d licn --test --discard-sampler-state --no-autoresume \
  -c /path/to/licn/training/chkpt-20000.pt \
  --seed 20261119 --mol-batch-size 1 --electron-batch-size 2048 \
  -n 30000 --chkpts-fast-interval 30000 --chkpts-slow-interval 30000 \
  -w runs/repro_licn_eval

# 乙烷单节点四卡，20 个结构平均各评估 6,000 步
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/transferable.py \
  -d bbmep/Ethane --test --discard-sampler-state --no-autoresume \
  -c /path/to/ethane/training/chkpt-32000.pt \
  --seed 20261119 --mol-batch-size 4 --electron-batch-size 256 \
  -n 30000 --chkpts-fast-interval 30000 --chkpts-slow-interval 30000 \
  -w runs/repro_ethane_eval
~~~

`--test` 固定 checkpoint 参数，`--discard-sampler-state` 初始化新采样链。

## 最终精度对齐指标

### 1. 评估输出

每条评估命令在其 `-w` 目录下写入以下文件。例如 H₄ 的原始能量在 `runs/repro_h4_eval/training/result.h5`。

| 路径（相对于 `-w`） | 内容 |
| --- | --- |
| `oneqmc_train.log` | checkpoint 加载、采样平衡和评估进度 |
| `training/config.yaml` | 实际评估配置。可核对输入 checkpoint、评估步数、结构批量和电子构型批量 |
| `training/result.h5` | 能量原始记录。`metrics/mol_idx` 是结构编号，`metrics/E_loc/mean_elec` 是同位置的批平均局部能量（Ha） |
| `training/chkpt-*.pt` | 评估期间保存的状态。能量汇总直接读取 HDF5 |

每个全局步占 HDF5 一行，每一列是该步选中的一个结构。结构位置随批次变化，汇总时用 `mol_idx` 对应编号。H₄ 每个结构有 6,000 条，LiCN 和乙烷平均每个结构有 6,000 条，实际次数看后处理输出的 `visits`。

### 2. 计算逐结构能量和 MARE

按结构汇集固定 checkpoint 下的**全部评估能量记录**，求 Huber 稳健均值（`delta=1 Ha`），得到各结构的绝对能量 `E_i`。H₄ 另外计算六个结构的等权平均。LiCN 和乙烷按结构编号匹配仓库中的参考能量。令 `d_i = E_i - E_ref,i`，则 `MARE = 627.5 × mean_i(abs(d_i − mean_j(d_j)))`，单位为 kcal/mol。每个结构在 MARE 中等权。[Huber 稳健均值函数](../src/oneqmc/analysis/energy.py) · [LiCN 参考能量](../data/references/licn_reference_energy.csv) · [乙烷参考能量](../data/references/ethane_bbmep_reference_energy.csv)

等评估结束后，在仓库根目录执行下面的命令。第一行可以只传入要汇总的 `-w` 目录，也可以一次传入多个。命令会按 `training/config.yaml` 识别算例并检查评估步数。

~~~bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python - runs/repro_h4_eval runs/repro_licn_eval runs/repro_ethane_eval <<'PY'
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

from oneqmc.analysis.energy import robust_mean

cases = {
    "h4_globe6": (6, None),
    "licn": (5, ("licn_reference_energy.csv", "geometry", "reference_energy_hartree")),
    "bbmep/Ethane": (20, ("ethane_bbmep_reference_energy.csv", "Image_ID", "Reference_E(Hartree)")),
}

for directory in sys.argv[1:]:
    output = Path(directory)
    with (output / "training/config.yaml").open() as stream:
        args = yaml.safe_load(stream)["args"]
    if not args.get("test"):
        raise ValueError(f"{output} 不是 --test 评估目录")
    size, reference_spec = cases[args["dataset"]]
    with h5py.File(output / "training/result.h5", "r") as stream:
        ids = stream["metrics/mol_idx"][:]
        energies = stream["metrics/E_loc/mean_elec"][:].astype(np.float64)
    expected_shape = (args["train_steps"], args["mol_batch_size"])
    if ids.shape != expected_shape or energies.shape != expected_shape:
        raise ValueError(f"{output} 的评估记录不完整，预期形状为 {expected_shape}")
    if not np.isfinite(energies).all() or not np.array_equal(np.unique(ids), np.arange(size)):
        raise ValueError(f"{output} 包含非有限能量或结构编号异常")

    rows = []
    for geometry in range(size):
        values = energies[ids == geometry]
        rows.append(dict(geometry=geometry, visits=int(values.size),
                         energy_Ha=robust_mean(values, delta=1.0).item()))
    estimates = np.array([row["energy_Ha"] for row in rows])
    summary = dict(dataset=args["dataset"], checkpoint=args["chkpt"],
                   evaluation_steps=ids.shape[0], mean_energy_Ha=float(estimates.mean()))
    if reference_spec is not None:
        filename, index_key, energy_key = reference_spec
        with (Path("data/references") / filename).open(newline="") as stream:
            reference = {int(row[index_key]): float(row[energy_key]) for row in csv.DictReader(stream)}
        if set(reference) != set(range(size)):
            raise ValueError(f"{filename} 的结构编号与评估数据不一致")
        ref = np.array([reference[i] for i in range(size)])
        delta = estimates - ref
        centered = (delta - delta.mean()) * 627.5
        for row, value, error in zip(rows, ref, centered):
            row.update(reference_Ha=float(value), centered_relative_error_kcal_mol=float(error))
        summary["MARE_kcal_mol"] = float(np.abs(centered).mean())

    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    with (analysis / "energies_evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (analysis / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{output}: {summary}")
PY
~~~

每个评估目录会新增 `analysis/energies_evaluation.csv` 和 `analysis/evaluation.json`。CSV 保存逐结构绝对能量和使用的记录数，LiCN 和乙烷还包含参考能量及带符号的居中误差。JSON 保存算例、输入 checkpoint、全局评估步数及汇总指标。

### 3. 怎么看结果

| 算例 | 主要看什么 | 如何判断 |
| --- | --- | --- |
| H₄ | JSON 的六结构平均绝对能量 `mean_energy_Ha`，以及 CSV 的 6 个 `energy_Ha` | 在相同结构和评估设置下，平均能量越低（越负）越好。本算例未配置参考能量，直接与 JAX 基线比较，并逐结构检查偏差 |
| LiCN | JSON 的相对能量误差 `MARE_kcal_mol`，以及 CSV 的 5 个 `energy_Ha` | MARE 越小、越接近 0 越好，表示五个结构的相对能量越接近参考。用居中误差列定位偏差较大的结构 |
| 乙烷 | JSON 的相对能量误差 `MARE_kcal_mol`，以及 CSV 的 20 个 `energy_Ha` | MARE 越小、越接近 0 越好，表示整条路径的相对能量越接近参考。用居中误差列检查偏差是否集中在某个路径区段 |

先核对两边的结构顺序、评估批量和采样预算。验证评估实现时，两边加载同一份权重。验证训练精度时，比较相同训练步数的 checkpoint。三个算例都可按结构编号计算 `ΔE_i = E_i(适配版) − E_i(JAX 基线)`，报告平均绝对差和最大绝对差。即使两边 MARE 接近，`ΔE_i` 也能揭示共同的绝对能量偏移。

最终对齐比较固定 checkpoint 的独立评估结果。训练期间的在线 MARE 用于观察收敛趋势。独立 MCMC 存在统计波动，小幅差异应结合重复评估的波动和双方约定的容差判断。
