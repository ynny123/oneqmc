# H₂ 平衡键长单卡基准

中性单重态 H₂，核间距 **1.4011 bohr**，共两个电子。非相对论、固定核 Born–Oppenheimer 基态总能量（含核间排斥）为 **−1.1744759314002167 Ha**。结构文件也记录了参考能量。[数据](../../data/H2_equilibrium/00.yaml) · [Pachucki, Phys. Rev. A 82, 032509 (2010)](https://arxiv.org/abs/1007.0322)

## 已归档的训练曲线

以下文件记录了单张 A100 80GB 上从头训练的全部 10,000 步。能量取训练的指数加权平均值，绝对误差相对于本算例的参考能量。

- [能量和绝对误差 CSV](../../training_records/h2_equilibrium_scratch_1gpu_10k_s42/results/analysis/absolute_energy_error_online.csv)
- [能量曲线（Ha）](../../training_records/h2_equilibrium_scratch_1gpu_10k_s42/results/analysis/energy_online.png)
- [绝对误差曲线（mHa）](../../training_records/h2_equilibrium_scratch_1gpu_10k_s42/results/analysis/absolute_energy_error_online.png)

## 运行方法

在仓库根目录用 OneQMC 环境执行。单卡从随机初始化训练 10,000 步，再固定最终 checkpoint 独立评估 1,000 步。训练与评估均为每步一个结构、1,024 个电子构型。

~~~bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export AMLT_DATA_DIR="$PWD/data"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=false
export CUDA_VISIBLE_DEVICES=0

python scripts/transferable.py -d H2_equilibrium -a orbformer-se \
  --mol-batch-size 1 --electron-batch-size 1024 --seed 42 \
  --pretrain-steps 0 --no-autoresume --max-restarts 1 \
  -n 10000 --chkpts-slow-interval 10000 -w runs/h2_equilibrium_train

ORBFORMER_PICKLE_LOADING=1 python scripts/transferable.py \
  -d H2_equilibrium -a orbformer-se --mol-batch-size 1 --electron-batch-size 1024 \
  --test --discard-sampler-state --no-autoresume --max-restarts 1 --seed 20261119 \
  -c runs/h2_equilibrium_train/training/chkpt-10000.pt \
  -n 1000 -w runs/h2_equilibrium_eval
~~~

每次运行使用新的 `-w` 目录。训练曲线用 `training/result.h5` 的 `metrics/E_loc/mean/ewm`，原始批平均能量在 `metrics/E_loc/mean_elec`。原生训练还会把绝对误差 `abs(E_EWM - E_ref)`（Ha）写入 `metrics/E_ewm_vs_ref/mae`。画能量曲线时可叠加参考线，也可画有符号误差 `(E - E_ref) × 1000`（mHa）。

训练启动后，在另一个终端从仓库根目录运行下面的命令。它每 60 秒刷新 `runs/h2_equilibrium_train/analysis/` 中的 `absolute_energy_error_online.csv` 和同名 PNG，误差单位为 mHa，虚线标出化学精度。按 Ctrl+C 可以停止刷新。训练结束后可去掉 `--watch`，一次性导出完整曲线。

~~~bash
python examples/plot_absolute_energy_error.py --run-dir runs/h2_equilibrium_train --watch
~~~

独立评估结束后，对全部批平均局部能量取 Huber 稳健均值，再计算绝对误差 `abs(E - E_ref) × 1000`（mHa）。**绝对误差越小越好，目标是低于 1 kcal/mol（约 1.59 mHa）的化学精度阈值。** [单位换算](https://cccbdb.nist.gov/hartreex.asp)

~~~bash
python - <<'PY'
import h5py
from oneqmc.analysis.energy import robust_mean

reference = -1.1744759314002167
with h5py.File("runs/h2_equilibrium_eval/training/result.h5", "r") as result:
    values = result["metrics/E_loc/mean_elec"][:]
    if values.shape != (1000, 1):
        raise ValueError(f"评估尚未完整结束，实际形状为 {values.shape}")
energy = robust_mean(values.astype("float64"), delta=1.0).item()
print(f"E = {energy:.8f} Ha, |E - E_ref| = {abs(energy-reference)*1000:.4f} mHa")
PY
~~~
