# 硼原子单卡基准

中性硼原子 B 的基态为自旋双重态，含 **5 个电子**（3 个自旋向上、2 个自旋向下）。非相对论、无限核质量的 ²P 基态总能量参考值为 **−24.65386890 Ha**，取自显式相关高斯计算的外推结果。[结构文件](../../data/B_atom/00.yaml) · [参考论文表 I](https://arxiv.org/html/2009.08723)

## 已归档的训练曲线

以下文件记录了单张 A100 80GB 上从头训练的全部 10,000 步。能量取训练的指数加权平均值，绝对误差相对于本算例的参考能量。

- [能量和绝对误差 CSV](../../training_records/b_atom_scratch_1gpu_10k_s42/results/analysis/absolute_energy_error_online.csv)
- [能量曲线（Ha）](../../training_records/b_atom_scratch_1gpu_10k_s42/results/analysis/energy_online.png)
- [绝对误差曲线（mHa）](../../training_records/b_atom_scratch_1gpu_10k_s42/results/analysis/absolute_energy_error_online.png)

## 运行方法

在仓库根目录用 OneQMC 环境执行。单卡从随机初始化训练 **10,000 步**，再固定最终 checkpoint 独立评估 **1,000 步**。训练和评估每步均为一个结构、**1,024 个电子构型**。每次运行使用新的 `-w` 目录。

~~~bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export AMLT_DATA_DIR="$PWD/data"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=false
export CUDA_VISIBLE_DEVICES=0

python scripts/transferable.py -d B_atom -a orbformer-se \
  --mol-batch-size 1 --electron-batch-size 1024 --seed 42 \
  --pretrain-steps 0 --no-autoresume --max-restarts 1 \
  -n 10000 --chkpts-slow-interval 10000 -w runs/b_atom_train

ORBFORMER_PICKLE_LOADING=1 python scripts/transferable.py \
  -d B_atom -a orbformer-se --mol-batch-size 1 --electron-batch-size 1024 \
  --test --discard-sampler-state --no-autoresume --max-restarts 1 --seed 20261119 \
  -c runs/b_atom_train/training/chkpt-10000.pt \
  -n 1000 -w runs/b_atom_eval
~~~

训练的指数加权平均能量和绝对误差分别在 `training/result.h5` 的 `metrics/E_loc/mean/ewm` 和 `metrics/E_ewm_vs_ref/mae`，单位均为 Ha。训练启动后，在另一个终端从仓库根目录运行下面的命令。它每 60 秒刷新 `runs/b_atom_train/analysis/` 中的 `absolute_energy_error_online.csv` 和同名 PNG，误差单位为 mHa，虚线标出化学精度。按 Ctrl+C 可以停止刷新。训练结束后可去掉 `--watch`，一次性导出完整曲线。

~~~bash
python examples/plot_absolute_energy_error.py --run-dir runs/b_atom_train --watch
~~~

独立评估完成后，从评估 HDF5 读取全部批平均局部能量并取 Huber 稳健均值。**绝对误差越小越好，目标低于 1 kcal/mol（约 1.59 mHa）。**

~~~bash
python - <<'PY'
import h5py
from oneqmc.analysis.energy import robust_mean

reference = -24.65386890
with h5py.File("runs/b_atom_eval/training/result.h5", "r") as result:
    values = result["metrics/E_loc/mean_elec"][:]
    if values.shape != (1000, 1):
        raise ValueError(f"评估尚未完整结束，实际形状为 {values.shape}")
energy = robust_mean(values.astype("float64"), delta=1.0).item()
print(f"E = {energy:.8f} Ha, |E - E_ref| = {abs(energy-reference)*1000:.4f} mHa")
PY
~~~
