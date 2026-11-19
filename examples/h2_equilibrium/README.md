# H₂ 平衡键长单卡基准

中性单重态 H₂，核间距 **1.4011 bohr**，共两个电子。非相对论、固定核 Born–Oppenheimer 基态总能量（含核间排斥）为 **−1.1744759314002167 Ha**。结构文件也记录了参考能量。[数据](../../data/H2_equilibrium/00.yaml) · [Pachucki, Phys. Rev. A 82, 032509 (2010)](https://arxiv.org/abs/1007.0322)

在仓库根目录用 OneQMC 环境执行。从随机初始化开始，每步使用一个结构和 256 个电子构型。以下 2,000 步用于首轮验证，训练精度以独立评估结果为准。

~~~bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export AMLT_DATA_DIR="$PWD/data"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=false
export CUDA_VISIBLE_DEVICES=0

python scripts/transferable.py -d H2_equilibrium -a orbformer-se \
  --mol-batch-size 1 --electron-batch-size 256 --seed 42 \
  --pretrain-steps 0 --no-autoresume --max-restarts 1 \
  -n 2000 --chkpts-slow-interval 2000 -w runs/h2_equilibrium_train

ORBFORMER_PICKLE_LOADING=1 python scripts/transferable.py \
  -d H2_equilibrium -a orbformer-se --mol-batch-size 1 --electron-batch-size 256 \
  --test --discard-sampler-state --no-autoresume --max-restarts 1 --seed 20261119 \
  -c runs/h2_equilibrium_train/training/chkpt-2000.pt \
  -n 1000 -w runs/h2_equilibrium_eval
~~~

每次运行使用新的 `-w` 目录。训练曲线用 `training/result.h5` 的 `metrics/E_loc/mean/ewm`，原始批平均能量在 `metrics/E_loc/mean_elec`。原生训练还会把绝对误差 `abs(E_EWM - E_ref)`（Ha）写入 `metrics/E_ewm_vs_ref/mae`。画能量曲线时可叠加参考线，也可画有符号误差 `(E - E_ref) × 1000`（mHa）。单个结构的居中 MARE 恒为零。

独立评估结束后，对全部批平均局部能量取 Huber 稳健均值，再计算绝对误差。

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
