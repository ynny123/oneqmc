# Globe H₄(6) 的 Orbformer 单卡实验

对应 Gao & Günnemann, *Generalizing Neural Wave Functions* (ICML 2023), Fig. 3 的六个氢矩形构型。四个 H 原子位于 `(±x, ±y, 0)`，`x=R cos(theta/2)`、`y=R sin(theta/2)`，`R=3.2843 bohr`，`theta=85,86,87,88,89,89.999` 度。R 是中心到原子的距离，不是边长；六个构型均为中性、上下自旋各 2 个电子。

数据定义按 [Globe 固定版本的生成函数](https://github.com/n-gao/globe/blob/942feba753a0ab912c23fdebc873ff9aefeb49f6/globe/systems/__init__.py#L29) 和[实验配置](https://github.com/n-gao/globe/blob/942feba753a0ab912c23fdebc873ff9aefeb49f6/configs/seml/train_hydrogen.yaml)生成。它不同于 OneQMC 自带的 LAC 50 点 H₄ 路径。

本实验使用完整 `orbformer-se`，从随机参数开始、seed 219、不加载外部 checkpoint、HF 预训练 0 步。单张 A100 上每步处理全部 6 个 geometry，每个 geometry 256 个电子构型，全局 1536 个构型；K-FAC 更新 16000 步。基础学习率 0.05、damping 0.001、norm constraint 0.001、16 个行列式、每核 8 个包络、Metropolis 去相关 60 步。训练启动时采样器平衡上限 500 步，每 2000 步保存阶段 checkpoint。

这是用 Orbformer 训练相同几何结构的实验；原文使用 Globe+Moon/FermiNet、HF 预训练和不同采样及优化设置，训练步数不能直接等同原图的成本或精度。六点全部参与训练，此设置不包含未见几何测试集，也不计算缺少参考基线的 `E-E_min` 或 MARE。

使用已经安装好的 OneQMC 环境，在仓库根目录运行：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export ORBFORMER_PICKLE_LOADING=1
python examples/h4_rectangle/generate.py

# 输出目录须不存在或为空；stdout 日志放在输出目录之外。
CUDA_VISIBLE_DEVICES=0 python examples/h4_rectangle/run.py \
  --mode smoke --output runs/h4_globe6_smoke

CUDA_VISIBLE_DEVICES=0 python examples/h4_rectangle/run.py \
  --mode train --output runs/h4_globe6_scratch_1gpu_16k_s219
```

`smoke` 也使用完整的 6×256 batch，但只更新两步、初始采样器平衡 1 步，供运行检查，不作为精度评估。会检查模型参数发生更新、参数和记录的能量有限、六个构型都被访问、checkpoint 可重载。正式任务从新的随机参数开始，不接续 smoke 输出。

预览命令可加 `--dry-run`。若想先检查每步 1 个 geometry，可增加 `--mol-batch-size 1`。默认正式实验为 6；其他参数不变。

续训使用原输出目录、相同总步数和 batch，并增加 `--resume`。脚本检查保存的 `training/config.yaml`，阻止配置不一致的续训。

主要输出：

- `launch.json`：启动参数和六个结构的 SHA256。
- `runtime_status.json`：运行阶段，结束后的更新数、checkpoint 重载和显存统计。
- `oneqmc_train.log`：上游日志。
- `training/config.yaml`：上游实际配置。
- `training/result.h5`：逐步能量、采样和耗时；只读查看时使用 HDF5 SWMR。
- `training/chkpt-2000.pt` 等阶段 checkpoint；完成时为 `chkpt-16000.pt`。
- `training/chkpt--1.pt`：最新快速 checkpoint。

六个 geometry 的文件索引 00–05 对应上述角度顺序。检查程序完整运行与判断能量收敛是两项不同的验收：后者还需要固定模型后的独立采样和一致的参考能量。

已有实验的能量曲线位于 `runs/h4_globe6_scratch_1gpu_16k_s219/results/analysis/h4_convergence.png`。曲线直接显示六个 geometry 等权平均后的绝对训练能量，使用 200 步向后滑动平均，单位 Hartree，不减去最低能量。`h4_convergence_fullrange.png` 另叠加从第 1 步开始的未平滑平均能量。数据及三个训练示例的设置、输出位置见[训练示例总说明](../../docs/TRAINING_EXAMPLES_zh.md)。
