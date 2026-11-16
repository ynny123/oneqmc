# GitHub 交付版本说明

本版本基于 Microsoft OneQMC 提交 `46ddc5a239650672a3b9eb276b292d3326c79ad5`，提供 LiCN、H₄ 两个单卡从头训练示例，以及乙烷、甲酰胺两个四卡从头训练示例。上游许可证、版权声明和引用信息保留。四个算例的设置、输出说明及指标定义见[训练示例总说明](TRAINING_EXAMPLES_zh.md)。

保留内容：

- 模型、采样、优化和训练源码，以及上游测试。
- 数据集和数据生成 notebook；notebook 的执行输出已清空。
- 乙烷 20 个结构及其参考能量输入 `data/references/ethane_bbmep_reference_energy.csv`。
- LiCN 五个结构及其参考能量输入 `data/references/licn_reference_energy.csv`；参考表每行附上游固定版本来源。
- H₄ 六个矩形结构、生成脚本和 `examples/h4_rectangle/` 单卡入口。
- 甲酰胺 20 点 N–H 断键路径、参考输入 `data/references/formamide_bbmep_reference_energy.csv` 和 `examples/formamide_scratch/` 四卡 32k 步入口，含每分钟刷新的 MARE 后处理及绘图脚本。
- 中文说明、固定依赖、乙烷与 LiCN 的 MARE 后处理，以及乙烷 / H₄ 绘图脚本。

排除内容：

- `experiment_results/` 中已有的论文评估结果、绘图数据和结果图片。
- 本机 `runs/` 软链接、训练日志、HDF5 指标和新生成的 checkpoint。
- 所有模型权重，包括此前随上游提供的 LAC checkpoint；当前文件树不保留 `checkpoints/`。
- 在线 MARE 导出结果、虚拟环境、安装包和缓存。

参考能量是计算误差需要的基准输入，不是本次训练产生的评估结果。配套 MARE 入口已改为读取 `data/references/`，不依赖被移除的目录。

原有论文绘图 notebook 的源代码保留；重新绘制论文图表时，需要另行从上游获取对应的 `experiment_results/`。四个示例的从头训练，以及 LiCN / 乙烷 / 甲酰胺配套 MARE 计算，不需要这些已发布结果或预训练权重。H₄ 曲线直接绘制绝对训练能量（Ha），保留六个结构等权平均和 200 步平滑，不减最低能量。

这是公开 fork 的交付分支。删除结果和权重文件只影响当前文件树；上游及此前提交的历史仍保留原文件，不代表从 Git 历史中清除这些文件。本次更新不新增或修改任何 checkpoint 权重 blob。
