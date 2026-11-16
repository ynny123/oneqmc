# GitHub 交付版本说明

本版本基于 Microsoft OneQMC 提交 `46ddc5a239650672a3b9eb276b292d3326c79ad5`，增加乙烷多几何从头训练的运行说明、固定依赖和可移植示例入口。上游许可证、版权声明和引用信息保留。

保留内容：

- 模型、采样、优化和训练源码，以及上游测试。
- 数据集和数据生成 notebook；notebook 的执行输出已清空。
- 乙烷 20 个结构及其参考能量输入 `data/references/ethane_bbmep_reference_energy.csv`。
- 原仓库公开的 LAC checkpoint，供上游功能使用；乙烷从头训练不加载它。
- [乙烷运行说明](ETHANE_SCRATCH_RUNBOOK_zh.md)、固定依赖和 `examples/ethane_scratch/` 配套脚本。

排除内容：

- `experiment_results/` 中已有的论文评估结果、绘图数据和结果图片。
- 本机 `runs/` 软链接、训练日志、HDF5 指标和新生成的 checkpoint。
- 在线 MARE 导出结果、虚拟环境、安装包和缓存。

参考能量是计算误差需要的基准输入，不是本次训练产生的评估结果。配套 MARE 入口已改为读取 `data/references/`，不依赖被移除的目录。

原有论文绘图 notebook 的源代码保留；重新绘制论文图表时，需要另行从上游获取对应的 `experiment_results/`。乙烷从头训练与配套 MARE 计算不需要这些已发布结果。

这是公开 fork 的交付分支。删除结果文件只影响该分支的当前文件树；上游历史仍保留原文件，不代表从 Git 历史中清除这些文件。
