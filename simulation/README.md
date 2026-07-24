# MiniDuck 仿真工作区

本目录是仿真侧的稳定入口。现有 Isaac Gym 实现暂时保留在仓库根目录的
`legged_panguin/`、`resources/` 和 `third_party/` 中，避免移动第三方依赖后破坏
已有训练脚本及 checkpoint 路径。

## 入口

```bash
cd simulation
python train.py --task miniduck_flat
python play.py --task miniduck_flat
```

仿真专属参数保存在 `config/runtime.json`。训练环境的详细参数仍由
`legged_panguin/envs/miniduck/miniduck_config.py` 定义。真机标定值不得写入仿真
配置。

共享边界只有导出的策略文件及其 metadata：

- 输入：64 维 observation
- 输出：10 维 action
- 单位：SI（rad、rad/s、m/s、s）
- 关节顺序：以策略 metadata 的 `joint_names` 为准

当前目录已经包含完整的 Isaac Gym、mesh、训练代码和 checkpoint，可作为独立
仿真工作区使用。
