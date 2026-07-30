# MiniDuck 强化学习初版项目

本项目覆盖四条链路：Isaac Gym 训练、仿真回放、ONNX 导出、真机部署接口。课程考核目标暂按 **约 2 m 直线竞速** 定义，核心指标为用时、平均速度、横向漂移和终点朝向误差。

机器人已经封装为 `miniduck_api.MiniDuckAgent`。它对外只接收 `RobotState + Command`，输出10个关节目标；相同智能体可以连接仿真 transport 或真机 transport。

## 团队分工与稳定接口

| 角色 | 主要入口 | 交付内容 |
|---|---|---|
| 训练/奖励 | `legged_panguin/envs/miniduck/miniduck_config.py`、`miniduck.py` | checkpoint |
| 仿真/评测 | `play_miniduck.py`、`miniduck_api.benchmark` | 2 m 指标记录 |
| 模型部署 | `export_miniduck_policy_onnx.py` | `.onnx` + 同名 `.json` |
| 真机 | `miniduck_api.contracts.RobotTransport` | 硬件 adapter |

团队不可随意修改的策略契约：64 维 `float32` 观测、10 维动作、20 ms 策略周期、关节顺序以策略 JSON 的 `joint_names` 为准。真机同学只实现 `read_state()`、`write_joint_targets()`、`disable_torque()`，不需要依赖 Isaac Gym。

## 快速检查（本机）

```powershell
python -m unittest discover -s tests -v
python -m tools.check_policy --model miniduck_stable_policy.onnx --metadata miniduck_stable_policy.json
```

## Ubuntu 训练服务器

Windows 无法完整还原压缩包内 Isaac Gym 的 Linux 符号链接。上传服务器时请直接解压原始 `miniducktraining.zip`，然后在 `legged_panguin` 根目录安装：

```bash
pip install -e third_party/isaacgym/python
pip install -e third_party/rsl_rl
pip install -e .
python legged_panguin/scripts/train.py --task miniduck_flat --headless --num_envs 8 --max_iterations 1 --run_name smoke_test
```

正式训练和可视化命令见 `MINIDUCKTRAINING_GUIDE.md`。该文件来自原压缩包，部分中文在 Windows 终端显示可能存在编码问题，但不影响代码。

## 仿真性能测试（第一阶段）

在 Ubuntu/NVIDIA/Isaac Gym 环境中，对一个 checkpoint 并行执行 64 次约 2 m 直线测试：

```bash
python legged_panguin/scripts/evaluate_miniduck.py \
  --task miniduck_flat --headless --num_envs 64 \
  --load_run <训练目录名> \
  --distance_m 2.0 --speed_mps 0.10 --timeout_s 40 \
  --output_json benchmark_result.json
```

输出包括完成率、摔倒数、超时数、平均完成时间、平均速度、横向漂移和航向误差。调步幅或奖励后必须用相同参数复测，才能公平比较 checkpoint。

## 竞速建议

先保证直线，再提高步幅/速度：训练命令只给 `vx > 0, vy = 0, yaw = 0`；横向速度、横向位移和 yaw 误差需进入奖励或评测。真机首次运行将 `vx` 从 0.04 m/s 逐步提高，动作幅度优先通过 `action_scale` 小步调整，并保留急停与悬空测试。

接口细节见 [docs/TEAM_API.md](docs/TEAM_API.md)。

本项目在原 zip 训练环境上加入的首轮竞速优化、训练命令和验收阈值见 [docs/RACE_OPTIMIZATION.md](docs/RACE_OPTIMIZATION.md)。默认新实验名为 `miniduck_race_v1_12000`，不会覆盖原始稳定模型。

## 动作课程路线

从稳定模型继续开发“急停站立 → 蹲起 → 动作切换 → 跌倒起身 → 斜向运动 →
跨障碍 → 踢球”的阶段顺序、当前实现边界和验收要求见
[docs/ACTION_CURRICULUM.md](docs/ACTION_CURRICULUM.md)。

阶段 0-3（截止到急停后稳定站立）的训练、专项轨迹指标、安全接口和可视化命令见
[docs/EMERGENCY_STOP_STAGE.md](docs/EMERGENCY_STOP_STAGE.md)。

2026-07-30 的实际训练过程、失败尝试、最终指标和适用边界见
[docs/EMERGENCY_STOP_TRAINING_REPORT.md](docs/EMERGENCY_STOP_TRAINING_REPORT.md)。

阶段 4（蹲起）和阶段 5（动作切换）的检查点、128 环境验收结果、复现命令及
已知边界见 [docs/STAGE4_STAGE5_TRAINING_REPORT.md](docs/STAGE4_STAGE5_TRAINING_REPORT.md)。
