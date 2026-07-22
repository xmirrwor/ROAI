# 团队接口约定

## 数据方向

`RobotTransport.read_state()` -> `MiniDuckAgent.act()` -> `RobotTransport.write_joint_targets()`。智能体内部依次执行 `ObservationBuilder.build()`、`PolicyRunner.infer()` 和 `PolicyRunner.joint_targets()`。

训练中的智能体由三部分组成：`MiniDuck` 环境提供状态、奖励和终止条件；`ActorCritic` 是策略/价值网络；`OnPolicyRunner.learn()` 使用 PPO 更新网络。导出的 ONNX 是训练完成后的 Actor，`MiniDuckAgent` 则是它在仿真和真机上的统一运行外壳。

单位统一使用 SI：角度 rad、角速度 rad/s、加速度 m/s²、时间 s、位置 m。坐标为 `x` 前、`y` 左、`z` 上；正 yaw 为逆时针。

## 观测布局

| 索引 | 内容 | 数量 |
|---|---|---:|
| 0:3 | 机体系 gyro | 3 |
| 3:6 | 机体系 gravity | 3 |
| 6:9 | 机体系 acceleration | 3 |
| 9:12 | vx、vy、yaw_rate 命令 | 3 |
| 12:22 | 关节位置减默认位置 | 10 |
| 22:32 | 关节速度 | 10 |
| 32:62 | 当前及前两帧动作 | 30 |
| 62:64 | gait phase cos、sin | 2 |

所有缩放由策略 JSON 的 `obs_scales` 提供。不要在硬件 adapter 中重复缩放。

## 真机安全边界

策略循环必须保持 20 ms。发生通信超时、NaN/Inf、关节越限或 IMU 倾倒时，调用 `disable_torque()`。首测按悬空低幅度、保护架原地、0.5 m、1 m、2 m 的顺序推进。硬件同学需把急停实现放在 adapter 内，确保上层 Python 卡死时仍可停机。

## 2 m 结果格式

每次测试保存：模型版本、命令速度、action_scale、地面、用时、平均速度、最大/终点横向漂移、终点 yaw 误差、是否摔倒。统一用 `StraightLineRace` 计算终点坐标，避免各组坐标定义不一致。
