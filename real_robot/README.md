# MiniDuck 真机工作区

真机代码与 Isaac Gym 仿真代码隔离。本目录可以复用导出的 ONNX 策略和
`miniduck_api` 中稳定的 observation/action 契约，但所有与实体机器人有关的
数值只允许放在 `config/robot.json`。

## 当前状态

已按照仿真的 10 关节顺序建立真机映射：

| 策略关节 | 舵机 ID |
|---|---:|
| left_hip_yaw ... left_ankle | 20 ... 24 |
| right_hip_yaw ... right_ankle | 10 ... 14 |

舵机 ID 来自装配说明。零偏、正方向和机械限位尚未经过实体标定，因此配置中的
`calibrated` 默认为 `false`。在全部关节标定完成前，不应接通策略自动控制。

## 配置检查

```powershell
cd real_robot
python -m scripts.check_config
```

## 接入真实 SDK

1. 实现 `ServoBus`：读取 10 个舵机的位置/速度并发送目标位置。
2. 实现 `ImuSource`：输出机体坐标系下的 gyro、gravity 和 acceleration。
3. 将它们传给 `MiniDuckHardwareAdapter`。
4. 完成每个关节的 `zero_offset_rad`、`direction`、机械限位和速度测试。
5. 最后才通过 `miniduck_api.MiniDuckAgent.run()` 启动策略循环。

真机适配层负责单位转换、关节重排、限位和急停；策略层始终只处理 SI 单位和
metadata 关节顺序。
