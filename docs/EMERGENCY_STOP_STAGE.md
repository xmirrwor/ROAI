# 阶段 0-3：急停后稳定站立

这一阶段从同学的 `model_12000.pt` 继续训练，不改变 64 维观测、10 维动作和
20 ms 策略周期。范围截止到“低速运动后收到零速度命令，快速恢复稳定站立”，
不包含蹲起。

## 四个阶段的交付

0. 固化基线：使用 `model_12000.pt`，保存训练前专项评估结果。
1. 轨迹与验收：`evaluate_emergency_stop.py` 执行前进/后退各半的并行测试，保留
   每步聚合轨迹、逐试验结果、JSON 指标和 PNG 图。
2. 接口安全：部署接口将速度命令限制在训练范围内，并按电机额定速度限制相邻
   策略周期的关节目标变化；异常仍会触发 `disable_torque()`。
3. 急停训练：每个运动探针后强制进入零命令段；零命令下同时优化残余线速度、
   角速度和稳定成功率，并将循环步态相位固定到标称站立相位，避免零命令时继续
   原地踏步；从 iteration 12000 续训 500 个 PPO iteration。

## 验收协议

- 128 个并行试验，一半前进、一半后退。
- 运动 2 s，随后零命令 6 s。
- 连续 0.5 s 同时满足：线速度 `< 0.035 m/s`、角速度 `< 0.25 rad/s`、
  相对标称姿态倾角 `< 12 deg`，记为稳定。
- 防止“原地不动也算急停”：停止前最后 0.5 s 的平均有向速度至少为
  `0.02 m/s`。
- 跌倒率不高于 2%，稳定成功率至少 95%，第 95 百分位稳定时间不超过 2.5 s。

上述阈值是当前仿真阶段的工程验收线，不代表真机安全认证。训练前后必须使用
相同协议、环境数和随机种子比较，不平滑轨迹，不删除跌倒试验。

## 训练

```bash
cd /root/miniducktraining/ROAI
MINIDUCK_RUN_NAME=emergency_stop_stage3_12500 \
  bash scripts/launch_emergency_stop_training_detached.sh

tail -f logs/train_stdout/emergency_stop_stage3_12500.log
```

训练脚本默认从同学的 iteration 12000 checkpoint 续训 500 iteration，并使用
4096 个环境。完成后 run 目录中的最终文件应为 `model_12500.pt`。

## 专项评估

```bash
python legged_panguin/scripts/evaluate_emergency_stop.py \
  --task miniduck_flat --headless --num_envs 128 \
  --load_run <run目录名> --checkpoint <iteration> \
  --run_label <标签> --output_dir evaluation/emergency_stop
```

加入 `--randomized` 可保留训练时的传感器噪声和动力学随机化，作为第二轮鲁棒性
检查。输出的 CSV 是图表底层数据，PNG 只作呈现。

## 可视化回放

在服务器桌面已经可访问、`DISPLAY=:20` 时执行：

```bash
DISPLAY=:20 python legged_panguin/scripts/play_miniduck.py \
  --task miniduck_flat \
  --load_run <run目录名> --checkpoint <iteration> \
  --demo emergency_stop
```

也可以自动选择最新的已完成急停 run 并在后台启动：

```bash
bash scripts/launch_emergency_stop_visualization.sh
```

窗口会循环显示：稳定站立、向前探针、急停、向后探针、急停。终端同步打印当前
阶段，便于观察零命令切换时刻。
