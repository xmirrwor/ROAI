# MiniDuck 2 m 直线竞速优化 v1

本版本直接修改原始 zip 中的 `legged_panguin` 环境，不替换 Isaac Gym、PPO、teacher gait 或机器人 URDF。

## 修改内容

- 最终前进命令上限：`0.12 -> 0.16 m/s`
- 最终课程的前进样本比例：`0.25 -> 0.60`
- 前进伴随转向概率：`0.55 -> 0.15`
- 动作幅度：通过课程从 `0.10` 增长到 `0.25`
- 新增纯前进累计航向误差奖励
- 新增纯前进累计横向位移奖励
- 加强前进速度跟踪、迈步进度和打滑惩罚
- 略微放松动作变化惩罚，使策略能形成更有力的迈步动作

## 服务器训练

先做最小测试：

```bash
MINIDUCK_RUN_NAME=miniduck_race_v1_smoke \
MINIDUCK_NUM_ENVS=8 \
MINIDUCK_MAX_ITERATIONS=1 \
bash scripts/run_miniduck_training.sh
```

再从零正式训练，保留原始模型：

```bash
MINIDUCK_RESET_LOG=1 bash scripts/launch_miniduck_training_detached.sh
tail -f logs/train_stdout/miniduck_race_v1_12000.log
```

## 公平评测

分别对原始 run 和 `miniduck_race_v1_12000` 使用完全相同的参数：

```bash
python legged_panguin/scripts/evaluate_miniduck.py \
  --task miniduck_flat --headless --num_envs 64 \
  --load_run miniduck_race_v1_12000 \
  --distance_m 2.0 --speed_mps 0.12 --timeout_s 30 \
  --output_json race_v1_speed_012.json

python legged_panguin/scripts/evaluate_miniduck.py \
  --task miniduck_flat --headless --num_envs 64 \
  --load_run miniduck_race_v1_12000 \
  --distance_m 2.0 --speed_mps 0.16 --timeout_s 30 \
  --output_json race_v1_speed_016.json
```

首轮通过标准：完成率不低于 85%，摔倒率不高于 5%，平均绝对横向漂移不高于 0.10 m。未达到标准时不要继续扩大到 `0.20 m/s`，应先检查哪一项奖励或动力学限制导致失败。

## 真机前置条件

只有仿真通过后才导出 ONNX。真机从 `0.04 m/s` 开始，依次测试 `0.06/0.08/0.10/0.12/0.14/0.16 m/s`。每档均检查舵机目标-实际误差、控制周期、供电电压、横向漂移和急停。
