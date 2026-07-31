# MiniDuck 阶段 6/7 与快速运动训练报告

日期：2026-07-31

## 结论

本轮新增了阶段 6 跌倒恢复和阶段 7 斜向组合运动的环境支持、奖励、评估与可视化入口，保持 64 维观测和 10 维动作契约不变。

交付采用独立专家检查点，避免新技能破坏已经验收的急停、蹲起和动作切换：

| 用途 | 检查点 | 状态 |
|---|---|---|
| 急停、蹲起、动作切换 | `miniduck_stage5_action_switch_model_14780.pt` | 保留原稳定专家 |
| 阶段 6 恢复雏形 | `miniduck_stage6_recovery_model_15420.pt` | 达到总体雏形门槛，仍缺一个俯仰方向 |
| 阶段 7 斜向与快速运动 | `miniduck_stage7_diagonal_fast_model_12320.pt` | 动作稳定可辨，严格轨迹门槛略未通过 |

所有策略均同时提供 `.pt`、`.onnx` 和 `.json` 元数据。

## 速度问题

旧阶段 5 策略在 5 秒前进/后退协议中的末段平均定向速度约为 `0.015 m/s`，确实存在实际速度不足；持续跟随机器人的相机又进一步造成“原地踏步”的观感。

队友 `main` 的 `model_12000.pt` 可达到约 `0.126-0.149 m/s`，但直接用于当前动作链会摔倒和明显偏航。短程单策略速度强化又把急停末速恶化到 `0.65-0.80 m/s`，因此这些候选全部弃用。

最终快速运动专家 `12320` 的 128 环境前进/后退急停评估为：

| 指标 | 结果 |
|---|---:|
| 移动末段平均定向速度 | `0.1037 m/s` |
| 摔倒率 | `0%` |
| 2 秒急停成功率 | `100%` |
| 平均稳定时间 | `0.426 s` |
| P95 最终线速度 | `0.0036 m/s` |
| P95 横向偏移 | `6.58 cm` |
| P95 航向误差 | `13.38 deg` |

速度和急停已经达到可视化要求，但横向偏移仍高于 `4 cm`、航向误差仍高于 `5.73 deg` 的严格直线门槛。它是快速运动专家，不替换原阶段 5 的严格急停专家。

## 阶段 6：恢复雏形

实现内容：

- 从正负 pitch 倾斜姿态重置；训练分布为 `0.50-0.70 rad`。
- 恢复期间关闭常规倾倒终止，设置 5 秒超时。
- 连续 0.5 秒达到高度、倾角和角速度条件才记为成功。
- 新增恢复对齐、高度、成功和成功后稳定奖励。
- 最后两个观测位使用 `[-1, -1]` 表示恢复技能。

`15420` 的 128 环境确定性评估总体成功率为 `55.47%`，成功样本平均用时 `0.678 s`。分方向检查显示一个俯仰方向 `100%`，相反方向 `0%`。因此该检查点只标记为单方向恢复动作雏形，不代表完整俯卧/仰卧起身已经完成。

## 阶段 7：斜向组合运动

实现内容：

- 同时采样正负 `vx` 和 `vy`，覆盖四个斜向方向。
- 用航向闭环避免把斜向平移学成转弯。
- 新增斜向速度跟踪、方向进度、航向误差和累计路径误差奖励。

`12320` 的 128 环境、5 秒四向斜走评估为：

| 指标 | 结果 |
|---|---:|
| 摔倒率 | `3.91%` |
| 平均定向速度 | `0.1571 m/s` |
| P95 最大路径偏差 | `0.1453 m` |
| P95 最大航向误差 | `12.13 deg` |

该策略满足“动作雏形且具有一定稳定性”的本轮要求。路径偏差比严格 `0.12 m` 门槛高 `2.53 cm`，后续需要继续做四方向分别统计和路径保持微调。

## 可视化

在已打开 `DISPLAY=:20` 的服务器桌面中，从仓库根目录执行。每个新的 SSH 会话先加载 Conda 和 Isaac Gym 动态库：

```bash
cd /root/miniducktraining/ROAI
source /opt/conda/etc/profile.d/conda.sh
conda activate miniduck_cu118
export DISPLAY=:20
export LD_LIBRARY_PATH="$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$CONDA_PREFIX/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# 正常速度前进、后退和 2 秒急停；固定相机可直接看见位移
python legged_panguin/scripts/play_miniduck.py \
  --task miniduck_flat \
  --checkpoint_path_override checkpoints/miniduck_stage7_diagonal_fast_model_12320.pt \
  --demo emergency_stop --fixed_camera \
  --heading_hold_kp 6 --cross_track_heading_kp 2

# 阶段 6 恢复雏形；环境会自动重复采样两种俯仰方向
python legged_panguin/scripts/play_miniduck.py \
  --task miniduck_flat \
  --checkpoint_path_override checkpoints/miniduck_stage6_recovery_model_15420.pt \
  --demo fall_recovery

# 阶段 7 四向斜走
python legged_panguin/scripts/play_miniduck.py \
  --task miniduck_flat \
  --checkpoint_path_override checkpoints/miniduck_stage7_diagonal_fast_model_12320.pt \
  --demo diagonal_motion --fixed_camera

# 原有稳定动作链
python legged_panguin/scripts/play_miniduck.py \
  --task miniduck_flat \
  --checkpoint_path_override checkpoints/miniduck_stage5_action_switch_model_14780.pt \
  --demo action_switch --fixed_camera
```

## 复现评估

```bash
python legged_panguin/scripts/evaluate_advanced_stages.py \
  --task miniduck_flat --headless --num_envs 128 \
  --protocol fall_recovery --duration_s 5 \
  --load_run Jul31_16-26-44_stage6_true_supine_to15560 \
  --checkpoint 15420 --run_label stage6_true_balanced_15420

python legged_panguin/scripts/evaluate_advanced_stages.py \
  --task miniduck_flat --headless --num_envs 128 \
  --protocol diagonal_motion --duration_s 5 \
  --load_run Jul31_16-42-54_stage7_diagonal_path_to12400 \
  --checkpoint 12320 --run_label stage7_path_12320
```

评估保存原始逐步轨迹、逐试验结果、JSON 指标和无平滑图表；摔倒样本不会从总体统计中删除。
