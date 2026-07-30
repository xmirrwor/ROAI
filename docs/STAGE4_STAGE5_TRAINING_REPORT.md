# MiniDuck 阶段 4/5 训练报告

日期：2026-07-30

## 结论

阶段 4 的独立蹲起策略已通过 128 环境确定性验收。阶段 5 的候选策略能完成
“站立 -> 前进 -> 急停 -> 蹲下 -> 站立 -> 后退 -> 急停”，全程零跌倒；其中
前进/后退和 2 秒急停的独立回归严格通过，但 14 秒完整动作链的累计直线误差仍
略超严格门槛，因此阶段 5 保留为可视化验收候选，不写成完全通过。

## 选定检查点

| 阶段 | run | checkpoint | 状态 |
|---|---|---:|---|
| 4 蹲起 | `Jul30_18-56-55_squat_stage4_linehold_to14400` | `model_13900.pt` | 通过 |
| 5 动作切换 | `Jul30_21-20-57_action_switch_stage5_symmetry_to14800` | `model_14780.pt` | 候选 |

阶段 5 使用 75% 后退移动样本强化较弱的后退能力，镜像一致性损失系数为 0.1。
蹲起目标由 0.143 m 平滑切换到 0.132 m；更深的 0.123 m 在测试中会造成跌倒，
因此未采用。

## 128 环境验收

| 指标 | 阶段 4 `13900` | 阶段 5 动作链 `14780` | 阶段 5 急停回归 `14780` |
|---|---:|---:|---:|
| 跌倒率 | 0% | 0% | 0% |
| 平均蹲下深度 | 7.84 mm | 10.82 mm | - |
| P95 站立高度误差 | 1.46 mm | 2.19 mm | - |
| P95 蹲起高度误差 | 3.50 mm | 0.36 mm | - |
| 平均定向速度 | - | 0.0353 m/s | 0.0356 m/s |
| P95 急停终速 | - | 0.0048 m/s | 0.0048 m/s |
| 急停成功率 | - | - | 100% |
| P95 横向偏移 | 1.74 cm | 4.50 cm | 3.88 cm |
| P95 航向误差 | 5.08 deg | 8.49 deg | 3.41 deg |
| 严格验收 | 通过 | 未通过 | 通过 |

动作链严格门槛为横向偏移不超过 4 cm、航向误差不超过 0.1 rad（5.73 deg）。
`14780` 分别超出 0.50 cm 和 2.76 deg；其余动作链指标通过。

## 复现评估

```bash
python legged_panguin/scripts/evaluate_skill_stages.py \
  --task miniduck_flat --headless --num_envs 128 \
  --load_run Jul30_21-20-57_action_switch_stage5_symmetry_to14800 \
  --checkpoint 14780 --protocol action_switch --segment_s 2 \
  --run_label stage5_final_14780_action_switch_128

python legged_panguin/scripts/evaluate_emergency_stop.py \
  --task miniduck_flat --headless --num_envs 128 \
  --load_run Jul30_21-20-57_action_switch_stage5_symmetry_to14800 \
  --checkpoint 14780 --move_s 2 --stop_s 2 --speed_mps 0.08 \
  --run_label stage5_final_14780_emergency_stop_128
```

## 可视化

```bash
python legged_panguin/scripts/play_miniduck.py \
  --task miniduck_flat \
  --load_run Jul30_21-20-57_action_switch_stage5_symmetry_to14800 \
  --checkpoint 14780 --demo action_switch
```

播放脚本现在会遵守 `--checkpoint`，不会自动改播同一 run 中更新但已退化的模型。

## 已弃用尝试

- `14800/14900`：航向一度改善，但跌倒率升至 86.7%/100%，属于过训。
- 100% 镜像推理：横向偏移很小，但前后推进被抵消。
- 继续阶段 0 镜像微调：推进速度持续下降，未替代 `13400`。
- 提高航向/横向控制增益：没有同时改善速度、偏移和航向。

下一步不应继续单纯增加迭代。应在阶段 5 中引入动作切换前后的短时轨迹奖励，
并分别统计前进、后退、站立和蹲起阶段的横向误差，再从 `14740-14780` 区间做
小学习率训练。
