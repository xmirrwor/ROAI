# MiniDuck 动作课程路线

目标动作顺序固定为：

1. 急停后稳定站立
2. 蹲起
3. 动作切换
4. 跌倒起身
5. 斜向组合运动
6. 跨障碍
7. 踢球

阶段定义位于 `miniduck_api/action_curriculum.py`，训练参数位于
`MiniDuckFlatCfg.skill_curriculum`。阶段从稳定模型 `model_12000.pt` 对应的
第 288000 个 simulator step 开始，保持现有 64 维观测和 10 维动作契约。

## 当前实现范围

当前开放阶段 0-2，对应急停、蹲起和动作切换。`max_implemented_stage = 2`
仍是安全门；训练步数越过阶段 2 后会继续停留在动作切换，不会进入尚未实现的
跌倒起身。

| 内部阶段 | 训练内容 | 主要实现 |
|---|---|---|
| 0 | 前进/后退后 2 秒急停 | 成对的移动/停止采样、整段直线参考、稳定与停止成功 reward |
| 1 | 站立与蹲起 | 0.143 m/0.132 m 高度目标、0.6 秒平滑转场、高度/稳定/双脚支撑 reward |
| 2 | 动作切换 | 站立/移动/蹲起随机切换、同一 episode 直线参考、切换与速度 reward |

64 维观测的最后两维用于技能条件：移动时为步态相位，站立时为 `[1, 0]`，
蹲起时为高度深度编码。动作维数仍为 10。`forced_stage` 默认是 `None`；它只用于
专项续训时锁定已实现阶段，提交和常规训练不得长期保持为具体数字。

## 后续阶段前置条件

| 阶段 | 尚需实现 |
|---|---|
| 跌倒起身 | 俯卧/仰卧 reset 分布、恢复期间 termination 规则、恢复超时指标 |
| 斜向组合运动 | 同时采样 `vx/vy`、组合跟踪与漂移评测 |
| 跨障碍 | 障碍地形、前方高度观测、足端净空和碰撞指标 |
| 踢球 | 球 actor、球相对状态观测、目标方向和击球结果 reward |

这些前置条件未实现前，不应把对应阶段标记为 `implemented=True`，也不应以一次
smoke test 声称机器人已经掌握该技能。

## 最小验证

服务器环境变量配置完成后，先运行单元测试：

```bash
python -m unittest discover -s tests -v
```

动作切换检查点的确定性评估：

```bash
python legged_panguin/scripts/evaluate_skill_stages.py \
  --task miniduck_flat --headless --num_envs 128 \
  --load_run Jul30_21-20-57_action_switch_stage5_symmetry_to14800 \
  --checkpoint 14780 --protocol action_switch --segment_s 2 \
  --run_label stage5_final_14780_action_switch_128
```

可视化同一检查点：

```bash
python legged_panguin/scripts/play_miniduck.py \
  --task miniduck_flat \
  --load_run Jul30_21-20-57_action_switch_stage5_symmetry_to14800 \
  --checkpoint 14780 --demo action_switch
```

训练结果和未通过项见 `docs/STAGE4_STAGE5_TRAINING_REPORT.md`。
