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

## 本次实现范围

当前只开放阶段 0：急停后稳定站立。命令采样由 80% 零速度命令和 20%
低速前后运动探针组成，使训练能观察“运动后切换到停止命令”的场景。新增稳定性
reward 仅在阶段 0 且命令为零时惩罚机身线速度和角速度。

`max_implemented_stage = 0` 是安全门。即使训练步数越过后续阶段的计划边界，
训练也会继续停留在阶段 0，直到相应场景、观测、reward 和验收测试完成。

## 后续阶段前置条件

| 阶段 | 尚需实现 |
|---|---|
| 蹲起 | 高度命令、逐环境高度目标、蹲起轨迹与双脚稳定 reward |
| 动作切换 | 不破坏 64 维部署契约的 skill conditioning、转场采样器 |
| 跌倒起身 | 俯卧/仰卧 reset 分布、恢复期间 termination 规则、恢复超时指标 |
| 斜向组合运动 | 同时采样 `vx/vy`、组合跟踪与漂移评测 |
| 跨障碍 | 障碍地形、前方高度观测、足端净空和碰撞指标 |
| 踢球 | 球 actor、球相对状态观测、目标方向和击球结果 reward |

这些前置条件未实现前，不应把对应阶段标记为 `implemented=True`，也不应以一次
smoke test 声称机器人已经掌握该技能。

## 最小验证

从同学的稳定 checkpoint 续训 1 次 iteration：

```bash
python legged_panguin/scripts/train.py \
  --task miniduck_flat --headless --num_envs 8 --max_iterations 1 \
  --resume \
  --load_run Jul17_04-41-06_miniduck_stable_compressed_curriculum_resume_to_12000_from_8850 \
  --checkpoint 12000 \
  --run_name action_curriculum_stage0_smoke
```

日志必须出现：

```text
MiniDuck action curriculum stage: 0:emergency_stop_stand
Learning iteration 12000/12001
```
