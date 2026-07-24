# MiniDuckTraining 独立训练包说明

这个目录是 MiniDuck 专用训练包，目标是让别人拿到 `miniducktraining` 文件夹后，不依赖原始开发机上的其他工程或其他机器人资源，也能复现当前 MiniDuck 训练和导出流程。

## 目录结构

```text
miniducktraining/
  README.md
  exported_policy/
  legged_panguin/
    data/best_walk_teacher_fullgrid_action_no_head_legs_v2.pkl
    legged_panguin/envs/base/
    legged_panguin/envs/miniduck/
    legged_panguin/scripts/train.py
    legged_panguin/scripts/play_miniduck.py
    legged_panguin/scripts/export_miniduck_policy_onnx.py
    resources/robots/miniduck/
    third_party/isaacgym/python/
    third_party/rsl_rl/
    logs/flat_miniduck/
```

只保留了 MiniDuck 训练需要的内容。`base` 是 MiniDuck 环境继承的基础类，不是另一个机器人环境。

## 环境准备

建议使用和原训练一致的环境：

```text
Ubuntu 20.04/22.04
NVIDIA GPU
Python 3.8
PyTorch 1.10.2 + CUDA 11.3
Isaac Gym Preview 4
```

创建 conda 环境。这里用 `miniducktraining_zero` 作为环境名：

```bash
conda create -n miniducktraining_zero python=3.8 -y
conda activate miniducktraining_zero

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
  torchaudio==0.10.2+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install numpy==1.23.5 scipy matplotlib tensorboard onnx onnxruntime tqdm \
  pyyaml ninja imageio wandb torchinfo
```

安装本包里的 Isaac Gym、rsl_rl 和训练工程：

```bash
cd /path/to/miniducktraining/legged_panguin
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

pip install -e third_party/isaacgym/python
pip install -e third_party/rsl_rl
pip install -e .
```

如果自检或训练时报：

```text
ModuleNotFoundError: No module named 'wandb'
ModuleNotFoundError: No module named 'torchinfo'
```

说明环境缺少 rsl_rl runner 的辅助包，执行：

```bash
pip install wandb torchinfo
```

如果手写自检脚本时报：

```text
ImportError: PyTorch was imported before isaacgym modules.
```

不要先 `import torch`，要先导入 `legged_panguin.envs` 或 Isaac Gym 相关模块。正式训练脚本已经按正确顺序导入。

如果你绕过 `conda activate`，直接用绝对路径调用 Python，可能会看到：

```text
RuntimeError: Ninja is required to load C++ extensions
```

这通常不是没装 `ninja`，而是环境的 `bin` 没进 `PATH`。需要：

```bash
export CONDA_PREFIX=/path/to/miniconda3/envs/miniducktraining_zero
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

正常按教程 `conda activate miniducktraining_zero` 时，这些路径会自动处理。

如果训练只跑了几轮就停止，但日志末尾没有 Python 报错，也没有 `train_exit_status=...`，通常是后台启动方式没有真正脱离当前 shell。长时间训练不要直接依赖普通 `nohup`，优先使用下面的 detached 后台启动脚本。

## 快速自检

确认只注册了 MiniDuck task：

```bash
cd /path/to/miniducktraining/legged_panguin
conda activate miniducktraining_zero
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import os
import legged_panguin
from legged_panguin.utils import task_registry
import legged_panguin.envs
import torch

print("root:", legged_panguin.LEGGED_GYM_ROOT_DIR)
print("tasks:", sorted(task_registry.task_classes.keys()))
print("teacher exists:", os.path.exists("data/best_walk_teacher_fullgrid_action_no_head_legs_v2.pkl"))
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY
```

期望看到：

```text
tasks: ['miniduck_flat']
teacher exists: True
torch: 1.10.2+cu113 cuda: 11.3
```

正式长时间训练前，建议先跑 1 个 iteration 的 smoke test。这个测试可以提前发现 Isaac Gym、CUDA、`ninja`、Python 包和资源路径问题：

```bash
cd /path/to/miniducktraining/legged_panguin
conda activate miniducktraining_zero
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python legged_panguin/scripts/train.py \
  --task miniduck_flat \
  --headless \
  --num_envs 8 \
  --max_iterations 1 \
  --run_name miniduck_smoke_test
```

期望看到类似：

```text
Using teacher reference trajectory
Learning iteration 0/1
```

如果 smoke test 通过，再开始正式训练。

## 从零训练

当前独立包使用压缩课程，目标是让从零训练 `12000` iteration 就完整经历前进/后退、左右平移、左右旋转、三组动作混合、advanced 速度范围和 sim-to-real 扰动。课程按 simulator step 切换，而每个 PPO iteration 是 24 个 simulator step，所以大致对应关系是：

```text
0-1000 iteration:     前进/后退和前进转向
1000-2000 iteration:  纯左右平移
2000-3000 iteration:  原地左右旋转
3000-5000 iteration:  三组动作混合巩固，基础速度范围
5000-6000 iteration:  advanced 左右平移强化
6000-7000 iteration:  advanced 原地旋转强化
7000-8000 iteration:  advanced 三组动作平衡
8000-12000 iteration: advanced final 混合命令
5000-7000 iteration:  x/y 速度逐步扩大到 x [-0.08, 0.12]、y [-0.20, 0.20]
6000-9000 iteration:  yaw 速度逐步扩大到 [-1.00, 1.00]
500-4000 iteration:   电机强度、PD、编码器偏置、IMU 噪声等 sim-to-real 随机化逐步打开
4000-10000 iteration: push 扰动逐步打开到 0.06 m/s
```

本教程只描述从 0 开始训练这套 `12000` iteration 压缩课程。强化学习训练存在随机性，不同显卡、驱动和 Isaac Gym 运行细节可能让最终动作有小幅差异；但按照下面的环境、代码和训练命令执行，使用的是同一套 MiniDuck URDF、reward、teacher reference、sim-to-real 随机化和课程逻辑。

```bash
cd /path/to/miniducktraining/legged_panguin
conda activate miniducktraining_zero
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
mkdir -p logs/train_stdout

stdbuf -oL -eL python legged_panguin/scripts/train.py \
  --task miniduck_flat \
  --headless \
  --num_envs 4096 \
  --max_iterations 12000 \
  --run_name miniduck_stable_compressed_curriculum_12000 \
  2>&1 | tee logs/train_stdout/miniduck_stable_compressed_curriculum_12000.log
```

如果只是验证这个包能从 0 开始训练，可以先跑 1 iteration smoke test。正式 12000 训练命令就是最终课程。

```bash
cd /path/to/miniducktraining/legged_panguin
conda activate miniducktraining_zero
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
mkdir -p logs/train_stdout

stdbuf -oL -eL python legged_panguin/scripts/train.py \
  --task miniduck_flat \
  --headless \
  --num_envs 4096 \
  --max_iterations 1 \
  --run_name miniduck_smoke_test \
  2>&1 | tee logs/train_stdout/miniduck_smoke_test.log
```

本包也提供了通用训练脚本。默认是面向最终效果的 `12000` iteration：

```bash
cd /path/to/miniducktraining/legged_panguin
MINIDUCK_RESET_LOG=1 bash scripts/launch_miniduck_training_detached.sh
```

如果要覆盖默认实验名：

```bash
cd /path/to/miniducktraining/legged_panguin
MINIDUCK_RUN_NAME=my_miniduck_training \
MINIDUCK_MAX_ITERATIONS=12000 \
MINIDUCK_RESET_LOG=1 \
bash scripts/launch_miniduck_training_detached.sh
```

推荐的长时间训练方式是 detached 后台启动。这样关闭终端也不会中断训练：

```bash
cd /path/to/miniducktraining/legged_panguin
MINIDUCK_RESET_LOG=1 bash scripts/launch_miniduck_training_detached.sh
```

检查训练是否仍在运行：

```bash
ps -p "$(cat logs/train_stdout/miniduck_stable_compressed_curriculum_12000.pid)"
tail -f logs/train_stdout/miniduck_stable_compressed_curriculum_12000.log
```

日志里持续出现新的 `Learning iteration 当前轮数/目标轮数` 就说明训练在推进。刚开始的 curriculum 主要让机器人先学稳定前后步态，因此前期看到 `rew_tracking_lin_vel_y=0.0000`、`rew_tracking_ang_vel_yaw=0.0000` 是正常现象；后续课程打开侧移和旋转命令后，这些值才会变成主要指标。

脚本默认参数是 `num_envs=4096`、`max_iterations=12000`。如果显存不够，可以降低并行环境数量：

```bash
MINIDUCK_NUM_ENVS=2048 \
MINIDUCK_RESET_LOG=1 \
bash scripts/launch_miniduck_training_detached.sh
```

如果要换一个实验名：

```bash
MINIDUCK_RUN_NAME=miniduck_cold_start_test \
MINIDUCK_RESET_LOG=1 \
bash scripts/launch_miniduck_training_detached.sh
```

如果你的 conda 环境不在默认位置，可以这样指定：

```bash
MINIDUCK_CONDA_PREFIX=/path/to/miniconda3/envs/miniducktraining_zero \
  bash scripts/launch_miniduck_training_detached.sh
```

训练完成后的 checkpoint 通常在新的 run 目录里，例如：

```text
logs/flat_miniduck/<日期时间>_miniduck_stable_compressed_curriculum_12000/model_12000.pt
```

实际目录名会带当前日期和时间，以你机器上生成的目录为准。

## 可视化

训练完成后，先找到刚生成的 run 目录名：

```bash
cd /path/to/miniducktraining/legged_panguin
ls -td logs/flat_miniduck/*miniduck_stable_compressed_curriculum_12000 | head -n 1
```

然后把下面命令里的 `<训练目录名>` 替换成上一步输出路径最后一段：

```bash
cd /path/to/miniducktraining/legged_panguin
conda activate miniducktraining_zero
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python legged_panguin/scripts/play_miniduck.py \
  --task miniduck_flat \
  --load_run <训练目录名>
```

可视化脚本会按顺序播放前进、后退、左右平移和左右旋转。

## 导出 ONNX

导出从 0 训练出来的 12000 压缩课程模型。把 `<你的训练目录>` 替换成实际 run 目录名：

```bash
cd /path/to/miniducktraining/legged_panguin
conda activate miniducktraining_zero
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python legged_panguin/scripts/export_miniduck_policy_onnx.py \
  --checkpoint_path logs/flat_miniduck/<你的训练目录>/model_12000.pt \
  --output_onnx ../exported_policy/miniduck_compressed_curriculum_policy.onnx \
  --output_metadata ../exported_policy/miniduck_compressed_curriculum_policy.json
```

检查 ONNX：

```bash
python - <<'PY'
import numpy as np
import onnxruntime as ort

p = "../exported_policy/miniduck_compressed_curriculum_policy.onnx"
s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
out = s.run(None, {"obs": np.zeros((1, 64), dtype=np.float32)})[0]
print("input", s.get_inputs()[0].shape)
print("output", out.shape)
PY
```

期望输出：

```text
input ['batch', 64]
output (1, 10)
```

## 打包给别人

从 `/home/robot` 或任意 `miniducktraining` 的上级目录执行：

```bash
tar --hard-dereference -czf miniducktraining.tar.gz miniducktraining
```

`--hard-dereference` 会把硬链接实际打进压缩包，避免接收方依赖你本机原目录。

## 重要文件

```text
MiniDuck 环境:
  legged_panguin/legged_panguin/envs/miniduck/miniduck.py
  legged_panguin/legged_panguin/envs/miniduck/miniduck_config.py

MiniDuck URDF:
  legged_panguin/resources/robots/miniduck/urdf/MINIDUCK.urdf

老师参考数据:
  legged_panguin/data/best_walk_teacher_fullgrid_action_no_head_legs_v2.pkl

训练输出:
  legged_panguin/logs/flat_miniduck/<训练目录>/model_12000.pt

导出 policy:
  exported_policy/miniduck_compressed_curriculum_policy.onnx
```
