# SonicStar VLA-JEPA 自动化评测交接文档

本文档面向项目接手人，说明如何使用原始未魔改权重运行 SonicStar 的自动化评测脚本。

当前使用的主脚本是：

- [run_sonicstar_vlajepa_automation.sh](/home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh)

当前推荐的启动命令是：

```bash
CKPT_PATH="/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt" \
bash /home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh
```

这条命令对应的是：

- 原始 `SONICSTAR` 权重
- 当前仓库里的 `VLA-JEPA` 评测链路
- 推理模式为原始评测模式：`image + language + state -> action`

## 1. 脚本作用

这个脚本会自动拉起整条评测链路，包括：

1. Step1: Policy Server
2. Step2: Online Inference
3. Step3: MuJoCo Sim Loop
4. Step4: Camera Viewer
5. Step5: Deploy
6. Step6: 自动发送键盘控制指令

脚本本身会完成大部分启动工作，但仍然需要人工完成两个关键操作：

- 在 deploy 窗口输入 `y`
- 在 MuJoCo 窗口按 `9`

## 2. 运行前准备

运行前请确认下面这些路径和环境可用。

### 2.1 代码目录

- SonicStar 根目录：`/home/d013/桌面/SonicStar`
- starVLA 目录：`/home/d013/桌面/SonicStar/starVLA`
- wbc 目录：`/home/d013/桌面/SonicStar/wbc`
- GR00T-WholeBodyControl 目录：`/home/d013/桌面/GR00T-WholeBodyControl`

### 2.2 环境

需要以下 Python/conda 环境可正常使用：

- `starVLA`
- `g1_deploy`
- `/home/d013/桌面/GR00T-WholeBodyControl/.venv_sim`

### 2.3 系统命令

脚本运行前会检查：

- `gnome-terminal`
- `lsof`

如果缺少，脚本会直接退出。

### 2.4 权重

当前推荐权重：

```bash
/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt
```

如果要换权重，最简单的方式是启动时覆盖 `CKPT_PATH`，不要直接改脚本。

## 3. 一条命令启动

在任意终端运行：

```bash
CKPT_PATH="/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt" \
bash /home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh
```

更稳妥一点，也可以先切到项目目录：

```bash
cd /home/d013/桌面/SonicStar
CKPT_PATH="/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt" \
bash ./run_sonicstar_vlajepa_automation.sh
```

## 4. 脚本内部实际启动了什么

### Step1: Policy Server

脚本会在 `starVLA` 环境里执行：

```bash
cd /home/d013/桌面/SonicStar/starVLA
export CKPT_PATH=...
bash examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh
```

作用：

- 加载 `CKPT_PATH` 指定的权重
- 启动 websocket policy server
- 默认监听端口 `10093`

### Step2: Online Inference

脚本会执行：

```bash
cd /home/d013/桌面/SonicStar
PYTHONPATH=$PWD/starVLA:$PWD/wbc python starVLA/examples/SonicLatent/eval_files/run_starvla_inference.py \
  --ckpt-path "$CKPT_PATH" \
  --host 127.0.0.1 \
  --port 10093 \
  --prompt "pick up the cylinder and throw it into the trash bin" \
  --rate 1.0 \
  --log-action-stats
```

作用：

- 连接 Step1 的 policy server
- 接收相机图像和机器人 state
- 做在线推理
- 通过 ZMQ 把动作发给后面的控制链路

### Step3: MuJoCo Sim Loop

脚本会执行：

```bash
/home/d013/桌面/GR00T-WholeBodyControl/.venv_sim/bin/python \
/home/d013/桌面/SonicStar/wbc/gear_sonic/scripts/run_sim_loop.py \
  --robot-scene-override /home/d013/桌面/SonicStar/wbc/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml \
  --enable-image-publish \
  --enable-offscreen \
  --camera-port 5555
```

作用：

- 启动 MuJoCo 仿真
- 发布 `ego_view` 图像
- 给 Step2 提供视觉输入

### Step4: Camera Viewer

脚本会执行：

```bash
/home/d013/桌面/GR00T-WholeBodyControl/.venv_sim/bin/python \
/home/d013/桌面/SonicStar/wbc/gear_sonic/scripts/run_camera_viewer.py \
  --camera-host localhost \
  --camera-port 5555
```

作用：

- 查看当前相机流是否正常

### Step5: Deploy

脚本会在 `g1_deploy` 环境里执行：

```bash
cd /home/d013/桌面/SonicStar/wbc/gear_sonic_deploy
bash deploy.sh --input-type zmq_manager sim
```

作用：

- 启动 deploy 控制链路
- 让后续的动作真的作用到机器人仿真

### Step6: 键盘控制流程

脚本后续会自动按顺序发这些指令：

1. `k`
2. 等你在 MuJoCo 手动按 `9`
3. 自动发送一次 `backspace`
4. `i`
5. `i`
6. `p`

其中：

- `k`：进入控制模式
- `9`：你手动在 MuJoCo 里执行放下机器人
- `backspace`：让 MuJoCo 执行一次 reset
- `i / i`：让机器人进入准备姿态
- `p`：启动 policy

## 5. 你在运行过程中必须手动做的事

虽然叫自动化脚本，但它不是全无人值守。

### 5.1 在 Step5 deploy 窗口输入 `y`

当 deploy 终端出现：

```text
Proceed with deployment? [Y/n]:
```

需要手动输入：

```text
y
```

### 5.2 在 MuJoCo 窗口按 `9`

主终端会提示你：

- 先去 MuJoCo 窗口按 `9`
- 再回到主终端按回车

只有完成这一步，脚本才会继续自动发送 `backspace / i / i / p`。

## 6. 默认时间节奏

脚本中的等待时间是固定写在脚本里的：

- Step1 -> Step2: 20 秒
- Step2 -> Step3: 5 秒
- Step3 -> Step4: 5 秒
- Step4 -> Step5: 5 秒
- Step5 -> Step6: 30 秒

如果机器变慢或环境变了，这些等待时间可能需要调。

对应变量在 [run_sonicstar_vlajepa_automation.sh](/home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh:1) 里：

- `STEP1_TO_STEP2_DELAY`
- `STEP2_TO_STEP3_DELAY`
- `STEP3_TO_STEP4_DELAY`
- `STEP4_TO_STEP5_DELAY`
- `STEP5_TO_STEP6_DELAY`

## 7. 可选运行方式

### 7.1 固定场景

```bash
CKPT_PATH="/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt" \
bash /home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh
```

### 7.2 仅随机杯子位置

```bash
CKPT_PATH="/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt" \
bash /home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh --random-cup
```

### 7.3 仅随机桌面颜色

```bash
CKPT_PATH="/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt" \
bash /home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh --random-table-color
```

### 7.4 随机相机视角

```bash
CKPT_PATH="/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt" \
bash /home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh --random-camera-view
```

### 7.5 随机机器人初始前后位置

脚本支持在 reset 后，让机器人初始位置沿前后方向随机偏移 `2-7 cm`：

```bash
CKPT_PATH="/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt" \
bash /home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh --random-robot-start
```

默认范围由环境变量控制：

```bash
ROBOT_START_MIN_METERS=0.02
ROBOT_START_MAX_METERS=0.07
```

例如改成 `3-5 cm`：

```bash
ROBOT_START_MIN_METERS=0.03 \
ROBOT_START_MAX_METERS=0.05 \
CKPT_PATH="/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt" \
bash /home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh --random-robot-start
```

## 8. 脚本会自动做的预清理

每次启动前，脚本会先清理旧进程和旧端口，避免重复启动冲突。

包括：

- 旧的 policy server
- 旧的 online inference
- 旧的 sim loop
- 旧的 camera viewer
- 旧的 deploy
- 占用 `10093` 的进程
- 占用 `5555` 的进程

如果脚本已经在跑，又手工再起一份，前一份很可能会被清掉。

## 9. 常见问题排查

### 9.1 Step1 起不来

优先检查：

- `CKPT_PATH` 是否存在
- `starVLA` 环境是否能激活
- `10093` 端口是否被占用

### 9.2 Step2 一直打印等待相机

通常说明：

- Step3 还没正常起来
- 相机图像没有成功发布到 `5555`

先看 MuJoCo sim 和 camera viewer 有没有起来。

### 9.3 Step5 卡住

最常见原因是忘了在 deploy 窗口输入 `y`。

### 9.4 机器人没有开始执行

按顺序检查：

1. 是否在 MuJoCo 里按了 `9`
2. 主终端里是否在按 `9` 之后回车确认
3. `k / backspace / i / i / p` 是否都已经发出

### 9.5 如何停止整套流程

回到启动这份脚本的主终端，按：

```bash
Ctrl+C
```

脚本会尝试清理自己拉起的所有子进程。

## 10. 接手人最常用的两条命令

### 固定场景评测

```bash
CKPT_PATH="/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt" \
bash /home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh
```

### 固定场景 + 机器人初始位置随机前后 2-7 cm

```bash
CKPT_PATH="/home/d013/桌面/CKPT/JEPA21/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt" \
bash /home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh --random-robot-start
```

## 11. 备注

当前这份文档对应的是 2026 年 7 月 21 日时仓库里的可运行状态。

如果后续有人重新引入：

- `USE_REAL_VIDEO_LATENTS`
- `HWM_SUBGOAL_MODE`
- `motion_token_scale`

之类的推理侧适配逻辑，那么这份文档里的“默认推理模式”就需要重新核对，不应直接沿用。
