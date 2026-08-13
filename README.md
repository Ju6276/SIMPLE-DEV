# SonicStar 评测端

本仓库是 SonicStar / VLA-JEPA 在 Unitree G1 仿真评测端的代码整理版，主要用于自动拉起 policy server、在线推理、MuJoCo 仿真、camera viewer、deploy 控制链路，并记录 PyTorch / Triton / Draft 加速路径的评测结果。

当前仓库主要分成两部分：

- `starVLA/`：VLA 模型、policy server、在线推理、训练/评测脚本。
- `wbc/`：G1 Whole-Body-Control、MuJoCo 仿真、相机发布、deploy 控制链路。
- `run_sonicstar_vlajepa_automation.sh`：当前推荐的自动化评测入口。
- `run_sonicstar_vla_automation.sh`：原始 SonicLatent 链路的自动化入口，保留作参考。
- `data/result/`：评测结果输出目录，建议不要把大体积结果直接提交到 Git。

相关上游资料：

- GR00T-WholeBodyControl 教程：https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/
- StarVLA 文档：https://starvla.github.io/docs/zh-cn/

## 1. 适用环境

建议使用 Ubuntu Linux + NVIDIA GPU。自动化脚本会打开多个终端窗口，因此默认依赖桌面环境和 `gnome-terminal`。

基础环境：

- Linux，推荐 Ubuntu 20.04 / 22.04。
- NVIDIA GPU，推荐显存 24 GB 以上。
- NVIDIA Driver、CUDA Toolkit。
- Conda / Miniconda。
- Git、Git LFS。
- `gnome-terminal`、`lsof`。
- Python 3.10+，训练相关任务可能需要 Python 3.11。

可选加速环境：

- Triton / TensorRT 路径需要 TensorRT、CUDA Toolkit、Triton manifest。
- Draft / Flash 路径需要额外的 draft checkpoint。

安装常见系统依赖：

```bash
sudo apt update
sudo apt install -y git git-lfs gnome-terminal lsof build-essential cmake
git lfs install
```

## 2. Clone 后目录准备

```bash
git clone <your-repo-url> Sonicstar
cd Sonicstar
git lfs pull
```

如果仓库中没有提交模型权重、预训练模型、数据集或 Triton engine，需要手动准备这些文件。推荐放到统一目录，例如：

```text
Sonicstar/
  CKPTSONICSTAR/
    SONICSTAR/checkpoints/steps_90000_pytorch_model.pt
    Sonicstartriton/sonicstar_full_triton/manifest.json
    Sonicstartriton/sonicstar_draft_triton/manifest.json
    Sonicstardraft/draft_sonic_merged001_m40_exec20_vlmblock_stable.pt
  starVLA/starVLA/playground/Pretrained_models/
    Qwen3-VL-2B-Instruct/
    vjepa2-vitl-fpc64-256/
```

也可以把这些资源放在任意磁盘路径，启动时通过环境变量传入。

## 3. Python 环境配置

这个评测链路实际使用三个环境。

### 3.1 `VLA_JEPA`：policy server 和在线推理

```bash
conda create -n VLA_JEPA python=3.10 -y
conda activate VLA_JEPA
python -m pip install -U pip setuptools wheel
pip install -r starVLA/requirements.txt
pip install -e starVLA
pip install -e "wbc/gear_sonic[teleop,sim,data_collection]"
```

如果只跑评测，一般至少需要 `starVLA/requirements.txt`、`gear_sonic[sim]` 和 `gear_sonic[teleop]`。

### 3.2 `wbc/.venv_sim`：MuJoCo / camera viewer / 发键脚本

自动化脚本默认使用 `wbc/.venv_sim/bin/python` 运行仿真和相机相关脚本。

```bash
python3.10 -m venv wbc/.venv_sim
source wbc/.venv_sim/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e "wbc/gear_sonic[sim,teleop]"
deactivate
```

如果你的仿真环境在其他地方，可以启动时覆盖：

```bash
SIM_PYTHON=/path/to/your/sim/python bash ./run_sonicstar_vlajepa_automation.sh
```

### 3.3 `g1_deploy`：deploy 控制链路

`deploy.sh` 默认在 `g1_deploy` conda 环境里运行。

```bash
conda create -n g1_deploy python=3.10 -y
conda activate g1_deploy
python -m pip install -U pip setuptools wheel
pip install numpy pyyaml pyzmq onnxruntime
```

`wbc/gear_sonic_deploy/` 还依赖本地 C++/ONNX/TensorRT/Unitree SDK 相关文件。首次运行前请确认：

```bash
ls wbc/gear_sonic_deploy/policy/release/model_encoder.onnx
ls wbc/gear_sonic_deploy/policy/release/model_decoder.onnx
ls wbc/gear_sonic_deploy/policy/release/observation_config.yaml
```

如果要使用 Triton / TensorRT，请额外设置：

```bash
export TensorRT_ROOT=/path/to/TensorRT
export TENSORRT_ROOT=/path/to/TensorRT
export CUDA_TOOLKIT_ROOT=/usr/local/cuda
export CUDAToolkit_ROOT=/usr/local/cuda
```

## 4. 运行前必须准备的资源

### 4.1 PyTorch checkpoint

PyTorch baseline 至少需要一个 SonicStar checkpoint：

```bash
export CKPT_PATH=/path/to/steps_90000_pytorch_model.pt
```

脚本默认也会检查 `CKPT_PATH` 是否存在。

### 4.2 预训练模型

如果 checkpoint 配置依赖本地 Qwen / V-JEPA2 目录，请确认这些目录存在，并且 checkpoint 目录中的 `config.yaml` / `config.json` 指向正确路径：

```text
starVLA/starVLA/playground/Pretrained_models/Qwen3-VL-2B-Instruct
starVLA/starVLA/playground/Pretrained_models/vjepa2-vitl-fpc64-256
```

### 4.3 Triton manifest

Triton baseline 需要传入 full Triton manifest：

```bash
export TRITON_MANIFEST=/path/to/sonicstar_full_triton/manifest.json
```

Draft + Triton 需要传入 draft Triton manifest：

```bash
export TRITON_MANIFEST=/path/to/sonicstar_draft_triton/manifest.json
```

### 4.4 Draft checkpoint

Draft / Flash 路径需要：

```bash
export DRAFT_CKPT_PATH=/path/to/draft_sonic_merged001_m40_exec20_vlmblock_stable.pt
```

### 4.5 Warmup image

Triton 自动 warmup 时会检查 `SONICSTAR_POLICY_WARMUP_IMAGE`。如果默认 heatmap 图片没有提交到仓库，请手动指定一张可用的 RGB 图片：

```bash
export SONICSTAR_POLICY_WARMUP_IMAGE=/path/to/warmup_rgb.png
```

## 5. 一键自动化评测

当前推荐入口：

```bash
bash ./run_sonicstar_vlajepa_automation.sh
```

这个脚本会依次启动：

1. Policy Server
2. Online Inference
3. MuJoCo Sim Loop
4. Camera Viewer
5. Deploy
6. 发键控制流程

手动模式下仍需要人工完成两步：

- 在 deploy 终端出现 `Proceed with deployment? [Y/n]:` 时输入 `y`。
- 在 MuJoCo 窗口按 `9`，然后回到主终端按回车。

自动评测模式 `--auto-eval` 会自动确认 deploy、发送 `k / 9 / BackSpace / i / i / p`，并按 episode 重启链路。

## 6. 你之前使用的三组评测命令

下面命令中的路径请替换成自己机器上的真实路径。

### 6.1 Baseline PyTorch

```bash
cd /path/to/Sonicstar

CKPT_PATH=/path/to/CKPTSONICSTAR/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt \
SONICSTAR_REPLAN_STEPS=20 \
MAX_EXEC_STEPS=20 \
bash ./run_sonicstar_vlajepa_automation.sh \
  --backend pytorch \
  --timing-log data/result/baseline_pytorch_Multiple/sonic_timing.jsonl \
  --action-log data/result/baseline_pytorch_Multiple/sonic_actions.jsonl \
  --automation-log-dir data/result/automation_logs/baseline_pytorch_Multiple \
  --auto-eval \
  --num-episodes 10 \
  --max-episode-steps 2000 \
  --eval-result-log data/result/baseline_pytorch_Multiple/episode_results.jsonl \
  --save-video \
  --video-dir data/result/baseline_pytorch_Multiple/videos
```

### 6.2 Baseline Triton

```bash
cd /path/to/Sonicstar

CKPT_PATH=/path/to/CKPTSONICSTAR/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt \
SONICSTAR_POLICY_WARMUP_RUNS=auto \
SONICSTAR_POLICY_WARMUP_IMAGE=/path/to/warmup_rgb.png \
SONICSTAR_REPLAN_STEPS=20 \
MAX_EXEC_STEPS=20 \
bash ./run_sonicstar_vlajepa_automation.sh \
  --backend triton \
  --triton-manifest /path/to/CKPTSONICSTAR/Sonicstartriton/sonicstar_full_triton/manifest.json \
  --timing-log data/result/baseline_triton_Multiple/sonic_timing.jsonl \
  --action-log data/result/baseline_triton_Multiple/sonic_actions.jsonl \
  --automation-log-dir data/result/automation_logs/baseline_triton_Multiple \
  --auto-eval \
  --num-episodes 10 \
  --max-episode-steps 2000 \
  --eval-result-log data/result/baseline_triton_Multiple/episode_results.jsonl \
  --save-video \
  --video-dir data/result/baseline_triton_Multiple/videos
```

### 6.3 Draft + Triton

```bash
cd /path/to/Sonicstar

CKPT_PATH=/path/to/CKPTSONICSTAR/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt \
SONICSTAR_POLICY_WARMUP_RUNS=auto \
SONICSTAR_POLICY_WARMUP_IMAGE=/path/to/warmup_rgb.png \
SONICSTAR_REPLAN_STEPS=20 \
SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS=20 \
SONICSTAR_DRAFT_MIN_ACCEPT_STEPS=10 \
MAX_EXEC_STEPS=20 \
TAU_RADIUS=0.2 \
VERIFY_DIST_DIMS=78 \
T_LIST="0.1 0.05" \
PERIODIC_FULL_EVERY_N_DRAFT_ROUNDS=1 \
bash ./run_sonicstar_vlajepa_automation.sh \
  --backend triton \
  --triton-manifest /path/to/CKPTSONICSTAR/Sonicstartriton/sonicstar_draft_triton/manifest.json \
  --draft-checkpoint /path/to/CKPTSONICSTAR/Sonicstardraft/draft_sonic_merged001_m40_exec20_vlmblock_stable.pt \
  --timing-log data/result/draft_triton_Multiple/sonic_timing.jsonl \
  --action-log data/result/draft_triton_Multiple/sonic_actions.jsonl \
  --automation-log-dir data/result/automation_logs/draft_triton_Multiple \
  --auto-eval \
  --num-episodes 10 \
  --max-episode-steps 1200 \
  --eval-result-log data/result/draft_triton_Multiple/episode_results.jsonl \
  --save-video \
  --video-dir data/result/draft_triton_Multiple/videos
```

## 7. 常用参数

评测核心参数：

- `CKPT_PATH`：PyTorch checkpoint 路径。
- `--backend pytorch|triton`：选择 PyTorch 或 Triton 后端。
- `--triton-manifest PATH`：Triton runtime manifest。
- `--draft-checkpoint PATH`：启用 draft 加速。
- `SONICSTAR_REPLAN_STEPS=20`：每执行 20 个 action step 重新请求 policy。
- `MAX_EXEC_STEPS=20`：每个 chunk 最多执行步数。
- `--auto-eval`：自动跑多 episode。
- `--num-episodes N`：episode 数量。
- `--max-episode-steps N`：单个 episode 最大步数。
- `--timing-log PATH`：推理耗时 JSONL。
- `--action-log PATH`：action chunk JSONL。
- `--eval-result-log PATH`：episode 结果 JSONL。
- `--save-video --video-dir PATH`：保存视频。

泛化场景参数：

- `--random-cup`：随机杯子位置。
- `--random-table-color`：随机桌面颜色。
- `--random-object-colors`：随机杯子和桌面颜色。
- `--enable-lights`：固定顶光。
- `--random-lights`：随机顶光位置/强度。
- `--random-camera-view`：随机相机视角。
- `--random-robot-start`：随机机器人初始前后偏移。

## 8. 手动分步启动

如果自动化脚本失败，可以按链路拆开排查。

### 8.1 启动 policy server

```bash
conda activate VLA_JEPA
cd starVLA

CKPT_PATH=/path/to/steps_90000_pytorch_model.pt \
bash examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh
```

### 8.2 启动在线推理

```bash
conda activate VLA_JEPA
cd /path/to/Sonicstar

PYTHONPATH=$PWD/starVLA:$PWD/wbc \
python starVLA/examples/SonicLatent/eval_files/run_starvla_inference.py \
  --ckpt-path /path/to/steps_90000_pytorch_model.pt \
  --host 127.0.0.1 \
  --port 10093 \
  --prompt "pick up the cylinder and throw it into the trash bin" \
  --rate 1.0 \
  --replan-steps 20 \
  --log-action-stats
```

### 8.3 启动仿真、相机和 deploy

分别在独立终端运行：

```bash
cd wbc
./.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py \
  --enable-image-publish \
  --enable-offscreen \
  --camera-port 5555
```

```bash
cd wbc
./.venv_sim/bin/python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host localhost \
  --camera-port 5555
```

```bash
conda activate g1_deploy
cd wbc/gear_sonic_deploy
bash deploy.sh --input-type zmq_manager sim
```

### 8.4 发送控制键

```bash
cd wbc
./.venv_sim/bin/python gear_sonic/scripts/send_keyboard_cmd.py k
./.venv_sim/bin/python gear_sonic/scripts/send_keyboard_cmd.py i
./.venv_sim/bin/python gear_sonic/scripts/send_keyboard_cmd.py p
```

按键含义：

- `k`：机器人进入 CONTROL 模式。
- `9`：在 MuJoCo 窗口中手动按，放下机器人。
- `i`：准备姿态，通常可发送两次。
- `p`：启动或暂停 VLA policy。

## 9. 数据采集和训练入口

本仓库重点是评测端，但保留了采集和训练入口。

采集链路可参考 `wbc/` 下脚本：

```bash
cd wbc
python gear_sonic/scripts/run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port 5555
python gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5555
bash gear_sonic_deploy/deploy.sh --input-type zmq_manager sim
python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up the cylinder and throw it into the trash bin"
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

训练入口：

```bash
cd starVLA
bash examples/SonicLatent/train_files/run_sonic_latent_train.sh
```

默认训练配置：

```text
starVLA/examples/SonicLatent/train_files/train_sonic_latent.yaml
```

数据集格式可参考 StarVLA LeRobot 数据集教程：

- https://starvla.github.io/docs/zh-cn/training/lerobot-dataset/
- 示例数据集：https://huggingface.co/datasets/Tang-keke/merged_dataset_001

## 10. 上传仓库前建议

这个项目里很容易产生大文件，上传 Git 前建议重点检查：

```bash
git status --short
du -sh data heatmap heatmap_debug starVLA/starVLA/playground wbc/.venv_sim 2>/dev/null
```

通常不建议直接提交：

- `wbc/.venv_sim/`
- `**/__pycache__/`
- `*.pyc`
- `data/result/`
- `heatmap/`
- `heatmap_debug/`
- `*.mp4`
- `*.onnx`，除非是部署必须的小模型并确认许可允许。
- `*.pt`、`*.pth`、`*.safetensors`，建议放 Hugging Face、对象存储或 Git LFS。
- Triton engine / build 产物。

如果确实要提交大模型或 ONNX 文件，建议使用 Git LFS：

```bash
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.safetensors"
git lfs track "*.onnx"
git lfs track "*.mp4"
git add .gitattributes
```

同时建议在 README 或 release note 中说明外部资源下载方式，例如：

- PyTorch checkpoint 下载地址。
- Triton manifest / engine 下载地址。
- Draft checkpoint 下载地址。
- 预训练 Qwen / V-JEPA2 模型下载地址。
- LeRobot 数据集下载地址。

## 11. 常见问题

### 找不到 `gnome-terminal`

自动化脚本需要打开多个终端窗口。请安装：

```bash
sudo apt install -y gnome-terminal
```

如果在服务器无桌面环境上运行，需要改造脚本为 tmux / nohup 模式。

### 找不到 `conda.sh`

脚本会在以下位置查找：

```text
$HOME/anaconda3/etc/profile.d/conda.sh
$HOME/miniconda3/etc/profile.d/conda.sh
/opt/conda/etc/profile.d/conda.sh
```

如果你的 conda 安装在其他路径，请先在 shell 中 source 正确的 `conda.sh`，或修改脚本中的 `find_conda_sh`。

### `Missing required path: CKPT_PATH`

说明 checkpoint 路径不存在。运行前显式指定：

```bash
CKPT_PATH=/your/checkpoint/path.pt bash ./run_sonicstar_vlajepa_automation.sh
```

### Triton warmup 图片不存在

指定一张本地 RGB 图片：

```bash
SONICSTAR_POLICY_WARMUP_IMAGE=/path/to/warmup_rgb.png \
SONICSTAR_POLICY_WARMUP_RUNS=auto \
bash ./run_sonicstar_vlajepa_automation.sh --backend triton --triton-manifest /path/to/manifest.json
```

### Step2 没有 ready

查看自动化日志目录下的 Step2 日志，确认是否出现：

```text
ZMQ action socket bound
```

常见原因是 policy server 未启动成功、checkpoint 配置路径错误、端口 `10093` 被占用。

### MuJoCo 没有图像或相机窗口黑屏

检查：

- `wbc/.venv_sim` 是否安装了 `mujoco`、`opencv-python`、`pyzmq`。
- `run_sim_loop.py` 是否带了 `--enable-image-publish --enable-offscreen`。
- camera port 是否一致，默认 `5555`。

### Deploy 没有进入 `Init Done`

检查：

- `g1_deploy` 环境是否能正常运行。
- `TensorRT_ROOT`、`CUDA_TOOLKIT_ROOT` 是否正确。
- `wbc/gear_sonic_deploy/policy/release/` 下 ONNX 和配置文件是否存在。
- 仿真和 ZMQ manager 是否已经启动。
