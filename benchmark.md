# SIMPLE 端评测流程

本文档说明如何在 `/home/d013/桌面/SIMPLE` 端进行 VLA-JEPA/StarVLA 策略评测。整体流程分三步：

1. 下载评测需要的测试集。
2. 下载对应的渲染/场景/资产文件。
3. 启动 SIMPLE 评测脚本。

评测前需要先在 VLA-JEPA 端启动 policy server。SIMPLE 端只负责创建仿真环境、读取 eval 数据、向 server 请求动作、保存日志和视频。

## 使用到的脚本

SIMPLE 端主要使用这些脚本：

```text
scripts/download_simple_eval_data.sh
scripts/run_eval_vlajepa_all_levels.sh
scripts/run_eval_vlajepa_clean.sh
```

职责如下：

- `scripts/download_simple_eval_data.sh`
  - 从 Hugging Face 下载 `simple-eval/<TASK>.zip`。
  - 解压到 `data/evals/simple-eval/<TASK>/`。
- `scripts/run_eval_vlajepa_all_levels.sh`
  - 推荐的完整 benchmark 入口。
  - 默认依次评测 `level-0 level-1 level-2`。
  - 内部会按 level 调用 `scripts/run_eval_vlajepa_clean.sh`。
- `scripts/run_eval_vlajepa_clean.sh`
  - 单个 level 的评测入口。
  - 通常不直接使用，除非只想测一个 level。

## 第 0 步：启动 VLA-JEPA 推理端

在 `/home/d013/桌面/VLA-JEPA` 启动 policy server，例如：

```bash
cd /home/d013/桌面/VLA-JEPA
conda activate VLA_JEPA

python deployment/model_server/server_policy_simple_g1.py \
  --ckpt_path /home/d013/桌面/CKPT/JEPA21/SIMPLE_OPEN_OVEN_30/steps_10000_pytorch_model.pt \
  --base_vlm_path /home/d013/桌面/VLA-JEPA/Qwen3-VL-2B-Instruct \
  --base_encoder_path /home/d013/桌面/VLA-JEPA/VJEPA21 \
  --port 10090 \
  --cuda 0 \
  --use_bf16
```

看到下面输出表示 server 已经启动：

```text
server listening on 0.0.0.0:10090
```

SIMPLE 端评测时的端口必须和这里一致，例如 `10090`。

## 第 1 步：下载评测测试集

eval 测试集来自：

```text
USC-PSI-Lab/psi-data
```

目录结构应放在：

```text
data/evals/simple-eval/<TASK_NAME>/<LEVEL>/
```

例如 OpenOven：

```text
data/evals/simple-eval/G1WholebodyOpenOvenTeleop-v0/level-0/
data/evals/simple-eval/G1WholebodyOpenOvenTeleop-v0/level-1/
data/evals/simple-eval/G1WholebodyOpenOvenTeleop-v0/level-2/
```

推荐使用仓库脚本下载：

```bash
cd /home/d013/桌面/SIMPLE

bash scripts/download_simple_eval_data.sh G1WholebodyOpenOvenTeleop-v0
```

如果只想下载或更新某几个 level：

```bash
bash scripts/download_simple_eval_data.sh \
  G1WholebodyOpenOvenTeleop-v0 \
  level-0 level-1 level-2
```

等价的手动命令是：

```bash
cd /home/d013/桌面/SIMPLE

export TASK_NAME=G1WholebodyOpenOvenTeleop-v0

hf download USC-PSI-Lab/psi-data \
  simple-eval/${TASK_NAME}.zip \
  --repo-type dataset \
  --local-dir data/evals

unzip -o data/evals/simple-eval/${TASK_NAME}.zip -d data/evals/simple-eval
```

检查目录：

```bash
find data/evals/simple-eval/G1WholebodyOpenOvenTeleop-v0 -maxdepth 2 -type d | sort
```

每个 level 下通常应包含：

```text
data/
meta/
videos/
```

## 第 2 步：下载渲染/场景/资产文件

评测需要的渲染资源主要包括：

- HSSD 场景：`data/scenes/hssd/<SCENE_NAME>/`
- 机器人资源：`data/robots/`
- 物体资源：`data/assets/`
- 材质资源：`data/vMaterials_2/`

这些资源来自：

```text
USC-PSI-Lab/SIMPLE
```

### 2.1 常用基础资源

如果是第一次配置 SIMPLE，先准备基础资源：

```bash
cd /home/d013/桌面/SIMPLE

bash scripts/pre-minimal-download.sh
```

该脚本会下载并解压常用资源，例如：

```text
assets_graspnet.zip
robots_*.zip
vMaterials_2.zip
```

注意：这个脚本默认没有下载所有 HSSD scene，具体 scene 仍可能需要按任务单独补。

### 2.2 下载任务对应 HSSD 场景

任务使用哪个 HSSD scene，可以从任务文件中查看：

```text
src/simple/tasks/<task_file>.py
```

找到类似配置：

```python
room_choices=["hssd:scene1"]
```

再到下面文件中查 `scene1` 对应的真实目录名：

```text
src/simple/resources/hssd-scenes/config.yaml
```

例如 OpenOven：

```text
G1WholebodyOpenOvenTeleop-v0
room_choices=["hssd:scene1"]
hssd:scene1 -> 102344280
```

所以 OpenOven 的 HSSD 场景目录应为：

```text
data/scenes/hssd/102344280/102344280.usd
```

如果本地没有这个目录，手动下载：

```bash
cd /home/d013/桌面/SIMPLE

export SCENE_NAME=102344280

hf download USC-PSI-Lab/SIMPLE \
  scenes_hssd_${SCENE_NAME}.zip \
  --repo-type dataset \
  --local-dir data

unzip -o data/scenes_hssd_${SCENE_NAME}.zip -d data
```

检查：

```bash
test -f data/scenes/hssd/${SCENE_NAME}/${SCENE_NAME}.usd && echo "scene ok"
```

### 2.3 自动下载说明

`eval_decoupled_wbc.py` 会在评测前根据 eval episode 里的 `dr_state_dict.scene` 检查 HSSD 场景。如果缺失，它会尝试自动下载。

如果自动下载失败，日志会提示类似：

```text
Missing HSSD scene assets required by this evaluation dataset
Expected local directory data/scenes/hssd/<SCENE_NAME>
or downloadable archive scenes_hssd_<SCENE_NAME>.zip
```

此时按报错里的 `<SCENE_NAME>` 手动下载对应 zip，然后解压到 `data/`。

## 第 3 步：开始评测

推荐使用：

```text
scripts/run_eval_vlajepa_all_levels.sh
```

它会按顺序评测：

```text
level-0
level-1
level-2
```

OpenOven 示例：

```bash
cd /home/d013/桌面/SIMPLE

bash scripts/run_eval_vlajepa_all_levels.sh \
  simple/G1WholebodyOpenOvenTeleop-v0 \
  10090 \
  10
```

参数含义：

```text
参数 1: SIMPLE task id，例如 simple/G1WholebodyOpenOvenTeleop-v0
参数 2: VLA-JEPA server 端口，例如 10090
参数 3: 每个 level 的 episode 数，例如 10
```

等价地，可以通过环境变量控制更多参数：

```bash
cd /home/d013/桌面/SIMPLE

MAX_EPISODE_STEPS=1000 \
SAVE_VIDEO=1 \
LEVELS="level-0 level-1 level-2" \
bash scripts/run_eval_vlajepa_all_levels.sh \
  simple/G1WholebodyOpenOvenTeleop-v0 \
  10090 \
  10
```

如果只想测试 800 steps：

```bash
MAX_EPISODE_STEPS=800 \
bash scripts/run_eval_vlajepa_all_levels.sh \
  simple/G1WholebodyOpenOvenTeleop-v0 \
  10090 \
  10
```

如果只想先测 `level-0`：

```bash
LEVELS="level-0" \
MAX_EPISODE_STEPS=800 \
bash scripts/run_eval_vlajepa_all_levels.sh \
  simple/G1WholebodyOpenOvenTeleop-v0 \
  10090 \
  10
```

## 单 level 评测

如果不想用 all-level 脚本，可以直接使用：

```text
scripts/run_eval_vlajepa_clean.sh
```

示例：

```bash
cd /home/d013/桌面/SIMPLE

TASK=simple/G1WholebodyOpenOvenTeleop-v0 \
NUM_EPISODES=10 \
MAX_EPISODE_STEPS=1000 \
EVAL_DIR=data/evals_decoupled_wbc/level-0 \
bash scripts/run_eval_vlajepa_clean.sh level-0 10090
```

一般建议优先使用 `run_eval_vlajepa_all_levels.sh`，因为它会自动把不同 level 的输出分开写入。

## 输出目录

评测结果写入：

```text
data/evals_decoupled_wbc/<LEVEL>/
```

典型结构：

```text
data/evals_decoupled_wbc/
├── level-0/
│   ├── eval_latest.log
│   ├── eval_stats.txt
│   └── vlajepa_decoupled_wbc/<TASK_NAME>/train/
├── level-1/
│   ├── eval_latest.log
│   ├── eval_stats.txt
│   └── vlajepa_decoupled_wbc/<TASK_NAME>/train/
└── level-2/
    ├── eval_latest.log
    ├── eval_stats.txt
    └── vlajepa_decoupled_wbc/<TASK_NAME>/train/
```

查看成功率：

```bash
cat data/evals_decoupled_wbc/level-0/eval_stats.txt
cat data/evals_decoupled_wbc/level-1/eval_stats.txt
cat data/evals_decoupled_wbc/level-2/eval_stats.txt
```

查看最新日志：

```bash
less data/evals_decoupled_wbc/level-0/eval_latest.log
```

视频保存在：

```text
data/evals_decoupled_wbc/<LEVEL>/vlajepa_decoupled_wbc/<TASK_NAME>/train/
```

## 常用任务示例

OpenOven：

```bash
bash scripts/download_simple_eval_data.sh G1WholebodyOpenOvenTeleop-v0

bash scripts/run_eval_vlajepa_all_levels.sh \
  simple/G1WholebodyOpenOvenTeleop-v0 \
  10090 \
  10
```

OpenFaucet：

```bash
bash scripts/download_simple_eval_data.sh G1WholebodyOpenFaucetTeleop-v0

bash scripts/run_eval_vlajepa_all_levels.sh \
  simple/G1WholebodyOpenFaucetTeleop-v0 \
  10090 \
  10
```

## 注意事项

- VLA-JEPA server 必须先启动，且端口要和 SIMPLE 端一致。
- SIMPLE 端不要在 VLA-JEPA 的 conda 环境里运行；评测脚本会用 `uv run` 调用 SIMPLE 的 `.venv`。
- 如果 `data dir not found`，说明测试集没有下载或目录层级不对。
- 如果缺 HSSD scene，按日志里的 `scenes_hssd_<SCENE_NAME>.zip` 下载并解压。
- 如果只想快速检查链路，先用 `LEVELS="level-0"` 和 `NUM_EPISODES=1`。
- `CLEAR_OLD_RESULTS=1` 会清理当前任务旧结果；想保留旧结果时设置 `CLEAR_OLD_RESULTS=0`。
- `SAVE_VIDEO=0` 可以减少视频写入开销。
- `MAX_EPISODE_STEPS` 会覆盖任务源码里的默认步数。
