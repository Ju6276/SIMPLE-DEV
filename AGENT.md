# SIMPLE 新权重评测操作说明

本文档用于说明：当需要使用一个全新的 VLA-JEPA/StarVLA 权重评测 SIMPLE 任务时，应该改哪些地方、不应该动哪些地方，如何下载测试集和场景资源，以及推荐评测命令。

## 基本原则

换新权重评测时，通常不需要修改 SIMPLE 源码。推荐做法是：

1. 在 VLA-JEPA 仓库启动新的 policy server，并把 `--ckpt_path` 指向新权重。
2. 在 SIMPLE 仓库下载对应任务的 eval 数据和场景资源。
3. 在 SIMPLE 仓库用 `scripts/run_eval_vlajepa_all_levels.sh` 启动评测客户端。

也就是说，新权重路径主要属于 VLA-JEPA server 的启动参数，不属于 SIMPLE 评测脚本本身。

## 可以改动的内容

优先通过命令行参数和环境变量改动，不直接编辑文件。

### 推荐通过命令行改动

- 任务名：`simple/G1WholebodyOpenFaucetTeleop-v0`
- server 地址：`HOST=127.0.0.1`
- server 端口：`PORT=10090`
- 每个 level 的 episode 数：`NUM_EPISODES=10`
- 最大步数：`MAX_EPISODE_STEPS=800` 或 `1200`
- 是否保存视频：`SAVE_VIDEO=1` 或 `SAVE_VIDEO=0`
- 评测 level：`LEVELS="level-0 level-1 level-2"`
- 输出目录根路径：`EVAL_DIR_ROOT=data/evals_decoupled_wbc`

示例：

```bash
MAX_EPISODE_STEPS=1200 \
bash scripts/run_eval_vlajepa_all_levels.sh \
  simple/G1WholebodyOpenFaucetTeleop-v0 \
  10090 \
  10
```

### 仅在需要修改默认值时改动

- `scripts/run_eval_vlajepa_all_levels.sh`
  - 只建议改默认 `TASK`、`PORT`、`MAX_EPISODE_STEPS`、`NUM_EPISODES`。
  - 不建议改内部调用逻辑。
- `scripts/run_eval_vlajepa_clean.sh`
  - 单 level 评测脚本。
  - 只建议改默认 `TASK`、`LEVEL`、`PORT`、`MAX_EPISODE_STEPS`。

如果只是换权重，不需要改这两个脚本。

### VLA-JEPA 仓库中需要改的启动参数

在 `/home/d013/桌面/VLA-JEPA` 启动 server 时，主要替换：

- `--ckpt_path`
- `--base_vlm_path`
- `--base_encoder_path`
- `--port`
- `--cuda`
- 是否使用 `--use_bf16`

示例：

```bash
cd /home/d013/桌面/VLA-JEPA
conda activate VLA_JEPA

python deployment/model_server/server_policy_simple_g1.py \
  --ckpt_path /path/to/new/steps_xxxxx_pytorch_model.pt \
  --base_vlm_path /home/d013/桌面/VLA-JEPA/Qwen3-VL-2B-Instruct \
  --base_encoder_path /home/d013/桌面/VLA-JEPA/vjepa2-vitl-fpc64-256 \
  --port 10090 \
  --cuda 0 \
  --use_bf16
```

## 不应该改动的内容

除非明确是在修 bug 或开发新功能，否则不要改这些文件或目录：

- `src/simple/tasks/**`
  - 任务定义、DR 配置、success criteria、场景选择都在这里。
  - 评测新权重时不应该改任务逻辑。
- `src/simple/cli/eval_decoupled_wbc.py`
  - decoupled WBC eval 主流程。
  - 改这里会影响所有 teleop 评测。
- `src/simple/baselines/vlajepa_decoupled_wbc.py`
  - VLA-JEPA SIMPLE client 适配层。
  - 除非 server 协议或动作维度变了，否则不要动。
- `src/simple/agents/**`
  - policy agent、WBC agent、remote client 等核心逻辑。
- `pyproject.toml`、`uv.lock`
  - 不要为了评测单个权重改依赖。
- `third_party/**`
  - 外部依赖，不作为普通评测配置入口。
- `data/evals/simple-eval/**/data`、`meta`、`videos`
  - eval 数据集内容应保持原样。
- `data/assets/**`、`data/scenes/**`
  - 只下载/解压缺失资源，不手工编辑资源文件。
- `data/evals_decoupled_wbc/**`
  - 这是输出目录，可以清理旧结果，但不要把它当输入数据改。

## 下载测试数据集

eval 数据来自 Hugging Face dataset：

```text
USC-PSI-Lab/psi-data
```

目录规则：

```text
data/evals/simple-eval/<TASK_NAME>/<LEVEL>/
```

其中：

- `<TASK_NAME>` 不带 `simple/` 前缀，例如 `G1WholebodyOpenFaucetTeleop-v0`
- `<LEVEL>` 是 `level-0`、`level-1`、`level-2`

下载命令：

```bash
cd /home/d013/桌面/SIMPLE

export TASK=G1WholebodyOpenFaucetTeleop-v0

mkdir -p data/evals/simple-eval

hf download USC-PSI-Lab/psi-data \
  simple-eval/${TASK}.zip \
  --repo-type dataset \
  --local-dir data/evals

unzip -o data/evals/simple-eval/${TASK}.zip -d data/evals/simple-eval
```

下载后目录应类似：

```text
data/evals/simple-eval/G1WholebodyOpenFaucetTeleop-v0/
├── level-0/
│   ├── data/
│   ├── meta/
│   └── videos/
├── level-1/
│   ├── data/
│   ├── meta/
│   └── videos/
└── level-2/
    ├── data/
    ├── meta/
    └── videos/
```

检查命令：

```bash
find data/evals/simple-eval/${TASK} -maxdepth 2 -type d | sort
```

## 下载对应场景资源

场景资源来自 Hugging Face dataset：

```text
USC-PSI-Lab/SIMPLE
```

HSSD 场景目录规则：

```text
data/scenes/hssd/<SCENE_NAME>/<SCENE_NAME>.usd
```

场景 zip 命名规则：

```text
scenes_hssd_<SCENE_NAME>.zip
```

### 如何确定任务对应的场景

1. 打开任务文件，例如：

```text
src/simple/tasks/g1_wholebody_open_faucet_teleop.py
```

2. 找到 `TabletopSceneDRCfg` 里的 `room_choices`：

```python
room_choices=["hssd:scene2"]
```

3. 在 HSSD 配置中查 `scene2` 对应的真实目录名：

```text
src/simple/resources/hssd-scenes/config.yaml
```

OpenFaucet 的对应关系是：

```text
hssd:scene2 -> 107733960_175999701
```

### OpenFaucet 场景下载示例

```bash
cd /home/d013/桌面/SIMPLE

export SCENE=107733960_175999701

hf download USC-PSI-Lab/SIMPLE \
  scenes_hssd_${SCENE}.zip \
  --repo-type dataset \
  --local-dir data

unzip -o data/scenes_hssd_${SCENE}.zip -d data
```

检查命令：

```bash
test -f data/scenes/hssd/${SCENE}/${SCENE}.usd && echo "scene ok"
```

### 资源自动下载说明

`eval_decoupled_wbc.py` 会在评测前根据 episode 里的 `dr_state_dict.scene` 做 preflight，并尝试自动下载缺失 HSSD 场景。

如果机器无法访问 Hugging Face，日志可能会提示类似：

```text
Missing HSSD scene assets required by this evaluation dataset ...
Expected local directory data/scenes/hssd/<SCENE_NAME>
or downloadable archive scenes_hssd_<SCENE_NAME>.zip
```

此时按报错里的 zip 名手动下载并解压到 `data/`。

## 评测指令

### 1. 启动 VLA-JEPA server

在 VLA-JEPA 仓库中启动，保持该终端不要关闭：

```bash
cd /home/d013/桌面/VLA-JEPA
conda activate VLA_JEPA

python deployment/model_server/server_policy_simple_g1.py \
  --ckpt_path /home/d013/桌面/CKPT/JEPA2/SIMPLE_OPENFAUCET__30/steps_40000_pytorch_model.pt \
  --base_vlm_path /home/d013/桌面/VLA-JEPA/Qwen3-VL-2B-Instruct \
  --base_encoder_path /home/d013/桌面/VLA-JEPA/vjepa2-vitl-fpc64-256 \
  --port 10090 \
  --cuda 0 \
  --use_bf16
```

看到类似输出表示 server 已启动：

```text
server listening on 0.0.0.0:10090
```

### 2. 跑全部 level

在 SIMPLE 仓库另开一个终端：

```bash
cd /home/d013/桌面/SIMPLE

MAX_EPISODE_STEPS=1200 \
bash scripts/run_eval_vlajepa_all_levels.sh \
  simple/G1WholebodyOpenFaucetTeleop-v0 \
  10090 \
  10
```

参数含义：

- 第一个参数：任务 ID
- 第二个参数：VLA-JEPA server 端口
- 第三个参数：每个 level 的 episode 数
- `MAX_EPISODE_STEPS`：每个 episode 最多执行步数

### 3. 只跑单个 level

```bash
cd /home/d013/桌面/SIMPLE

TASK=simple/G1WholebodyOpenFaucetTeleop-v0 \
NUM_EPISODES=10 \
MAX_EPISODE_STEPS=1200 \
EVAL_DIR=data/evals_decoupled_wbc/level-0 \
bash scripts/run_eval_vlajepa_clean.sh level-0 10090
```

## 输出目录

all-level 脚本会把不同 level 分开写入：

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

查看结果：

```bash
cat data/evals_decoupled_wbc/level-0/eval_stats.txt
cat data/evals_decoupled_wbc/level-1/eval_stats.txt
cat data/evals_decoupled_wbc/level-2/eval_stats.txt
```

查看日志：

```bash
less data/evals_decoupled_wbc/level-0/eval_latest.log
```

视频文件位于：

```text
data/evals_decoupled_wbc/<LEVEL>/vlajepa_decoupled_wbc/<TASK_NAME>/train/
```

## 注意事项

- VLA-JEPA server 和 SIMPLE client 的端口必须一致，例如都用 `10090`。
- server 终端必须一直保持运行。
- SIMPLE 侧使用 `uv run` 和仓库 `.venv`，不要手动切到 VLA-JEPA 的 conda 环境里跑 SIMPLE eval。
- `scripts/run_eval_vlajepa_clean.sh` 会清理 `CONDA_PREFIX`、`PYTHONPATH`、`LD_LIBRARY_PATH` 等变量，避免 conda 环境污染 Isaac Sim。
- 如果只想快速 smoke test，可以先设 `NUM_EPISODES=1` 或只跑 `level-0`。
- 如果完整评测，建议每个 level 跑 10 个 episode。
- `CLEAR_OLD_RESULTS=1` 会清理当前任务旧结果；如果想保留旧结果，设置 `CLEAR_OLD_RESULTS=0`。
- Isaac Sim 首次启动慢是正常现象。
- 如果日志显示 `data dir not found`，说明 eval 数据集没有下载或目录层级不对。
- 如果日志显示缺 `scenes_hssd_<SCENE_NAME>.zip`，下载对应场景 zip 并解压到 `data/`。
- 如果出现 Hugging Face 权限或网络问题，确认已登录：

```bash
hf auth login
```

## OpenFaucet 快速清单

目标任务：

```text
simple/G1WholebodyOpenFaucetTeleop-v0
```

训练数据来源示例：

```text
/home/d013/桌面/SIMPLE/data/simple/G1WholebodyOpenFaucetTeleop-v0
```

eval 数据目录：

```text
data/evals/simple-eval/G1WholebodyOpenFaucetTeleop-v0/level-0
data/evals/simple-eval/G1WholebodyOpenFaucetTeleop-v0/level-1
data/evals/simple-eval/G1WholebodyOpenFaucetTeleop-v0/level-2
```

HSSD 场景：

```text
hssd:scene2 -> data/scenes/hssd/107733960_175999701/107733960_175999701.usd
```

推荐评测命令：

```bash
cd /home/d013/桌面/SIMPLE

MAX_EPISODE_STEPS=1200 \
bash scripts/run_eval_vlajepa_all_levels.sh \
  simple/G1WholebodyOpenFaucetTeleop-v0 \
  10090 \
  10
```
