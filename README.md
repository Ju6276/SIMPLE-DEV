# Pi05-Simple-Client

本仓库是 SIMPLE 的 pi0.5 客户端评测环境，专门用于配合 `Flash-Pi05-Simple-Server` 跑 SIMPLE 数据集上的 Unitree G1 humanoid 任务。

当前推荐任务：

```text
simple/G1WholebodyHandoverTeleop-v0
```

当前推荐策略：

```text
pi05_decoupled_wbc
```

这个仓库负责仿真环境、G1 decoupled WBC、episode 评测、动作执行、日志和视频保存；模型推理 server 由另一个仓库负责：

```bash
/cpfs_infra/shared/fangbaozhong/Flash-Pi05-Simple-Server
```

## 两个仓库如何配合

典型实验需要两个终端：

```text
Terminal 1: Flash-Pi05-Simple-Server
  启动 pi0.5 policy server
  - PyTorch baseline
  - Triton full baseline
  - Triton + draft route

Terminal 2: Pi05-Simple-Client
  启动 SIMPLE eval
  - 连接 127.0.0.1:22085
  - 发送 observation/image、states、prompt
  - 接收 actions、accepted_prefix_len、timing
  - 在 MuJoCo G1 decoupled WBC 中执行动作
```

server 返回的动作通常是 `[30, 36]`。客户端会根据 `SIMPLE_PI05_REPLAN_STEPS` 和 server 返回的 `accepted_prefix_len` 决定本轮实际执行多少步。

## GitHub 分支内容和外部资源

推送到 GitHub 的分支只包含 SIMPLE 客户端代码、配置、脚本和文档，不包含本地运行生成的大文件或外部资源。clone 后需要重新准备这些内容：

- `.venv/`、`.venv-nix/`：Python/Nix 虚拟环境，不推送；clone 后重新运行 `uv sync` 或 `nix develop`。
- `data/`：SIMPLE 数据、eval 数据、视频、jsonl 日志等，不推送；需要按本文档准备到本机路径。
- `third_party/` 下的大型外部库：仓库只应记录 submodule 或源码引用，不应直接提交本地构建产物；clone 后运行 `git submodule update --init --recursive ...`。
- IsaacSim、MuJoCo、CuRobo、Unitree SDK、XRoboToolkit、decoupled_wbc、gear_sonic 等外部依赖不随普通 GitHub 代码分支一起打包；需要按环境配置重新安装。
- `assets/`、`artifacts/`、`cache/`、`videos/`、`wandb/`、`output/`、`.runtime-state/`：本地缓存、日志或结果，不推送。
- pi0.5 模型和 Triton/draft artifact 由 `Flash-Pi05-Simple-Server` 管理；客户端只连接 server，不在本仓库保存模型权重。

如果别人 clone 本仓库，需要先准备 submodule、Python 环境、MuJoCo headless 运行环境、SIMPLE eval 数据，再连接已经启动的 `Flash-Pi05-Simple-Server`。

## 代码位置

pi0.5 相关客户端代码主要在：

```text
src/simple/baselines/pi05.py                  # 基础 pi0.5 websocket agent
src/simple/baselines/pi05_decoupled_wbc.py    # G1 handover 推荐使用的 decoupled WBC agent
src/simple/cli/eval_decoupled_wbc.py          # eval-decoupled-wbc 命令入口
scripts/eval_remote_policy.sh                 # 远程 policy eval 包装脚本
```

`Pi05DecoupledWbcAgent` 会：

- 从 SIMPLE observation 中取 `head_stereo_left` 作为 `observation/image`
- 从 G1 `joint_qpos` 中拼出 pi0.5 需要的状态
- 连接 `openpi_client.websocket_client_policy.WebsocketClientPolicy(host, port)`
- 向 server 发送 `__reset_policy_state__` 和 `__executed_steps__`
- 读取 `accepted_prefix_len`、`policy_timing`、`server_timing`
- 将 36 维 pi0.5 action 转成 decoupled WBC 的 upper-body、waist、base height 和 navigation command
- 写出 timing log 和 executed action log

## 固定实验设置

为了让 PyTorch baseline、Triton full baseline 和 draft route 可比较，建议固定：

- 任务：`G1WholebodyHandoverTeleop-v0`
- split：`train`
- 数据格式：`lerobot`
- 数据目录：`data/evals/simple-eval/G1WholebodyHandoverTeleop-v0/level-0`
- 仿真后端：`mujoco`
- policy：`pi05_decoupled_wbc`
- server 地址：`127.0.0.1:22085`
- episode 数：baseline 可先用 `5`，正式统计建议 `10` 或更多
- draft route：显式设置 `SIMPLE_PI05_REPLAN_STEPS=20`

## 环境准备

进入客户端仓库：

```bash
cd /cpfs_infra/shared/fangbaozhong/Pi05-Simple-Client
```

初始化 submodule。HandoverTeleop 使用 decoupled WBC，下面这些 submodule 建议统一切到 https URL，避免无 SSH key 时拉取失败：

```bash
git config submodule.third_party/decoupled_wbc.url https://github.com/songlin/decoupled_wbc.git
git config submodule.third_party/gear_sonic.url https://github.com/songlin/gear_sonic.git
git config submodule.third_party/unitree_sdk2_python.url https://github.com/songlin/unitree_sdk2_python.git
git config submodule.third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64.url https://github.com/songlin/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64.git
git config submodule.third_party/openpi-client.url https://github.com/songlin/openpi-client.git

git submodule update --init --recursive third_party/decoupled_wbc
git submodule update --init --recursive third_party/gear_sonic
git submodule update --init --recursive third_party/unitree_sdk2_python
git submodule update --init --recursive third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64
git submodule update --init --recursive third_party/openpi-client
```

安装 Python 环境。优先使用已有 `.venv`；如果需要重建：

```bash
GIT_LFS_SKIP_SMUDGE=1 UV_HTTP_TIMEOUT=3000 \
uv sync --all-groups --index-strategy unsafe-best-match
```

如果需要 CuRobo：

```bash
bash scripts/install_curobo.sh
```

验证 SIMPLE 能导入：

```bash
./.venv/bin/python -c "import simple; print(simple.__version__)"
```

## MuJoCo-only 运行方式

pi0.5 handover 评测建议先用 MuJoCo-only 跑通，绕开 IsaacSim 变量。

每次 eval 前设置：

```bash
cd /cpfs_infra/shared/fangbaozhong/Pi05-Simple-Client

export MUJOCO_GL=egl
export PYTHONFAULTHANDLER=1
export TASK=G1WholebodyHandoverTeleop-v0
export SERVER_HOST=127.0.0.1
export SERVER_PORT=22085
export DATA_DIR=data/evals/simple-eval/$TASK/level-0
```

确认数据目录存在：

```bash
ls "$DATA_DIR"
```

如果数据目录不存在，先确认 SIMPLE 数据和 eval 数据是否已经准备到 `data/evals/simple-eval/` 下。

## 跑 PyTorch baseline eval

先在 `Flash-Pi05-Simple-Server` 终端启动 PyTorch baseline server：

```bash
cd /cpfs_infra/shared/fangbaozhong/Flash-Pi05-Simple-Server

export MUJOCO_GL=egl
export PYTHONFAULTHANDLER=1

uv run --no-sync scripts/serve_policy.py \
  --port 22085 \
  --simple-dataset-root /cpfs_infra/shared/fangbaozhong/simple-data/simple/G1WholebodyHandoverTeleop-v0 \
  policy:checkpoint \
  --policy.config pi05_simple_g1_handover_teleop \
  --policy.dir /cpfs_infra/shared/fangbaozhong/psi-model/openpi-05/G1WholebodyHandoverTeleop-v0/40000
```

然后在本仓库运行 SIMPLE eval：

```bash
cd /cpfs_infra/shared/fangbaozhong/Pi05-Simple-Client

export MUJOCO_GL=egl
export PYTHONFAULTHANDLER=1
export TASK=G1WholebodyHandoverTeleop-v0

SIMPLE_PI05_TIMING_LOG=data/evals_decoupled_wbc/pi05_baseline_pytorch_timing.jsonl \
SIMPLE_PI05_ACTION_LOG=data/evals_decoupled_wbc/pi05_baseline_pytorch_actions.jsonl \
./.venv/bin/eval-decoupled-wbc \
  simple/$TASK \
  pi05_decoupled_wbc \
  train \
  --data-format lerobot \
  --data-dir data/evals/simple-eval/$TASK/level-0 \
  --host 127.0.0.1 \
  --port 22085 \
  --sim-mode mujoco \
  --headless \
  --num-episodes 5 \
  --save-video
```

这一组结果作为普通 pi0.5 baseline，主要记录成功率、episode 时间和请求耗时。

## 跑 Triton full baseline eval

先在 server 仓库启动 Triton full route：

```bash
cd /cpfs_infra/shared/fangbaozhong/Flash-Pi05-Simple-Server

export MUJOCO_GL=egl
export PYTHONFAULTHANDLER=1

uv run --no-sync scripts/spec/spec_serve_policy.py \
  --port 22085 \
  --config pi05_simple_g1_handover_teleop \
  --backend triton \
  --base-only \
  --simple-dataset-root /cpfs_infra/shared/fangbaozhong/simple-data/simple/G1WholebodyHandoverTeleop-v0 \
  --base-triton-path data/triton/pi05_simple_g1_handover_teleop_base/base_weights.pkl \
  --num-views 2
```

客户端 eval 命令保持一致，只换 log 文件名：

```bash
cd /cpfs_infra/shared/fangbaozhong/Pi05-Simple-Client

export MUJOCO_GL=egl
export PYTHONFAULTHANDLER=1
export TASK=G1WholebodyHandoverTeleop-v0

SIMPLE_PI05_TIMING_LOG=data/evals_decoupled_wbc/pi05_baseline_triton_timing.jsonl \
SIMPLE_PI05_ACTION_LOG=data/evals_decoupled_wbc/pi05_baseline_triton_actions.jsonl \
./.venv/bin/eval-decoupled-wbc \
  simple/$TASK \
  pi05_decoupled_wbc \
  train \
  --data-format lerobot \
  --data-dir data/evals/simple-eval/$TASK/level-0 \
  --host 127.0.0.1 \
  --port 22085 \
  --sim-mode mujoco \
  --headless \
  --num-episodes 5 \
  --save-video
```

## 跑 Triton + draft eval

先在 server 仓库启动 Triton + draft route：

```bash
cd /cpfs_infra/shared/fangbaozhong/Flash-Pi05-Simple-Server

export MUJOCO_GL=egl
export PYTHONFAULTHANDLER=1

SPEC_TRITON_ONLINE_PROMPT_EMBED=1 \
SPEC_TRITON_SIMPLE_STABILITY_GUARD=1 \
SPEC_DEBUG_DRAFT_ALIGN=1 \
uv run --no-sync scripts/spec/spec_serve_policy.py \
  --port 22085 \
  --config pi05_simple_g1_handover_teleop \
  --simple-dataset-root /cpfs_infra/shared/fangbaozhong/simple-data/simple/G1WholebodyHandoverTeleop-v0 \
  --base-triton-path data/triton/pi05_simple_g1_handover_teleop_base \
  --draft-triton-path data/triton/pi05_simple_g1_handover_teleop_flash/draft_triton.pkl \
  --backend triton \
  --tau-radius 0.25 \
  --t-list 0.1 0.05 \
  --dist-dims 28 \
  --max-exec-steps 12 \
  --periodic-full-every-n-draft-rounds 2 \
  --no-enable-gripper-verify \
  --no-enable-gripper-post-verify
```

然后在本仓库运行 draft eval。draft 实验建议显式设置 `SIMPLE_PI05_REPLAN_STEPS=20`，server 返回的 `accepted_prefix_len` 仍然会限制实际执行长度。

```bash
cd /cpfs_infra/shared/fangbaozhong/Pi05-Simple-Client

export MUJOCO_GL=egl
export PYTHONFAULTHANDLER=1
export TASK=G1WholebodyHandoverTeleop-v0

SIMPLE_PI05_REPLAN_STEPS=20 \
SIMPLE_PI05_USE_ACCEPTED_PREFIX=1 \
SIMPLE_PI05_TIMING_LOG=data/evals_decoupled_wbc/pi05_flash_draft_handover_timing.jsonl \
SIMPLE_PI05_ACTION_LOG=data/evals_decoupled_wbc/pi05_flash_draft_handover_actions.jsonl \
./.venv/bin/eval-decoupled-wbc \
  simple/$TASK \
  pi05_decoupled_wbc \
  train \
  --data-format lerobot \
  --data-dir data/evals/simple-eval/$TASK/level-0 \
  --host 127.0.0.1 \
  --port 22085 \
  --sim-mode mujoco \
  --headless \
  --num-episodes 10 \
  --save-video
```

调试时可以先把 `--num-episodes 10` 改成 `2`。

## 客户端环境变量

pi0.5 客户端常用环境变量：

```bash
SIMPLE_PI05_REPLAN_STEPS=20
SIMPLE_PI05_USE_ACCEPTED_PREFIX=1
SIMPLE_PI05_FULL_EXEC_STEPS=0
SIMPLE_PI05_TIMING_LOG=data/evals_decoupled_wbc/pi05_timing.jsonl
SIMPLE_PI05_ACTION_LOG=data/evals_decoupled_wbc/pi05_executed_actions.jsonl
```

含义：

- `SIMPLE_PI05_REPLAN_STEPS`：客户端每轮最多执行多少个动作。
- `SIMPLE_PI05_USE_ACCEPTED_PREFIX`：是否使用 server 返回的 `accepted_prefix_len`。默认 `1`，调试时可设为 `0`。
- `SIMPLE_PI05_FULL_EXEC_STEPS`：full route 的执行长度覆盖值。默认 `0` 表示仍按 replan/accepted prefix 逻辑。
- `SIMPLE_PI05_TIMING_LOG`：每次 server 请求的耗时 jsonl。
- `SIMPLE_PI05_ACTION_LOG`：每个实际执行动作的 jsonl。

通用仿真变量：

```bash
MUJOCO_GL=egl
PYTHONFAULTHANDLER=1
```

## 日志文件

`SIMPLE_PI05_TIMING_LOG` 每行是一条 server 请求记录，常用字段包括：

- `episode_idx`
- `global_step_idx`
- `server_query_idx`
- `client_roundtrip_ms`
- `policy_time_ms`
- `accepted_prefix_len`
- `exec_len`
- `replan_steps`
- `policy_timing.route_type`
- `policy_timing.total_ms`
- `server_timing`

`SIMPLE_PI05_ACTION_LOG` 每行是一条实际执行动作记录，常用字段包括：

- `episode_idx`
- `global_step_idx`
- `server_query_idx`
- `chunk_action_idx`
- `route_type`
- `accepted_prefix_len`
- `exec_len`
- `upper_0_28`
- `waist_28_31`
- `base_height_31`
- `nav_32_36`

## 快速统计 timing log

把下面的 `LOG` 改成你的 timing jsonl，然后直接运行整段命令：

```bash
LOG=data/evals_decoupled_wbc/pi05_flash_draft_handover_timing.jsonl \
./.venv/bin/python - <<'PY'
import json
import os
from statistics import mean

path = os.environ["LOG"]
records = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

def number(x):
    return x if isinstance(x, (int, float)) else None

def route(rec):
    pt = rec.get("policy_timing") or {}
    rt = pt.get("route_type")
    if rt:
        return rt
    if number(pt.get("used_full_fallback")) and pt["used_full_fallback"] >= 0.5:
        return "full"
    return "draft"

print("queries:", len(records))
for name in ["full", "draft"]:
    subset = [r for r in records if route(r) == name]
    if not subset:
        continue
    total_ms = [
        number((r.get("policy_timing") or {}).get("total_ms"))
        for r in subset
    ]
    total_ms = [x for x in total_ms if x is not None]
    accepted = [number(r.get("accepted_prefix_len")) for r in subset]
    accepted = [x for x in accepted if x is not None]
    exec_len = [number(r.get("exec_len")) for r in subset]
    exec_len = [x for x in exec_len if x is not None]
    print(f"{name}.queries:", len(subset))
    if total_ms:
        print(f"{name}.policy_total_ms_mean:", round(mean(total_ms), 3))
    if accepted:
        print(f"{name}.accepted_mean:", round(mean(accepted), 3))
        print(f"{name}.accepted_gt0_ratio:", round(sum(x > 0 for x in accepted) / len(accepted), 3))
    if exec_len:
        print(f"{name}.exec_steps:", int(sum(exec_len)))
PY
```

也可以直接把脚本保存成临时分析脚本后运行。

## 推荐结果表

建议每次实验记录下面这些指标：

```text
Task: G1WholebodyHandoverTeleop-v0
Episodes: 10
Policy: pi05_decoupled_wbc
Server: pytorch baseline / triton full / triton draft
Client replan steps: 20

query 数:
policy_timing.total_ms mean:
client_roundtrip_ms mean:
draft 平均接受长度:
draft accepted>0 比例:
draft 完整接受 20 steps 比例:
draft 执行动作步数占比:
任务成功率:
平均 episode 时间:
```

你之前的 draft route 记录可以写成：

```text
Full route queries: 198
Draft route queries: 193
Full route policy_timing.total_ms: 136.7 ms
Draft route policy_timing.total_ms: 31.3 ms
Typical speedup: 4.4x
Draft mean accepted prefix: 8.38 / 20 steps
Draft accepted>0 ratio: 71.0%
Draft full accepted 20-step ratio: 18.7%
Draft executed action ratio: 1617 / 5577 = 29.0%
Success rate: 8 / 10 = 80%
```

## 常见问题

### `eval-decoupled-wbc` 找不到

确认 `.venv` 已经创建，并且 console script 存在：

```bash
ls ./.venv/bin/eval-decoupled-wbc
```

如果不存在，重新安装：

```bash
GIT_LFS_SKIP_SMUDGE=1 UV_HTTP_TIMEOUT=3000 \
uv sync --all-groups --index-strategy unsafe-best-match
```

### submodule 拉取失败

先把 SSH URL 改成 https URL，再重新拉：

```bash
git config submodule.third_party/openpi-client.url https://github.com/songlin/openpi-client.git
git submodule update --init --recursive third_party/openpi-client
```

其他 submodule 同理。

### headless 渲染失败

确认：

```bash
export MUJOCO_GL=egl
```

如果仍失败，先用 `--headless` 和 `--sim-mode mujoco` 跑最小 episode。

### 连接不上 server

确认 server 正在监听：

```bash
curl http://127.0.0.1:22085/healthz
```

如果 server 跑在另一台机器，把客户端命令里的：

```bash
--host 127.0.0.1
--port 22085
```

改成对应 IP 和端口。

### draft route 执行步数很少

先看 timing log 中的：

```text
accepted_prefix_len
policy_timing.radius_dist
policy_timing.route_type
```

如果接受长度偏低，可以在 server 侧尝试：

```bash
--tau-radius 0.3
--periodic-full-every-n-draft-rounds 1
```

如果 base/nav 抖动明显，确保 server 侧打开：

```bash
SPEC_TRITON_SIMPLE_STABILITY_GUARD=1
```

## 上游项目

本仓库基于 SIMPLE：

```text
SIMPLE: SIMulation-based Policy Learning and Evaluation
```

上游项目：

- SIMPLE: <https://github.com/physical-superintelligence-lab/SIMPLE>
- OpenPI: <https://github.com/Physical-Intelligence/openpi>
- Realtime-VLA FLASH: <https://dexmal.github.io/realtime-vla-flash/>
