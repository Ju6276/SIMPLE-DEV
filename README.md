# JEPA-Simple-Client

本仓库是 VLA-JEPA + SIMPLE 数据集推理加速实验的仿真评测端，负责在 SIMPLE 环境中启动 `G1WholebodyLocomotionPickBetweenTablesTeleop-v0`，通过 websocket 请求策略服务器，并记录成功率、episode 时长、推理 timing 和动作日志。

配套服务端仓库：

```bash
/cpfs_infra/shared/fangbaozhong/Flash-JEPA2-21-Server
```

本客户端主要用于两条评测链路：

- `vlajepa_decoupled_wbc_source`：连接 VLA-JEPA PyTorch full path baseline。
- `vlajepa_decoupled_wbc`：连接 VLA-JEPA FLASH + Triton speculative server。

## Clone 后需要重新准备

本分支只推送 SIMPLE client 代码、配置和文档，不推送本机环境、数据集、仿真依赖和评测产物。clone 后需要重新配置：

- Python 环境：`.venv/`、`.venv-nix/` 不推送，请用 `uv sync --all-groups` 重建。
- SIMPLE 数据：`data/` 不推送，需要重新下载或软链接 `G1WholebodyLocomotionPickBetweenTablesTeleop-v0` eval 数据。
- 评测产物：`results/`、`output/`、`videos/`、`artifacts/`、`cache/` 不推送。
- 第三方控制/仿真依赖：`third_party/` 内容不推送，需要重新初始化 submodule 或按本机环境安装 IsaacSim、MuJoCo、CuRobo、Unitree SDK、XRoboToolkit、decoupled_wbc、gear_sonic。
- 服务端资源：VLA-JEPA checkpoint、Qwen3-VL、VJEPA2 encoder、Triton manifest 和 draft ckpt 由 `Flash-JEPA2-21-Server` 侧重新准备。

clone 当前分支：

```bash
git clone -b JEPA-Simple-Client https://github.com/Ju6276/SIMPLE-DEV.git JEPA-Simple-Client
cd JEPA-Simple-Client
GIT_LFS_SKIP_SMUDGE=1 UV_HTTP_TIMEOUT=3000 uv sync --all-groups --index-strategy unsafe-best-match
```

## 目录职责

常用代码入口：

- `src/simple/baselines/vlajepa_decoupled_wbc.py`：FLASH/Triton 路径的 SIMPLE policy wrapper。
- `src/simple/baselines/vlajepa_ws_client.py`：VLA-JEPA websocket client 和请求/日志整理。
- `src/simple/cli/eval_decoupled_wbc.py`：decoupled WBC 评测入口。
- `scripts/run_eval_vlajepa_clean.sh`：清理 conda 环境变量后启动评测的脚本模板。
- `scripts/run_eval_vlajepa_all_levels.sh`：批量评测不同 level 的脚本模板。

## 不随仓库分发的内容

clone 后需要重新准备环境、数据和外部依赖。以下内容通常体积较大或与机器环境强绑定，不应提交到 Git：

- Python 虚拟环境：`.venv/`、`.venv-nix/`
- 数据、结果和视频：`data/`、`results/`、`output/`、`videos/`、`artifacts/`、`cache/`
- 第三方仿真/控制依赖：`third_party/`
- 本机 IsaacSim、MuJoCo、CuRobo、Unitree SDK、XRoboToolkit、decoupled_wbc、gear_sonic 安装目录
- VLA-JEPA checkpoint、Qwen3-VL、VJEPA2 encoder、Triton manifest 和权重产物

## 环境配置

建议客户端用 `uv` 的 SIMPLE 环境运行，服务端用自己的 VLA-JEPA 环境运行，避免 conda 动态库污染 MuJoCo/Isaac。

```bash
cd /cpfs_infra/shared/fangbaozhong/JEPA-Simple-Client

GIT_LFS_SKIP_SMUDGE=1 \
UV_HTTP_TIMEOUT=3000 \
uv sync --all-groups --index-strategy unsafe-best-match
```

初始化 decoupled WBC 相关 submodule：

```bash
cd /cpfs_infra/shared/fangbaozhong/JEPA-Simple-Client

git submodule update --init --recursive third_party/decoupled_wbc
git submodule update --init --recursive third_party/gear_sonic
git submodule update --init --recursive third_party/unitree_sdk2_python
git submodule update --init --recursive third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64
```

如果 submodule 默认地址不可访问，可以先改成可用镜像或 fork：

```bash
git config submodule.third_party/decoupled_wbc.url https://github.com/songlin/decoupled_wbc.git
git config submodule.third_party/gear_sonic.url https://github.com/songlin/gear_sonic.git
git config submodule.third_party/unitree_sdk2_python.url https://github.com/songlin/unitree_sdk2_python.git
git config submodule.third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64.url https://github.com/songlin/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64.git
```

基础检查：

```bash
cd /cpfs_infra/shared/fangbaozhong/JEPA-Simple-Client
uv run --no-sync python -c "import simple; print(simple.__version__)"
```

## 固定实验设置

为了让 baseline、Full Triton 和 FLASH + Triton 可比较，建议固定：

- 任务：`G1WholebodyLocomotionPickBetweenTablesTeleop-v0`
- 数据集：`simple-data/simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0`
- split：`train`
- data format：`lerobot`
- eval level：`level-2`
- sim mode：`mujoco_isaac`
- episode 数：`10`
- max episode steps：`800`
- replan steps：FLASH 路径使用 `SIMPLE_JEPA_REPLAN_STEPS=20`
- 服务端 checkpoint、Triton manifest、draft ckpt 保持一致

## 评测前准备

建议每次评测前设置：

```bash
export MUJOCO_GL=egl
export PYTHONFAULTHANDLER=1
```

如果当前 shell 曾经 `conda activate VLA_JEPA`，客户端评测建议用 `env -u` 清掉 conda 相关变量：

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.7.6 \
MUJOCO_GL=egl env \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u PYTHONPATH \
  -u PYTHONHOME \
  -u LD_PRELOAD \
  -u LD_LIBRARY_PATH \
  uv run --no-sync eval-decoupled-wbc ...
```

## 1. VLA-JEPA PyTorch Full Path Baseline

先在服务端启动 PyTorch full path：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate VLA_JEPA

cd /cpfs_infra/shared/fangbaozhong/Flash-JEPA2-21-Server/VLA-JEPA-DEV

python -c "from accelerate import PartialState; PartialState(); import runpy; runpy.run_module('deployment.model_server.server_policy_simple_g1', run_name='__main__')" \
  --ckpt_path checkpoints/g1_pick_between_tables_ft/checkpoints/steps_40000_pytorch_model.pt \
  --base_vlm_path Qwen3-VL-2B-Instruct \
  --base_encoder_path vjepa2-vitl-fpc64-256 \
  --port 10093 \
  --cuda 0
```

再在客户端评测：

```bash
cd /cpfs_infra/shared/fangbaozhong/JEPA-Simple-Client

SIMPLE_JEPA_SOURCE_TIMING_LOG=data/result/baseline/vlajepa_source_timing.jsonl \
SETUPTOOLS_SCM_PRETEND_VERSION=0.7.6 \
MUJOCO_GL=egl env \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u PYTHONPATH \
  -u PYTHONHOME \
  -u LD_PRELOAD \
  -u LD_LIBRARY_PATH \
  uv run --no-sync eval-decoupled-wbc \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  vlajepa_decoupled_wbc_source \
  train \
  --data-format lerobot \
  --data-dir /cpfs_infra/shared/fangbaozhong/JEPA-Simple-Client/data/evals/simple-eval/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/level-2 \
  --host 127.0.0.1 \
  --port 10093 \
  --headless \
  --sim-mode mujoco_isaac \
  --eval-dir data/evals_decoupled_wbc \
  --max-episode-steps 800 \
  --episode-start 0 \
  --num-workers 1 \
  --num-episodes 10 \
  --save-video
```

## 2. VLA-JEPA FLASH + Triton

先在服务端启动 FLASH + Triton server：

```bash
cd /cpfs_infra/shared/fangbaozhong/Flash-JEPA2-21-Server

uv run --no-sync scripts/spec/jepa_serve_policy.py \
  --backend triton \
  --host 0.0.0.0 \
  --port 22085 \
  --checkpoint VLA-JEPA-DEV/checkpoints/g1_pick_between_tables_ft/checkpoints/steps_40000_pytorch_model.pt \
  --vlajepa-repo VLA-JEPA-DEV \
  --base-vlm-path VLA-JEPA-DEV/Qwen3-VL-2B-Instruct \
  --base-encoder-path VLA-JEPA-DEV/vjepa2-vitl-fpc64-256 \
  --triton-manifest data/triton/jepa_g1_pick_between_tables_qwen3_flash/manifest.json \
  --stats-key g1_pick_between_tables \
  --max-exec-steps 20 \
  --tau-radius 0.25 \
  --verify-dist-dims 28 \
  --t-list 0.1 0.05 \
  --periodic-full-every-n-draft-rounds 1 \
  --enable-shared-prefix-full \
  --device cuda:0
```

再在客户端评测：

```bash
cd /cpfs_infra/shared/fangbaozhong/JEPA-Simple-Client

SIMPLE_JEPA_REPLAN_STEPS=20 \
SIMPLE_JEPA_TIMING_LOG=data/result/draft4/vlajepa_timing.jsonl \
SIMPLE_JEPA_ACTION_LOG=data/result/draft4/vlajepa_actions.jsonl \
SETUPTOOLS_SCM_PRETEND_VERSION=0.7.6 \
MUJOCO_GL=egl env \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u PYTHONPATH \
  -u PYTHONHOME \
  -u LD_PRELOAD \
  -u LD_LIBRARY_PATH \
  uv run --no-sync eval-decoupled-wbc \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  vlajepa_decoupled_wbc \
  train \
  --data-format lerobot \
  --data-dir /cpfs_infra/shared/fangbaozhong/JEPA-Simple-Client/data/evals/simple-eval/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/level-2 \
  --host 127.0.0.1 \
  --port 22085 \
  --headless \
  --sim-mode mujoco_isaac \
  --eval-dir data/result/draft4 \
  --max-episode-steps 800 \
  --episode-start 0 \
  --num-workers 1 \
  --num-episodes 10 \
  --save-video
```

## 常用环境变量

- `SIMPLE_JEPA_REPLAN_STEPS`：每次策略请求对应的重规划步数，FLASH 实验通常设为 `20`。
- `SIMPLE_JEPA_TIMING_LOG`：FLASH/Triton timing JSONL 输出路径。
- `SIMPLE_JEPA_ACTION_LOG`：动作和 accepted prefix JSONL 输出路径。
- `SIMPLE_JEPA_SOURCE_TIMING_LOG`：PyTorch source baseline timing JSONL 输出路径。
- `VLAJEPA_INSTRUCTION_OVERRIDE`：覆盖默认语言指令，排查 prompt 差异时使用。

## 动作维度

本任务动作是 36 维：

```text
0:7    left_hand
7:14   right_hand
14:21  left_arm
21:28  right_arm
28:32  rpy height
32:36  torso_vx, torso_vy, torso_vyaw, target_yaw
```

FLASH verifier 推荐只用前 28 维做距离判断：

```bash
--verify-dist-dims 28
```

这表示只比较双手和双臂相关动作，不表示模型动作维度变成 28。

## 结果整理

建议至少记录：

- success rate
- episode return / episode length
- policy request 数
- 平均 request latency
- 平均每个实际执行动作上的延迟 `/Act`
- flash path rate
- accepted prefix mean
- full refresh / fallback 比例
- 是否出现中途失稳

本轮记录中，FLASH + Triton 的混合稳态延迟约 `25.7 ms`，相较 PyTorch baseline 的 `56.1 ms` 约 `2.18x`；单独看 flash path，约从 Full Triton 的 `31.3 ms` 降到 `20.2 ms`。成功率为 `70%`，比 baseline 低约 10 个百分点，后续应继续调 verifier 阈值、full refresh 频率和 draft 训练设置。

## 常见问题

1. `mujoco_isaac` 启动失败：先确认 IsaacSim、MuJoCo、GPU 驱动和 `MUJOCO_GL=egl`；如果只想排查网络请求，可先切换到 MuJoCo-only 任务或减少 episode。
2. conda 环境污染：客户端使用 `env -u CONDA_PREFIX ...` 清理服务端 conda 变量。
3. server 连接失败：确认 `--host`、`--port` 和客户端端口一致；本机评测用 `127.0.0.1`。
4. 结果不可复现：固定 checkpoint、dataset level、episode-start、num-episodes、max-episode-steps、replan steps 和 server 参数。
