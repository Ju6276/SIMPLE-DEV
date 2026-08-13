# SIMPLE 环境配置与 Psi-0 对接过程记录

这份文档用于记录 `SIMPLE` 项目环境配置、运行准备、与 `Psi-0` 对接过程中已经确认的步骤、常见问题、解决办法，以及仍需补充确认的历史细节。

这不是一份“凭记忆回放”的流水账，而是一份分层记录：

- `仓库已确认`：可以直接从当前仓库代码、脚本、README、文档里验证的内容
- `根据仓库可推断`：仓库没有逐字写明，但从脚本和调用关系可以合理推断的内容
- `待补充确认`：你提到我们之前实际折腾过，但当前仓库里无法完整还原的历史细节

---

## 1. 这份记录的目标

这份记录主要回答四类问题：

1. `SIMPLE` 这个项目到底要怎样把环境跑起来
2. 环境配置过程中有哪些关键分支
3. `Psi-0` 是怎么和 `SIMPLE` 对接起来的
4. 我们一路上容易踩哪些坑，应该怎么排查

### 已确认的记录方式

这份文档后续以“真实踩坑过程回放”为主，不按纯标准安装说明来写。

这意味着后续补充内容会优先记录：

- 我们当时实际选择了哪条安装路径
- 第一处卡点出现在哪里
- 每次报错的现场现象
- 当时采取了什么修复动作
- 哪些修复无效，哪些修复最终有效
- SIMPLE 与 Psi-0 对接时真正的联调过程

标准安装说明只保留为背景材料，不作为文档主体。

---

## 2. 当前已经确认的术语

详细术语见同目录下的 [CONTEXT.md](./CONTEXT.md)。

这里先列出后文会频繁使用的几个词：

- `SIMPLE`：仿真驱动的策略学习与评测项目
- `Task`：一个具体任务实例
- `Teleop Task`：遥操作任务，通常以 `Teleop-v0` 结尾
- `Motion Planning Task`：运动规划任务，通常以 `MP-v0` 结尾
- `Psi-0`：和 SIMPLE 对接的基础模型训练与推理栈
- `Inference Server`：运行在 Psi-0 仓库一侧的模型推理服务
- `Simulation Client`：运行在 SIMPLE 仓库一侧的仿真执行端
- `UV Setup`：基于 `uv` 的本地虚拟环境安装路径
- `Nix Runtime`：基于 Nix 的隔离运行时路径
- `CuRobo`：SIMPLE 当前关键 GPU 运动规划依赖

---

## 3. 仓库里已经确认的主线

### 3.1 项目定位

`README.md` 明确说明：

- SIMPLE 全称是 `SIMulation-based Policy Learning and Evaluation`
- 它建立在 `IsaacSim 4.5` 和 `MuJoCo 3.3` 之上
- 目标是支持类人机器人移动操作任务的仿真、数据生成、训练接入和评测

### 3.2 环境安装有三条主路径

仓库 README 明确给了三条安装路径：

1. `UV Setup`
2. `Nix Setup`
3. `Docker Setup`

其中：

- `UV Setup` 被标成最快路径
- `Nix Setup` 更偏重隔离性和可复现
- `Docker Setup` 是容器化路径

### 3.3 Psi-0 对接不是附带功能，而是主流程之一

仓库中已经明确存在以下内容：

- `scripts/postprocess_psi0.py`
- `scripts/postprocess_psi0_sonic.py`
- `src/simple/baselines/psi0.py`
- `src/simple/baselines/psi0_decoupled_wbc.py`
- README 中专门有 `Fine-Tuning` 与 `Evaluation in SIMPLE` 章节

这说明 `SIMPLE -> Psi-0 compatible dataset -> Psi-0 training / inference -> SIMPLE evaluation client` 是项目正式支持的流程，不是临时实验。

---

## 4. 环境配置主流程

这一节只写当前仓库里已经能确认的标准步骤。

### 4.1 克隆项目与子模块

`仓库已确认`

首先需要：

```bash
git clone git@github.com:physical-superintelligence-lab/SIMPLE.git
cd SIMPLE
git submodule update --init --recursive
```

关键点：

- 这个仓库依赖子模块
- 如果子模块没有拉全，后续很多安装和运行步骤都可能失败

常见问题：

- `third_party` 目录内容不完整
- 某些依赖脚本存在，但对应代码没有初始化下来

解决办法：

- 重新执行 `git submodule update --init --recursive`

### 4.2 选择安装路径

`仓库已确认`

仓库支持三种路线：

1. `UV Setup`
2. `Nix Setup`
3. `Docker Setup`

`根据仓库可推断`

如果宿主机已经有较完整的 NVIDIA 驱动和 CUDA 基础，优先尝试 `UV Setup` 会更直接。

如果更看重隔离、可复现，以及减少宿主机环境污染，`Nix Runtime` 更合适。

---

## 5. UV Setup 路线

### 5.1 安装 `uv`

`仓库已确认`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

作用：

- 提供依赖解析、虚拟环境和包安装入口

### 5.2 同步 Python 依赖

`仓库已确认`

README 和 `docs/source/tutorials/installation.md` 都给出了类似命令：

```bash
UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match
```

`scripts/setup_python_env.sh` 还表明一条更脚本化的路线：

```bash
uv venv
uv pip install --python .venv/bin/python setuptools wheel pybind11
GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}" \
  uv sync --all-groups --index-strategy unsafe-best-match
```

作用：

- 创建 `.venv`
- 安装 Python 依赖
- 避免初始阶段就把所有 Git LFS 大文件拉下来

常见问题：

- 大包下载超时
- 依赖解析速度慢
- 缓存损坏导致下载中断

解决办法：

- 设置更大的 `UV_HTTP_TIMEOUT`
- 必要时清理 `~/.cache/uv`
- 重试 `uv sync`

### 5.3 安装 CuRobo

`仓库已确认`

README 和安装文档都要求执行：

```bash
bash scripts/install_curobo.sh
```

作用：

- 安装 GPU 加速的运动规划和相关能力

关键前提：

- 需要可用的 NVIDIA 驱动
- 需要兼容的 CUDA
- 需要 `git-lfs`

README 还建议设置：

```bash
export TORCH_CUDA_ARCH_LIST=8.9+PTX
```

或按显卡能力设置其他值。

### 5.4 激活虚拟环境并验证

`仓库已确认`

```bash
source .venv/bin/activate
python -c "import simple; print(simple.__version__)"
```

补充验证：

```bash
python scripts/list_env.py
```

作用：

- 验证 `simple` 包能否被正确导入
- 验证项目环境是否至少达到“能被 Python 看见”的状态

---

## 6. Nix Runtime 路线

### 6.1 安装 Nix

`仓库已确认`

```bash
sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon
```

安装完成后需要打开新 shell。

如果 `nix` 命令找不到：

```bash
export PATH=/nix/var/nix/profiles/default/bin:$PATH
```

### 6.2 做前置检查

`仓库已确认`

```bash
./scripts/nix/prereq-check.sh
```

作用：

- 在新主机上先检查 Nix 路线依赖是否满足

### 6.3 进入开发环境

`仓库已确认`

```bash
nix --extra-experimental-features "nix-command flakes" develop
```

或：

```bash
env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop -c <command>
```

关键规则：

- 不要直接 `source .venv/bin/activate`
- 不要把 `.venv` 或 `.venv-nix` 当成独立运行时来用
- 这个仓库预期 `Nix shell + Python environment` 配合使用

### 6.4 Nix 路线的核心约束

`仓库已确认`

README 和 `troubleshooting.md` 反复强调：

- `LD_LIBRARY_PATH`
- `PYTHONPATH`
- `PYTHONHOME`
- `LD_PRELOAD`

这些变量如果从宿主机泄漏进来，会导致运行时污染。

常见问题：

- `libstdc++.so.6` 找不到
- `nix develop` 本身起不来
- Vulkan / NVIDIA 驱动在 shell 里不可见

解决办法：

```bash
unset PYTHONPATH PYTHONHOME LD_PRELOAD
env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop
```

如果是新主机，优先执行：

```bash
./scripts/nix/prereq-check.sh
```

---

## 7. Docker 路线

`仓库已确认`

仓库支持 Docker，但当前 README 对 Docker 的详细说明主要引导到文档页面。

`根据仓库可推断`

Docker 路线适合：

- 想把系统依赖封装进镜像
- 想减少宿主机污染
- 想在 CI 或统一部署环境中运行

但它和 `Nix Runtime` 是两条不同路线，不建议混用。

---

## 8. 资源文件与大文件处理

### 8.1 Git LFS

`仓库已确认`

`scripts/setup_project.sh` 明确做了几件事：

1. 检查并安装 `git-lfs`
2. 执行 `git lfs install`
3. 执行 `git lfs pull`
4. 运行 `uv sync`

这说明项目不只是有 Python 包依赖，还有大量大文件依赖。

### 8.2 预下载资源

`仓库已确认`

可以运行：

```bash
scripts/pre-minimal-download.sh
```

作用：

- 预下载项目运行所需的各种资源和数据

说明：

- 如果下载中断，仓库文档建议直接重新运行脚本继续

---

## 9. 运行与数据主线

### 9.1 数据采集有两类入口

`仓库已确认`

README 明确提到两类数据采集入口：

1. `Teleoperation`
2. `Automated Motion Planning`

这和仓库术语中的 `Teleop Task`、`Motion Planning Task` 是一致的。

### 9.2 运行前的环境变量

`仓库已确认`

README 给出的示例包括：

```bash
export MUJOCO_GL="egl"
export CUDA_VISIBLE_DEVICES="0"
export DISPLAY=":1"
```

作用：

- 指定 MuJoCo 的图形后端
- 指定 GPU
- 指定图形显示环境

常见问题：

- 无头服务器上 GLFW 初始化失败
- 图形渲染或 Isaac Sim 启动报错

解决办法：

- 使用 `MUJOCO_GL=egl`
- 在 headless 场景确保相关参数正确

---

## 10. SIMPLE 到 Psi-0 的数据对接

### 10.1 为什么要做后处理

`仓库已确认`

README 写得非常明确：

- 原始采集数据不能直接喂给 Psi-0
- 需要做后处理，转换成严格兼容 `Psi-0` 训练流水线的数据格式

### 10.2 Motion Planning 数据后处理

`仓库已确认`

脚本：

```bash
python scripts/postprocess_psi0.py \
  --sim-root="data/datagen*/simple/G1WholebodyXMoveBendPickMP-v0/level-0/" \
  --out-dir=data/processed_psi0/G1WholebodyXMoveBendPickMP-v0 \
  --skip=60
```

已经确认的特征：

- 支持通配符 `*`
- 可以合并多个并行批次输出
- 输出目录是 `processed_psi0`

### 10.3 Teleop 数据后处理

`仓库已确认`

脚本：

```bash
python scripts/postprocess_psi0_sonic.py \
  --sim-root="data/replay_decoupled_wbc_output*/simple/G1WholebodyPushOfficeChairTeleop-v0/level-0/" \
  --out-dir=data/processed_psi0/G1WholebodyPushOfficeChairTeleop-v0 \
  --skip=0 \
  --total_episodes=100
```

已经确认的特征：

- 同样支持通配符合并
- 面向遥操作与回放数据
- 支持限制总 episode 数量

### 10.4 从脚本层面确认的语义

`仓库已确认`

从 `scripts/postprocess_psi0.py` 和 `scripts/postprocess_psi0_sonic.py` 可以确认：

- 后处理不仅仅是“拷贝文件”
- 它在重排 state / action 向量
- 它在构建 `Psi-0 compatible dataset` 所需的模态描述
- 它在处理视频、注释和统计信息

也就是说，`SIMPLE -> Psi-0` 之间存在明确的数据格式映射层。

---

## 11. Psi-0 训练与评测对接

### 11.1 训练侧关系

`仓库已确认`

README 明确写了：

- SIMPLE 可以产出可供 Psi-0 使用的数据
- 真正的训练说明以 `Psi-0` 仓库 README 为准
- SIMPLE 这边负责把数据准备到兼容格式

这说明边界大致是：

- `SIMPLE` 负责环境、数据、评测客户端
- `Psi-0` 仓库负责训练环境变量、训练脚本、推理服务部署

### 11.2 评测侧关系

`仓库已确认`

README 明确描述了一个 `Client-Server architecture`：

- `Inference Server` 在 `Psi-0` 仓库里运行
- `Simulation Client` 在 `SIMPLE` 仓库里运行

这不是单仓库端到端闭环，而是跨仓库协作。

### 11.3 Psi-0 服务端准备

`仓库已确认`

README 中给出的服务端步骤包括：

1. 在 `Psi-0` 仓库里配置 `.env`
2. 设置 `HF_TOKEN`、`WANDB` 变量、`PSI_HOME`
3. 下载 `psi0/simple-checkpoints`
4. 运行：

```bash
bash scripts/deploy/serve_psi0_simple.sh $RUN_DIR $CKPT_STEP
```

### 11.4 SIMPLE 客户端评测

`仓库已确认`

针对不同任务类型，README 给出不同入口：

- `Teleop Task`：`eval_decoupled_wbc` + `psi0_decoupled_wbc`
- `Motion Planning Task`：`eval` + `psi0`

说明：

- 任务类型和 agent 入口是强绑定的
- 不能随便混用

---

## 12. 从代码确认的 Psi-0 推理交互方式

`仓库已确认`

`src/simple/baselines/psi0.py` 已经确认：

- SIMPLE 侧通过 `HttpActionClient` 调用远端推理服务
- 会把图像观测和状态向量打包发给服务端
- 会维护 `session_id`、`episode_index`、`step_index`
- 会把返回动作重新映射成 SIMPLE 内部动作命令

这说明：

- `Psi-0` 并不是直接嵌在 SIMPLE 本地进程里
- 当前主路径是“远端推理服务 + 本地仿真客户端”

---

## 13. 已确认的常见问题与解决办法

这一节直接整理当前仓库 `docs/source/troubleshooting.md` 中最关键、最容易踩到的问题。

### 13.1 `uv sync` 下载超时

现象：

- Isaac Sim 相关 wheel 下载失败
- 提示网络超时

解决办法：

```bash
export UV_HTTP_TIMEOUT=300
uv sync
```

### 13.2 Nix 环境里 `libstdc++.so.6` 缺失

现象：

- `uv run eval ...` 或 `python -c 'import torch'` 失败

解决办法：

```bash
direnv reload
```

必要时：

```bash
env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop
```

### 13.3 `nix develop` 启动前就被污染

现象：

- `CXXABI_1.3.15 not found`

解决办法：

```bash
env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop
```

### 13.4 Vulkan / NVIDIA 驱动不可见

现象：

- `vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER`
- `libGLX_nvidia.so.0` 找不到

解决办法：

- 重新加载 dev shell
- 检查宿主机 NVIDIA 驱动库是否存在

### 13.5 CuRobo CUDA 版本不匹配

现象：

- 检测到 CUDA 11.8，但 PyTorch 是按 12.8 编译

解决办法：

- 安装与 PyTorch 兼容的 CUDA 版本
- 当前文档里给出的明确修复示例是 CUDA `12.8`

### 13.6 CuRobo 安装时 `Python.h` 找不到

现象：

- 编译扩展时报缺少 Python 头文件

解决办法：

```bash
sudo apt-get install python3-dev
```

### 13.7 缺少 `libgmpxx.so.4`

解决办法：

```bash
sudo apt install libgmp-dev libgmp10
```

### 13.8 Isaac Sim 缺少 `libGLU.so.1`

解决办法：

```bash
sudo apt-get install libglu1-mesa
```

### 13.9 Headless 场景下 GLFW 未初始化

解决办法：

```bash
export MUJOCO_GL=egl
```

### 13.10 Headless 服务器上 Isaac Sim reset 崩溃

现象：

- `world.reset()` 段错误

解决办法：

- 确保 headless 服务器场景下 `headless=True`

### 13.11 某些依赖解析失败

已确认示例：

- `lerobot` 依赖要求更高版本 `cmake`

解决办法：

```bash
uv sync --group lerobot --index-strategy unsafe-best-match
```

### 13.12 `warp-lang` 版本问题

现象：

- `TypeError: array.__init__() got an unexpected keyword argument 'owner'`

解决办法：

```bash
uv pip install "warp-lang==1.7.0" --index-strategy unsafe-best-match
```

---

## 14. 当前最靠谱的“实际操作顺序”建议

如果今天要从头把 SIMPLE 跑起来，并为后续 Psi-0 对接做准备，一个比较稳的顺序是：

1. 克隆仓库并拉全子模块
2. 确认宿主机是 Linux + NVIDIA 驱动可用
3. 决定走 `UV Setup` 还是 `Nix Runtime`
4. 如果走 `UV Setup`，先装 `uv`，再 `uv sync`
5. 安装 `git-lfs`
6. 执行 `scripts/install_curobo.sh`
7. 运行 `python -c "import simple; print(simple.__version__)"`
8. 运行 `python scripts/list_env.py`
9. 如需预拉资源，运行 `scripts/pre-minimal-download.sh`
10. 根据任务类型做 `Teleop Task` 或 `Motion Planning Task` 的数据采集
11. 运行 `postprocess_psi0.py` 或 `postprocess_psi0_sonic.py`
12. 在 `Psi-0` 仓库侧准备 `.env`、checkpoint 和推理服务
13. 在 SIMPLE 侧作为 `Simulation Client` 发起评测

---

## 15. 当前还缺失的历史细节

这一节专门记录“你明确说我们之前折腾过，但我现在还不能从仓库里百分百还原”的内容。

### 15.1 我已经知道但还没写死的点

- 我们之前花了很长时间配 `SIMPLE` 环境
- 你明确提到还折腾过 `Psi-0`
- 你希望把“途中每个步骤、遇到的问题和解决办法”都详细记下来
- 这份文档应按“真实踩坑过程回放”来写
- 我们一开始主走的是 `UV Setup`

### 15.2 我现在还不能直接当成事实写死的点

下面这些内容，目前仓库内证据不够，需要你补充后我再追加进文档：

- 我们当时是在本机、服务器、WSL 还是远程 GPU 机器上配置
- 我们实际卡住过的第一个问题是什么
- 我们和 `Psi-0` 对接时，主要卡在训练环境、checkpoint 下载、服务端启动，还是客户端联调
- 你提到的“PSI-0 的那一段”里，最想保留下来的关键教训是什么

---

## 18. 真实踩坑过程回放

这一节开始按真实时间线补。

### 阶段 1：决定先走 `UV Setup`

已确认事实：

- 我们一开始没有先走 `Nix Runtime`
- 我们一开始主走的是 `UV Setup`

这意味着我们当时的默认判断很可能是：

- 先用最快路径把仓库跑起来
- 先解决 Python 依赖、虚拟环境和 CuRobo 安装
- 等 `UV Setup` 跑不通或稳定性不够时，再考虑更重的替代路径

这部分属于已确认方向，后续还要继续补：

- 第一次 CuRobo 报错的具体类型
- 我们是否因为 `UV Setup` 的问题才转向了其他方案

### 阶段 2：第一次明显卡在 `CuRobo`

已确认事实：

- 我们先走的是 `UV Setup`
- 第一次真正把进度卡住的问题，不是普通 Python 依赖，而是 `CuRobo`

这很重要，因为它把前期踩坑范围明显缩小到了下面几类：

1. CUDA 版本与 PyTorch 编译版本不匹配
2. CuRobo 编译或安装过程报错
3. 缺少系统级头文件或动态库
4. Isaac Sim / CuRobo 联动时才暴露的问题

仓库内已经存在、并且和这条时间线高度相关的 CuRobo 问题包括：

- CUDA 版本不匹配
- `CUDA error: misaligned address`
- 缺少 `Python.h`
- 缺少 `libgmpxx.so.4`

这说明我们后面补历史时，应该优先围绕“CuRobo 安装链路”展开，而不是先围绕一般 Python 包安装来写。

### 阶段 3：第一次明确报错是 CUDA 版本不匹配

已确认事实：

- 我们第一次把环境卡住的 `CuRobo` 问题，是 `CUDA` 版本不匹配
- 宿主机当时原本使用的是 `CUDA 13.0`
- 后来为了和当前 `PyTorch` / `CuRobo` 依赖链对齐，实际切换到了 `CUDA 12.8`

当前仓库里已经能对上的典型报错是：

> The detected CUDA version (11.8) mismatches the version that was used to compile PyTorch (12.8).

这类问题的本质是：

- 宿主机或当前可见的 CUDA 版本
- `PyTorch` 所使用的 CUDA 编译版本
- `CuRobo` 安装时实际依赖到的 CUDA 工具链

三者没有对齐。

### 当时真实采取的修复动作

当前已确认的真实修复动作不是“继续硬装当前版本”，而是主动把 CUDA 版本往下对齐：

- 原本环境：`CUDA 13.0`
- 最终对齐：`CUDA 12.8`

这一步很关键，因为它说明当时的问题不是简单的“少一个包”，而是底层 GPU 工具链版本已经超出当前这套依赖组合的稳定范围。

换句话说，当时真正有效的动作是：

1. 识别出 `CuRobo` 失败不是普通 Python 包问题
2. 确认问题落在 CUDA / PyTorch / CuRobo 的版本对齐
3. 放弃继续沿用 `CUDA 13.0`
4. 改为切换到 `CUDA 12.8`

在把 CUDA 版本对齐到 `12.8` 之后，我们当时不是直接继续跑主程序，而是先回到 `CuRobo` 这一步重新安装。

也就是说，当时的实际动作顺序已经确认到这里：

1. 先走 `UV Setup`
2. 卡在 `CuRobo`
3. 识别到是 `CUDA` 版本不匹配
4. 发现宿主机原本是 `CUDA 13.0`
5. 把环境对齐到 `CUDA 12.8`
6. 先重新安装 `CuRobo`

### 这次修复并没有一次过

已确认事实：

- 即使把 CUDA 从 `13.0` 对齐到 `12.8`
- 并且重新安装了 `CuRobo`
- 整个问题也没有在这一步一次性结束

这说明当时的真实排障节奏不是：

- 找到一个错
- 改一次
- 立即结束

而更像是：

- 先修掉最明显的版本级阻塞
- 让安装过程往前推进
- 然后暴露出下一层问题

这也是这份记录为什么必须按“真实踩坑过程”来写的原因之一，因为如果只写最终解法，会误导成“把 CUDA 改成 12.8 就全部解决了”，但真实情况并不是这样。

### 阶段 4：继续卡在系统库或编译依赖

已确认事实：

- 在把 CUDA 对齐到 `12.8` 并重装 `CuRobo` 之后
- 第二层卡点不是新的主流程决策问题
- 而是更底层的系统库或编译依赖问题

这说明当时的排障已经从“版本大方向不对”进入到了“缺哪个底层依赖就补哪个”的阶段。

结合当前仓库里的排障文档，这一层最可疑的典型问题包括：

- 缺少 `Python.h`
- 缺少 `libgmpxx.so.4`
- 缺少 `libGLU.so.1`
- 其他会影响 CuRobo / Isaac Sim 编译或加载的系统级依赖

这一步也很有代表性，因为它说明即使主版本方向调对了，类 Isaac Sim / CuRobo 的依赖链仍然会继续暴露宿主机层面的缺口。

### 当前还无法确认的细节

已确认但未定序：

- 我们在这一阶段不止遇到“版本不匹配”
- 还继续遇到了系统库或编译依赖问题

当前还不能写死的内容：

- 到底是先缺 `Python.h`
- 还是先缺 `libgmpxx.so.4`
- 还是先缺 `libGLU.so.1`

这里先保留为“顺序待确认”，避免把记不清的细节写成确定事实。

### 阶段 5：环境阶段本身就拉得很长

已确认事实：

- 在经历 `CUDA 13.0 -> 12.8`、重装 `CuRobo`、系统库与编译依赖排障之后
- 我们当时仍然没有进入“`simple` 已经可以稳定 import 或基本能跑”的阶段
- 也就是说，环境问题并不是几个命令就结束，而是占用了很长一段时间

这说明当时的真实感受更接近：

- 不是“修完 CuRobo 就通了”
- 而是“环境这一关本身就是一场长期拉锯”

这也解释了为什么你会特别强调“之前有很长一段时间都在配 SIMPLE 这个项目的环境”。从当前已确认事实看，这不是夸张描述，而是符合真实时间线的。

### 阶段 6：虽然拉扯很久，但没有切去 `Nix Runtime`

已确认事实：

- 我们当时没有正式切换到 `Nix Runtime`
- 也不是“UV 不行了就换 Nix”
- 而是一直在 `UV Setup` 这条路线上继续排障

这意味着当时的真实策略是：

- 不额外引入一条新的环境体系
- 优先在现有 `UV Setup` 路线上把问题一个个啃过去

这点很重要，因为如果后面有人看当前仓库里很重的 Nix 文档，可能会误以为我们当时已经转向了 Nix；但按现在已确认的历史，事实并不是这样。

### 阶段 7：`UV Setup` 后半段主要卡在 Isaac Sim / 图形运行时

已确认事实：

- 在一直坚持 `UV Setup` 的后半段
- 主要拉扯的问题已经不再只是 `CuRobo` 编译本身
- 而更像是 `Isaac Sim`、图形后端、动态库加载、无头运行这类运行时问题

这说明当时的排障层次又往前推进了一步：

1. 先是 Python / `uv`
2. 再是 `CuRobo`
3. 再是 CUDA 版本对齐
4. 再是系统库和编译依赖
5. 然后才暴露出 `Isaac Sim / 图形运行时` 这一层

结合当前仓库里的排障文档，这一阶段最可能关联的现象包括：

- 缺少 `libGLU.so.1`
- Vulkan 初始化失败
- GPU 设备创建失败
- Headless 模式与图形后端设置不一致
- `MUJOCO_GL`、`DISPLAY`、图形栈配置不合适

这类问题通常比“少一个 Python 包”更难受，因为它们往往不是安装时就报，而是运行到某一步才炸。

### 阶段 8：代表性的 Isaac Sim 运行时问题是图形动态库缺失

已确认事实：

- 在 `Isaac Sim / 图形运行时` 这一阶段
- 最像、也最容易代表当时痛感的问题，是类似 `libGLU.so.1` 这样的图形动态库缺失

这类问题的特点是：

- 前面的 Python 依赖、CUDA、CuRobo 可能都已经折腾过
- 但真正启动到图形相关运行阶段时，仍然会因为系统层面的图形库不完整而失败

当前仓库排障文档里已经有直接对应的修法：

```bash
sudo apt-get install libglu1-mesa
```

这说明当时我们的排障已经进入“补齐宿主机图形运行时依赖”的阶段，而不是还停留在 Python 包层面。

### 补完图形动态库后，环境仍然没有立刻稳定

已确认事实：

- 即使已经一路处理到图形动态库缺失这一层
- 环境也没有在这一步就真正完全拉起
- 也就是说，我们当时还没有顺利切入 `Psi-0` 联调主线

这再次说明当时的真实过程不是：

- 修一个关键库
- 环境立即恢复正常

而是：

- 每修掉一层问题
- 环境才往前推进一点
- 然后继续暴露下一层运行时或配置问题

这类“连锁暴露”是这个项目环境阶段最值得记录的特点之一。

这也是一个值得在文档里明确记下来的经验：

- 不是“CUDA 越新越好”
- 对这个项目来说，“和依赖链对齐”比“装最新版本”更重要

### 这个阶段暴露出的一个文档矛盾

当前仓库里与 CUDA 版本有关的描述并不完全一致：

- `docs/source/tutorials/installation.md` 提到测试过 `11.8` 和 `12.4`
- `docs/source/troubleshooting.md` 中这类 CuRobo 报错的修法又明确写到了安装 `12.8`
- README 的系统要求部分则把 CUDA 写成了 `12.x`

这意味着我们当时真实踩坑时，很可能遇到的不只是“版本错了”，而是：

- 文档里有多个版本信号
- 不同阶段的依赖对 CUDA 的要求并不完全一致
- 实际可行解可能不是“任意 12.x 都行”，而是要和当前 PyTorch / CuRobo 安装组合对齐

这一点后续值得继续补细，因为它属于真正会误导排障方向的关键信息。

---

## 16. 下一步要继续补写的内容

等你确认后，我会继续往这份文档里补下面几类内容：

1. 我们真实走过的安装路线
2. 真实踩坑时间线
3. 每个错误对应的现场现象
4. 当时为什么选了现在这条方案
5. 和 `Psi-0` 对接时真正的分界点和决策点

---

## 19. 记录节奏调整：先跳转到 `Psi-0` 对接主线

已确认事实：

- `SIMPLE` 环境阶段拉扯很久
- 即使处理到 CUDA、`CuRobo`、系统库、图形动态库这一层，环境问题当时仍未完全收口
- 现在这份记录不再继续深挖每一个环境细节，而是先转向记录 `Psi-0` 对接主线

这意味着接下来的文档补写策略会变成：

1. 先记清楚我们是什么时候开始碰 `Psi-0`
2. 先记清楚我们一开始碰的是训练、推理服务，还是 `SIMPLE` 客户端联调
3. 先把 `Psi-0` 这条线的关键分叉和卡点记录下来
4. 环境线剩余的细节以后再回补

## 20. `Psi-0` 对接主线

### 阶段 1：先碰训练环境，不是先碰推理服务

已确认事实：

- 我们开始碰 `Psi-0` 时，不是先起推理服务
- 也不是先做 `SIMPLE` 客户端联调
- 而是先去处理 `Psi-0` 的训练环境
- `Psi-0` 训练环境这一段本身没有成为主要阻塞点

这条顺序很重要，因为它说明当时的真实目标不是“先把现成模型跑起来看看”，而更像是：

- 先把 `Psi-0` 仓库自身的环境链路摸通
- 先确认训练侧依赖、变量、目录和资源组织方式
- 再进入后续下载权重和服务端启动

补充说明：

- 至少按当前回忆，训练环境这一段并没有像 `SIMPLE` 环境那样形成长期拉锯
- 这意味着 `Psi-0` 主线的第一个真正卡点，应该出现在 checkpoint、推理服务，或后续联调阶段

### 阶段 2：接着处理 checkpoint

已确认事实：

- 训练环境之后，我们接着处理的是 checkpoint
- 也就是先准备模型权重，再进入服务端启动
- 这一段的真实问题不是“不会下载”，而是完整 checkpoint 体量太大
- 因此当时没有把整套都下完，而是只下载了一个模型的 checkpoint

这和当前 README 里的主线是对得上的：

1. 先处理 `Psi-0` 环境变量与工作区
2. 再下载 `psi0/simple-checkpoints`
3. 最后再启动推理服务

### 这一阶段的真实取舍

这里已经能确认一个很重要的实际决策：

- 理论上的完整路径，是准备整套 `psi0/simple-checkpoints`
- 但真实操作里，checkpoint 太大，下载与存储成本太高
- 所以当时采取了更务实的做法：先只下载一个模型的 checkpoint

这说明 `Psi-0` 线的第一类障碍并不是纯技术错误，而是资源体量带来的现实约束：

- 下载量
- 存储占用
- 时间成本

这类问题虽然不一定会报错，但同样会直接改变联调策略。

### 只下一个 checkpoint 的真实目的

已确认事实：

- 当时只下载一个模型 checkpoint，并不是为了正式训练
- 主要目标是先把推理服务和 `SIMPLE` 联调整条链路跑通

这意味着当时的策略非常明确：

- 不是先追求“资源齐全”
- 而是先追求“链路可验证”

换句话说，当时的优先级排序更像是：

1. 先确认 `Psi-0` 服务能起来
2. 先确认 `SIMPLE` 能连上服务
3. 先确认动作请求和响应链路是通的
4. 在链路通了之后，再考虑是否扩展更多 checkpoint 或更完整资源

### 阶段 3：最后才碰推理服务和联调

已确认事实：

- 推理服务和联调不是 `Psi-0` 线的起点
- 而是排在训练环境和 checkpoint 之后

这意味着当时 `Psi-0` 这条线的真实顺序已经确认到：

1. 先处理训练环境
2. 再下载或准备 checkpoint
3. 最后才进入推理服务与 `SIMPLE` 联调

### 阶段 4：联调真正卡在 `SIMPLE` 评测阶段的环境错误

已确认事实：

- 在开始推理服务和联调之后
- 真正把链路卡住的，不是训练环境
- 也不是 checkpoint 本身
- 而是在 `SIMPLE` 发起评测时出现了环境错误

这说明当时的链路状态更接近：

- `Psi-0` 侧至少已经推进到了可以准备服务与权重的程度
- 但 `SIMPLE` 作为 `Simulation Client` 在评测入口这一侧仍然受环境问题影响

换句话说，当时并不是“模型侧完全没起来”，而是“联调走到评测这一步时，客户端环境仍然不稳”。

### 这里的“任务环境问题”具体指什么

结合当前仓库代码和 README，可以先把“`SIMPLE` 的任务环境问题”理解为下面这类问题，而不是泛指任意环境问题：

- 评测入口 `eval.py` / `eval_decoupled_wbc.py` 在创建任务环境时失败
- 评测所需的数据目录、任务名、split 或 `data-dir` 不匹配
- `simple-eval` 数据没有准备好，或者路径不对
- 任务环境本身依赖的资源、配置或图形运行条件不满足

也就是说，这里的“任务环境问题”更接近：

- `SIMPLE` 客户端在“进入具体任务评测”这一步卡住

而不是：

- `Psi-0` 模型本身不会推理
- checkpoint 本身不能用

---

## 17. 当前版本说明

当前版本已经完成：

- 基于仓库现有文档和代码确认 SIMPLE 的环境主线
- 确认 SIMPLE 和 Psi-0 的正式对接关系
- 整理当前仓库内已记录的常见问题和解决办法
- 建立最小术语表，避免后续文档混用词语

当前版本还没有完成：

- 你和我之前那段长时间实际配置经历的逐日、逐错、逐修复回放

这个部分需要你继续补充一轮，我再把它写成真正完整的“过程记录”。
