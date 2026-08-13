# VLA_JEPA 权重在 SonicStar 项目中的评测指南

本文档记录如何将 `VLA_JEPA` 训练得到的权重接入当前 `SonicStar` 项目，并在本地完成 policy server、在线推理与自动化评测。

本文档对应的是已经验证通过的一条完整链路：

- `VLA_JEPA` 训练得到的 checkpoint
- 当前 `SonicStar` 仓库中的 `starVLA + wbc`
- `run_starvla_inference.py`
- MuJoCo / `wbc` / deploy 自动化评测链路

## 1. 背景

当前项目里原始的 `SonicLatent` 评测链路默认面向 `QwenGR00T` 这类 framework，例如：

- `starVLA/examples/SonicLatent/eval_files/run_policy_server.sh`
- `run_sonicstar_vla_automation.sh`

而本次要评测的权重来自另一套训练框架：

- `framework.name: VLA_JEPA`

虽然这套 `VLA_JEPA` 权重与原始 `SonicLatent` 权重使用的是同一个数据集：

- `/home/d013/桌面/SonicStar/starVLA/starVLA/playground/Datasets/merged_dataset_001`

但它们的 framework 结构不同，不能只替换 checkpoint 路径直接复用原始 `QwenGR00T` server。

主要差异包括：

- `VLA_JEPA` 使用 `framework.name: VLA_JEPA`
- 权重中包含 `vj_encoder.*` 与 `vj_predictor.*`
- 推理 prompt 中会使用 `"{actions}"` 与 `"{e_actions}"` 占位替换
- 推理时从 Qwen 输出中抽取 `embodied_action_tokens` 再喂给 action head

因此，本项目中增加了一条独立的 `VLA_JEPA` 推理接入路径。

## 2. 目标

目标是让 `VLA_JEPA` 训练出来的权重可以在当前 `SonicStar` 项目中完成以下评测链路：

1. 启动 `VLA_JEPA` 专用 policy server
2. 使用现有 `run_starvla_inference.py` 作为在线推理客户端
3. 复用现有 `wbc` 仿真、相机发布、deploy 与发键流程
4. 通过自动化脚本完成整条评测流程

## 3. 依赖与前提

运行前请确保以下内容存在。

### 3.1 Conda / Python 环境

- `starVLA` conda 环境可正常使用
- `g1_deploy` conda 环境可正常使用
- `/home/d013/桌面/GR00T-WholeBodyControl/.venv_sim` 存在

### 3.2 权重与配置目录

`VLA_JEPA` 的 checkpoint 目录：

- `/home/d013/桌面/SonicStar/starVLA/starVLA/playground/myCKPT`

其中至少应包含：

- `config.yaml`
- `config.json`
- `dataset_statistics.json`
- `checkpoints/steps_20000_pytorch_model.pt`

### 3.3 预训练模型目录

以下预训练权重需要放到：

- `/home/d013/桌面/SonicStar/starVLA/starVLA/playground/Pretrained_models`

并至少包含以下目录：

- `Qwen3-VL-2B-Instruct`
- `vjepa2-vitl-fpc64-256`

本次验证使用到的实际路径为：

- `/home/d013/桌面/SonicStar/starVLA/starVLA/playground/Pretrained_models/Qwen3-VL-2B-Instruct`
- `/home/d013/桌面/SonicStar/starVLA/starVLA/playground/Pretrained_models/vjepa2-vitl-fpc64-256`

### 3.4 配置文件中的本地路径

`myCKPT` 下配置文件需要指向本地预训练目录，而不是云上路径。

当前已对齐为：

- [config.yaml](/home/d013/桌面/SonicStar/starVLA/starVLA/playground/myCKPT/config.yaml)
- [config.json](/home/d013/桌面/SonicStar/starVLA/starVLA/playground/myCKPT/config.json)

关键字段为：

- `framework.name: VLA_JEPA`
- `framework.qwenvl.base_vlm: /home/d013/桌面/SonicStar/starVLA/starVLA/playground/Pretrained_models/Qwen3-VL-2B-Instruct`
- `framework.vj2_model.base_encoder: /home/d013/桌面/SonicStar/starVLA/starVLA/playground/Pretrained_models/vjepa2-vitl-fpc64-256`

## 4. 本次接入做了哪些代码改动

为了让 `VLA_JEPA` 权重能在当前项目里评测，增加或修改了以下内容。

### 4.1 新增 `VLA_JEPA` framework

新增文件：

- [VLA_JEPA.py](/home/d013/桌面/SonicStar/starVLA/starVLA/model/framework/VLM4A/VLA_JEPA.py)

作用：

- 按 `framework.name: VLA_JEPA` 构建模型
- 加载 `qwen_vl_interface`
- 加载 `vj_encoder`
- 构建 `vj_predictor`
- 在推理时从 Qwen 输出中提取 `embodied_action_tokens`
- 调用 action head 输出 `normalized_actions`

### 4.2 补入 `V-JEPA2` 相关 world-model 模块

新增文件：

- [vj2_predictor.py](/home/d013/桌面/SonicStar/starVLA/starVLA/model/modules/world_model/vj2_predictor.py)
- [vj2_modules.py](/home/d013/桌面/SonicStar/starVLA/starVLA/model/modules/world_model/vj2_modules.py)
- [vj2_tensors.py](/home/d013/桌面/SonicStar/starVLA/starVLA/model/modules/world_model/vj2_tensors.py)

作用：

- 让 `VLA_JEPA` framework 中的 `vj_predictor` 结构与训练时保持一致

### 4.3 扩展当前 Qwen3 输入组装能力

修改文件：

- [QWen3.py](/home/d013/桌面/SonicStar/starVLA/starVLA/model/modules/vlm/QWen3.py)

新增能力：

- `prompt_replace_dict`
- `prompt_template`

作用：

- 支持 `VLA_JEPA` 在推理时将 `"{actions}"` 与 `"{e_actions}"` 替换为 action token / embodied action token 模板

### 4.4 新增 `VLA_JEPA` 专用 policy server 入口

新增文件：

- [server_policy_vlajepa.py](/home/d013/桌面/SonicStar/starVLA/deployment/model_server/server_policy_vlajepa.py)

作用：

- 从 `myCKPT` 加载 `VLA_JEPA` 模型
- 启动 websocket policy server
- 将现有在线推理客户端发来的 `examples` 请求转换为：
  - `batch_images`
  - `instructions`
  - `state`

这样无需修改现有的 `run_starvla_inference.py` 客户端协议。

### 4.5 新增 `VLA_JEPA` 专用启动脚本

新增文件：

- [run_policy_server_vlajepa.sh](/home/d013/桌面/SonicStar/starVLA/examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh)

默认使用：

- `/home/d013/桌面/SonicStar/starVLA/starVLA/playground/myCKPT/checkpoints/steps_20000_pytorch_model.pt`

### 4.6 新增独立自动化脚本

新增文件：

- [run_sonicstar_vlajepa_automation.sh](/home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh)

作用：

- 保留原来的 `run_sonicstar_vla_automation.sh`
- 新增一条只面向 `VLA_JEPA + myCKPT` 的自动化评测入口

## 5. 为什么不能只替换原始 checkpoint

原始 `SonicLatent` policy server 默认对应的是 `QwenGR00T` 风格模型。

而 `myCKPT` 实际对应：

- `framework.name: VLA_JEPA`

它与原始链路的关键差异包括：

- state_dict 中多出 `vj_encoder.*`
- state_dict 中多出 `vj_predictor.*`
- prompt 中额外使用 action token 与 embodied action token
- 推理时依赖 `embodied_action_tokens` 作为 action head 条件

因此，不能只把：

- `run_policy_server.sh`

里的 `CKPT_PATH` 替换成 `myCKPT`，否则会出现：

- framework 不匹配
- 权重 shape mismatch
- tokenizer / special token 不一致

## 6. 推荐评测流程

推荐按照以下顺序进行。

### 6.1 单独启动 `VLA_JEPA` policy server

在 `starVLA` 环境中运行：

```bash
cd /home/d013/桌面/SonicStar/starVLA
bash examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh
```

作用：

- 验证 `myCKPT`、`Qwen3-VL-2B-Instruct`、`vjepa2-vitl-fpc64-256` 是否都能正确加载
- 验证 websocket policy server 是否能正常起在 `10093`

如果这一步失败，不要先跑自动化，先解决 server 启动问题。

### 6.2 全链路自动化评测

在项目根目录运行：

```bash
cd /home/d013/桌面/SonicStar
bash run_sonicstar_vlajepa_automation.sh
```

这份脚本会自动拉起：

1. `VLA_JEPA` policy server
2. online inference
3. MuJoCo sim loop
4. camera viewer
5. deploy
6. 手动确认后发送 `k / i / p`

## 7. 自动化脚本使用说明

自动化脚本路径：

- [run_sonicstar_vlajepa_automation.sh](/home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh)

默认 checkpoint：

```bash
CKPT_PATH="/home/d013/桌面/SonicStar/starVLA/starVLA/playground/myCKPT/checkpoints/steps_20000_pytorch_model.pt"
```

默认 policy server 启动脚本：

```bash
POLICY_SERVER_SCRIPT="examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh"
```

默认 prompt：

```bash
POLICY_PROMPT="pick up the cylinder and throw it into the trash bin"
```

如果以后要切换别的 `VLA_JEPA` checkpoint，只需要修改：

- `CKPT_PATH`

## 8. 在线推理链路说明

当前 `VLA_JEPA` 评测链路复用了现有的在线推理脚本：

- [run_starvla_inference.py](/home/d013/桌面/SonicStar/starVLA/examples/SonicLatent/eval_files/run_starvla_inference.py)

原因是：

- `VLA_JEPA` policy server 最终同样返回 `normalized_actions`
- 当前数据集与原始 `SonicLatent` 使用的是同一个：
  - `merged_dataset_001`
- action 维度与拆分逻辑保持一致：
  - `64 motion_token`
  - `7 left_hand_joints`
  - `7 right_hand_joints`

因此可以直接复用现有 `step2` 客户端。

## 9. 常见问题与排查

### 9.1 `huggingface-hub` 版本冲突

如果出现类似错误：

```text
ImportError: huggingface-hub>=0.34.0,<1.0 is required ...
```

原因：

- `huggingface_hub` 被升级到了 `1.x`
- 当前 `transformers` 版本要求 `<1.0`

修复方式：

```bash
conda activate starVLA
pip install "huggingface-hub>=0.34.0,<1.0"
```

或者固定版本：

```bash
pip install huggingface-hub==0.36.0
```

### 9.2 `flash_attn` 未安装

如果看到：

```text
[WARNING] flash_attn not installed, falling back to sdpa
```

这是警告，不一定是错误。

当前实现已经支持自动回退到：

- `sdpa`

通常不会阻塞评测，只可能影响速度。

### 9.3 `vj_predictor` shape mismatch

如果出现：

```text
size mismatch for vj_predictor.predictor_embed.weight ...
```

优先检查：

- `framework.vj2_model.num_video_views`
- `VLA_JEPA.py` 中 `embed_dim` 的构造方式

本次最终采用的逻辑是：

- `embed_dim = vj_encoder.config.hidden_size * num_video_views`

而不是直接写死乘以 `2`。

### 9.4 policy server 能起，但 online inference 不通

优先检查：

- `10093` 端口是否已占用
- `run_starvla_inference.py` 是否连接到 `127.0.0.1:10093`
- 自动化脚本 preflight cleanup 是否清理了旧 server 进程

### 9.5 自动化第一次通，第二次不通

优先检查：

- 是否有旧的 gnome-terminal 进程残留
- 端口 `10093` 和 `5555` 是否被旧进程占用
- `pkill` 是否清理到了：
  - `run_policy_server_vlajepa.sh`
  - `run_starvla_inference.py`

## 10. 推荐测试策略

建议第一次接入成功后，不要只跑一遍，而是多轮验证：

1. 先单独验证 `step1` policy server 能否稳定启动
2. 再跑整条自动化链路
3. 多次重复启动与停止
4. 观察动作输出是否稳定
5. 观察 `k / i / p` 流程是否每次都能复现

重点关注：

- 是否偶发退出
- 是否偶发卡在 waiting for camera msg
- 是否端口残留
- 是否动作抖动或发散

## 11. 本次接入的结论

本次已经验证：

- `VLA_JEPA` 训练得到的权重不能直接替换原始 `QwenGR00T` policy server
- 需要补齐 `VLA_JEPA` framework、`vj_predictor` 相关模块与 prompt 替换逻辑
- 需要单独的 `VLA_JEPA` policy server 入口
- 但在完成这些适配后，可以复用现有：
  - `run_starvla_inference.py`
  - `wbc` 仿真链路
  - deploy
  - 自动化评测流程

当前推荐入口为：

- 单独 server：
  - [run_policy_server_vlajepa.sh](/home/d013/桌面/SonicStar/starVLA/examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh)
- 全链路自动化：
  - [run_sonicstar_vlajepa_automation.sh](/home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh)

## 12. 相关文件索引

本次评测链路相关文件如下：

- [run_policy_server_vlajepa.sh](/home/d013/桌面/SonicStar/starVLA/examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh)
- [server_policy_vlajepa.py](/home/d013/桌面/SonicStar/starVLA/deployment/model_server/server_policy_vlajepa.py)
- [VLA_JEPA.py](/home/d013/桌面/SonicStar/starVLA/starVLA/model/framework/VLM4A/VLA_JEPA.py)
- [QWen3.py](/home/d013/桌面/SonicStar/starVLA/starVLA/model/modules/vlm/QWen3.py)
- [vj2_predictor.py](/home/d013/桌面/SonicStar/starVLA/starVLA/model/modules/world_model/vj2_predictor.py)
- [run_sonicstar_vlajepa_automation.sh](/home/d013/桌面/SonicStar/run_sonicstar_vlajepa_automation.sh)
- [myCKPT/config.yaml](/home/d013/桌面/SonicStar/starVLA/starVLA/playground/myCKPT/config.yaml)
