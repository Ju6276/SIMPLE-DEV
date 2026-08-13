# SIMPLE

SIMPLE 是一个面向类人机器人移动操作任务的仿真、数据生成、策略训练接入与评测项目。它围绕环境运行、数据流水线、与 Psi-0 的训练和推理对接组织代码与文档。

## Language

**SIMPLE**:
一个仿真驱动的策略学习与评测项目，覆盖环境搭建、任务运行、数据生成、后处理和模型评测。
_Avoid_: 这个项目, 仿真脚本集合

**Task**:
SIMPLE 中一个可运行、可采集、可评测的具体任务实例，通常带有任务名，如 `G1WholebodyXMovePickTeleop-v0`。
_Avoid_: case, demo

**Teleop Task**:
通过人类遥操作采集或回放数据的任务类型，通常在命名上以 `Teleop-v0` 结尾，并常配合 `decoupled_wbc` 流程使用。
_Avoid_: 手动数据, 人控 case

**Motion Planning Task**:
通过运动规划流水线自动生成数据的任务类型，通常在命名上以 `MP-v0` 结尾。
_Avoid_: 自动 case, 规划 demo

**Psi-0**:
与 SIMPLE 对接的基础模型训练与推理栈。SIMPLE 负责产出兼容数据、对接推理服务，并作为评测客户端运行环境。
_Avoid_: PSI-0, psi zero 模型仓库

**Psi-0 Compatible Dataset**:
经过 `postprocess_psi0.py` 或 `postprocess_psi0_sonic.py` 处理后，可直接用于 Psi-0 训练或评测的数据格式。
_Avoid_: 清洗后数据, 训练版数据

**Inference Server**:
运行在 Psi-0 仓库侧的模型推理服务，SIMPLE 作为客户端通过 HTTP 向它请求动作。
_Avoid_: 模型端, 远端脚本

**Simulation Client**:
运行在 SIMPLE 仓库侧的仿真执行端，负责环境运行、观测组织和动作下发。
_Avoid_: 本地端, 环境脚本

**UV Setup**:
基于 `uv` 和本地虚拟环境的安装路径，适合已经具备 NVIDIA 驱动与 CUDA 基础的 Linux 主机。
_Avoid_: 普通安装, 默认安装

**Nix Runtime**:
基于 Nix 的隔离运行时路径，强调宿主机边界收敛、依赖可复现和开发环境一致性。
_Avoid_: nix 安装方式, shell 模式

**CuRobo**:
SIMPLE 当前关键依赖之一，用于 GPU 加速的运动规划、正逆运动学等能力。
_Avoid_: 可选 CUDA 包, 机械臂插件
