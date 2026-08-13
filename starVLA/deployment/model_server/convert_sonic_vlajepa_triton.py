from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import torch


DEFAULT_CHECKPOINT = (
    "/home/d086/fangbaozhong/Sonicstar/CKPTSONICSTAR/SONICSTAR/"
    "checkpoints/steps_90000_pytorch_model.pt"
)
DEFAULT_OUTPUT_DIR = "/home/d086/fangbaozhong/Sonicstar/CKPTSONICSTAR/Sonicstartriton/sonicstar_qwen3_flash"
DEFAULT_BASE_VLM = (
    "/home/d086/fangbaozhong/Sonicstar/SonicStar/starVLA/starVLA/playground/"
    "Pretrained_models/Qwen3-VL-2B-Instruct"
)
DEFAULT_BASE_ENCODER = (
    "/home/d086/fangbaozhong/Sonicstar/SonicStar/starVLA/starVLA/playground/"
    "Pretrained_models/VJEPA21/vjepa2_1_vitl_dist_vitG_384.pt"
)


def _resolve_sidecars(checkpoint: Path) -> tuple[Path, Path]:
    ckpt_dir = checkpoint.parent
    run_dir = checkpoint.parents[1]
    candidates = (
        (ckpt_dir / "config.yaml", ckpt_dir / "dataset_statistics.json"),
        (run_dir / "config.yaml", run_dir / "dataset_statistics.json"),
    )
    for config_path, stats_path in candidates:
        if config_path.exists() and stats_path.exists():
            return config_path, stats_path
    raise FileNotFoundError(f"Could not find config.yaml and dataset_statistics.json for {checkpoint}")


def _stats_dims(stats: dict[str, Any], stats_key: str) -> tuple[int, int]:
    if stats_key not in stats:
        raise KeyError(f"stats_key={stats_key!r} not found in dataset statistics keys={list(stats)}")
    state = stats[stats_key]["state"]
    action = stats[stats_key]["action"]
    state_ref = state.get("mean", state.get("min"))
    action_ref = action.get("mean", action.get("min"))
    return int(len(state_ref)), int(len(action_ref))


def _read_action_horizon(config_path: Path, default: int) -> int:
    text = config_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("action_horizon:"):
            return int(stripped.split(":", 1)[1].strip())
    return int(default)


def _load_draft_checkpoint_payload(draft_checkpoint_path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    ckpt = torch.load(Path(draft_checkpoint_path), map_location="cpu")
    meta: dict[str, Any] = {}
    if isinstance(ckpt, dict) and "draft_head" in ckpt:
        meta = dict(ckpt.get("meta", {}) or {})
        state_dict = ckpt.get("draft_head", {})
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("draft checkpoint must be a state_dict or a dict with key `draft_head`")
    if not isinstance(state_dict, dict):
        raise ValueError("draft_head state_dict missing or invalid")
    return meta, {str(k): v.detach().cpu() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}


def _draft_meta_from_state_dict(meta: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    inferred = dict(meta or {})
    if "_action_queries.weight" in state_dict:
        inferred.setdefault("chunk_m", int(state_dict["_action_queries.weight"].shape[0]))
        inferred.setdefault("img_dim", int(state_dict["_action_queries.weight"].shape[1]))
    if "_state_token.weight" in state_dict:
        inferred.setdefault("state_dim", int(state_dict["_state_token.weight"].shape[1]))
        inferred.setdefault("img_dim", int(state_dict["_state_token.weight"].shape[0]))
    if "_action_head.weight" in state_dict:
        inferred.setdefault("out_dim", int(state_dict["_action_head.weight"].shape[0]))
        inferred.setdefault("action_dim", int(state_dict["_action_head.weight"].shape[0]))
        inferred.setdefault("img_dim", int(state_dict["_action_head.weight"].shape[1]))
    if "_qwen_block.mlp.gate_proj.weight" in state_dict:
        inferred.setdefault("draft_hidden_dim", int(state_dict["_qwen_block.mlp.gate_proj.weight"].shape[0]))
        inferred.setdefault("hidden_dim", int(state_dict["_qwen_block.mlp.gate_proj.weight"].shape[0]))
    if "_gemma_block.mlp.gate_proj.weight" in state_dict:
        inferred.setdefault("draft_hidden_dim", int(state_dict["_gemma_block.mlp.gate_proj.weight"].shape[0]))
    if "_qwen_block.self_attn.q_proj.weight" in state_dict:
        inferred.setdefault("img_dim", int(state_dict["_qwen_block.self_attn.q_proj.weight"].shape[1]))
        q_out = int(state_dict["_qwen_block.self_attn.q_proj.weight"].shape[0])
        q_norm = state_dict.get("_qwen_block.self_attn.q_norm.weight")
        head_dim = int(q_norm.shape[0]) if q_norm is not None else int(inferred.get("draft_head_dim", 128))
        inferred.setdefault("draft_head_dim", head_dim)
        if head_dim > 0:
            inferred.setdefault("draft_num_heads", max(1, q_out // head_dim))
    if "_qwen_block.self_attn.k_proj.weight" in state_dict:
        k_out = int(state_dict["_qwen_block.self_attn.k_proj.weight"].shape[0])
        head_dim = int(inferred.get("draft_head_dim", 0) or 0)
        if head_dim > 0:
            inferred.setdefault("draft_num_kv_heads", max(1, k_out // head_dim))
    if "_gemma_block.self_attn.q_proj.weight" in state_dict:
        inferred.setdefault("img_dim", int(state_dict["_gemma_block.self_attn.q_proj.weight"].shape[1]))
    if "_gemma_block.self_attn.k_proj.weight" in state_dict:
        head_dim = int(state_dict["_gemma_block.self_attn.k_proj.weight"].shape[0])
        inferred.setdefault("draft_head_dim", head_dim)
        inferred.setdefault("draft_num_kv_heads", 1)
    return inferred


def _convert_draft_checkpoint(*, draft_checkpoint_path: str | Path, output_path: str | Path) -> Path:
    meta, state_dict = _load_draft_checkpoint_payload(draft_checkpoint_path)
    if "_qwen_block.self_attn.q_proj.weight" in state_dict:
        block_prefix = "_qwen_block"
        required_keys = {
            "_state_token.weight",
            "_state_token.bias",
            "_action_queries.weight",
            "_qwen_block.self_attn.q_proj.weight",
            "_qwen_block.self_attn.k_proj.weight",
            "_qwen_block.self_attn.v_proj.weight",
            "_qwen_block.self_attn.o_proj.weight",
            "_qwen_block.self_attn.q_norm.weight",
            "_qwen_block.self_attn.k_norm.weight",
            "_qwen_block.mlp.gate_proj.weight",
            "_qwen_block.mlp.up_proj.weight",
            "_qwen_block.mlp.down_proj.weight",
            "_qwen_block.input_layernorm.weight",
            "_qwen_block.post_attention_layernorm.weight",
            "_action_head.weight",
            "_action_head.bias",
        }
        runtime_family = "jepa_qwen3_vl_draft"
    else:
        block_prefix = "_gemma_block"
        required_keys = {
            "_state_token.weight",
            "_state_token.bias",
            "_action_queries.weight",
            "_gemma_block.self_attn.q_proj.weight",
            "_gemma_block.self_attn.k_proj.weight",
            "_gemma_block.self_attn.v_proj.weight",
            "_gemma_block.self_attn.o_proj.weight",
            "_gemma_block.mlp.gate_proj.weight",
            "_gemma_block.mlp.up_proj.weight",
            "_gemma_block.mlp.down_proj.weight",
            "_gemma_block.input_layernorm.weight",
            "_gemma_block.post_attention_layernorm.weight",
            "_action_head.weight",
            "_action_head.bias",
        }
        runtime_family = "pi0_gemma_draft"
    missing = sorted(required_keys.difference(state_dict))
    if missing:
        raise KeyError(f"draft checkpoint missing required tensors: {missing}")

    meta = _draft_meta_from_state_dict(meta, state_dict)
    meta.setdefault("runtime_family", runtime_family)
    meta.setdefault("draft_artifact_format", "spec_draft_triton_v2")
    meta.setdefault("draft_block_prefix", block_prefix)
    artifact = {
        "meta": dict(meta),
        "draft_state_in_proj_w": state_dict["_state_token.weight"].contiguous(),
        "draft_state_in_proj_b": state_dict["_state_token.bias"].contiguous(),
        "draft_action_queries": state_dict["_action_queries.weight"].contiguous(),
        "draft_qkv_w": torch.cat(
            [
                state_dict[f"{block_prefix}.self_attn.q_proj.weight"],
                state_dict[f"{block_prefix}.self_attn.k_proj.weight"],
                state_dict[f"{block_prefix}.self_attn.v_proj.weight"],
            ],
            dim=0,
        ).contiguous(),
        "draft_attn_o_w": state_dict[f"{block_prefix}.self_attn.o_proj.weight"].contiguous(),
        "draft_ffn_gate_w": state_dict[f"{block_prefix}.mlp.gate_proj.weight"].contiguous(),
        "draft_ffn_up_w": state_dict[f"{block_prefix}.mlp.up_proj.weight"].contiguous(),
        "draft_ffn_down_w": state_dict[f"{block_prefix}.mlp.down_proj.weight"].contiguous(),
        "draft_input_layernorm_w": state_dict[f"{block_prefix}.input_layernorm.weight"].contiguous(),
        "draft_post_attention_layernorm_w": state_dict[f"{block_prefix}.post_attention_layernorm.weight"].contiguous(),
        "draft_action_head_w": state_dict["_action_head.weight"].contiguous(),
        "draft_action_head_b": state_dict["_action_head.bias"].contiguous(),
    }
    if block_prefix == "_qwen_block":
        artifact["draft_attn_q_norm_w"] = state_dict["_qwen_block.self_attn.q_norm.weight"].contiguous()
        artifact["draft_attn_k_norm_w"] = state_dict["_qwen_block.self_attn.k_norm.weight"].contiguous()

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    return output_path


def _write_full_artifact(
    *,
    checkpoint: Path,
    output_path: Path,
    base_vlm_path: str | Path | None,
    base_encoder_path: str | Path | None,
    stats_key: str,
) -> tuple[Path, dict[str, Any]]:
    config_path, stats_path = _resolve_sidecars(checkpoint)
    state_dict = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"SonicStar checkpoint must be a state_dict: {checkpoint}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    state_dim, action_dim = _stats_dims(stats, stats_key)
    chunk_m = _read_action_horizon(config_path, default=40)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "meta": {
            "full_artifact_format": "jepa_full_triton_v1",
            "runtime_family": "sonicstar_qwen3_vl_full",
            "checkpoint": str(checkpoint),
            "config_path": str(config_path),
            "stats_path": str(stats_path),
            "base_vlm_path": None if base_vlm_path is None else str(Path(base_vlm_path).expanduser().resolve()),
            "base_encoder_path": None if base_encoder_path is None else str(Path(base_encoder_path).expanduser().resolve()),
            "stats_key": str(stats_key),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "chunk_m": chunk_m,
        },
        "config_yaml": config_path.read_text(encoding="utf-8"),
        "dataset_statistics": stats,
        "state_dict": {str(k): v.detach().cpu() for k, v in state_dict.items() if isinstance(v, torch.Tensor)},
    }
    torch.save(artifact, output_path)
    return output_path, dict(artifact["meta"])


def convert(args: argparse.Namespace) -> Path:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"SonicStar checkpoint not found: {checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    full_path, full_meta = _write_full_artifact(
        checkpoint=checkpoint,
        output_path=output_dir / "full_triton.pt",
        base_vlm_path=args.base_vlm_path,
        base_encoder_path=args.base_encoder_path,
        stats_key=args.stats_key,
    )

    manifest: dict[str, Any] = {
        "format": "jepa_runtime_manifest_v1",
        "runtime_family": "sonicstar_jepa_full_triton",
        "full_checkpoint_format": "jepa_full_triton_v1",
        "full_checkpoint_triton_supported": True,
        "checkpoint": str(checkpoint),
        "full_artifact": str(full_path.relative_to(output_dir)),
        "draft_checkpoint": None,
        "draft_artifact": None,
        "base_vlm_path": None if args.base_vlm_path is None else str(Path(args.base_vlm_path).expanduser().resolve()),
        "base_encoder_path": None if args.base_encoder_path is None else str(Path(args.base_encoder_path).expanduser().resolve()),
        "stats_key": str(args.stats_key),
        "env": "SonicStarMergedDataset001",
        "state_dim": int(full_meta["state_dim"]),
        "action_dim": int(full_meta["action_dim"]),
        "chunk_m": int(full_meta["chunk_m"]),
        "max_exec_steps": int(args.max_exec_steps),
        "note": (
            "SonicStar full checkpoint is packaged for the shared JEPA Triton/Inductor runtime. "
            "The runtime compiles safe tensor subgraphs and keeps the official Qwen3-VL/SonicStar model semantics."
        ),
    }

    if args.draft_checkpoint:
        draft_checkpoint = Path(args.draft_checkpoint).expanduser().resolve()
        if not draft_checkpoint.exists():
            raise FileNotFoundError(f"Sonic draft checkpoint not found: {draft_checkpoint}")
        draft_path = _convert_draft_checkpoint(
            draft_checkpoint_path=draft_checkpoint,
            output_path=output_dir / "draft_triton.pkl",
        )
        with draft_path.open("rb") as handle:
            draft_artifact = pickle.load(handle)
        draft_meta = dict(draft_artifact.get("meta", {}) or {})
        manifest.update(
            {
                "runtime_family": "sonicstar_jepa_full_and_draft_triton",
                "draft_checkpoint": str(draft_checkpoint),
                "draft_artifact": str(draft_path.relative_to(output_dir)),
                "draft_meta": draft_meta,
                "state_dim": int(draft_meta.get("state_dim", manifest["state_dim"])),
                "action_dim": int(draft_meta.get("action_dim", draft_meta.get("out_dim", manifest["action_dim"]))),
                "chunk_m": int(draft_meta.get("chunk_m", manifest["chunk_m"])),
                "max_exec_steps": int(draft_meta.get("max_exec_steps", manifest["max_exec_steps"])),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SonicStar VLA-JEPA checkpoints for the shared Triton runtime.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--draft-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-vlm-path", type=str, default=DEFAULT_BASE_VLM)
    parser.add_argument("--base-encoder-path", type=str, default=DEFAULT_BASE_ENCODER)
    parser.add_argument("--stats-key", type=str, default="sonic_humanoid")
    parser.add_argument("--max-exec-steps", type=int, default=20)
    return parser


def main() -> None:
    manifest_path = convert(build_argparser().parse_args())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Saved SonicStar Triton manifest to {manifest_path}")
    print(f"Saved SonicStar full artifact to {manifest_path.parent / 'full_triton.pt'}")
    if manifest.get("draft_artifact"):
        print(f"Saved SonicStar draft artifact to {manifest_path.parent / 'draft_triton.pkl'}")


if __name__ == "__main__":
    main()
