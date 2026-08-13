from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


def _prefer_local_starvla() -> None:
    starvla_root = Path(__file__).resolve().parents[2]
    for path in (starvla_root, starvla_root / "starVLA"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_prefer_local_starvla()

from deployment.model_server.server_policy_vlajepa import VLAJEPAPolicyAdapter
from starVLA.model.framework.base_framework import baseframework


DEFAULT_CKPT = "/home/d086/fangbaozhong/Sonicstar/CKPTSONICSTAR/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt"
DEFAULT_MANIFEST = "/home/d086/fangbaozhong/Sonicstar/CKPTSONICSTAR/Sonicstartriton/sonicstar_full_triton/manifest.json"
DEFAULT_IMAGE = "/home/d086/fangbaozhong/Sonicstar/SonicStar/heatmap/20260707_115629/step_000000_infer_0009_rgb.png"
DEFAULT_PROMPT = "pick up the cylinder and throw it into the trash bin"


def _load_state(path: str | None, *, state_dim: int) -> np.ndarray:
    if not path:
        return np.zeros((1, 1, state_dim), dtype=np.float32)
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("normalized_state", "state", "robot_state"):
            if key in payload:
                payload = payload[key]
                break
    arr = np.asarray(payload, dtype=np.float32)
    if arr.shape == (state_dim,):
        arr = arr.reshape(1, 1, state_dim)
    elif arr.shape == (1, state_dim):
        arr = arr.reshape(1, 1, state_dim)
    if arr.shape != (1, 1, state_dim):
        raise ValueError(
            f"expected state shape {(1, 1, state_dim)}, {(1, state_dim)} or {(state_dim,)}, got {arr.shape}"
        )
    return arr.astype(np.float32)


def _metrics(diff: np.ndarray, lo: int, hi: int) -> dict[str, float]:
    part = np.asarray(diff[..., lo:hi], dtype=np.float32)
    return {
        "max_abs": float(np.max(np.abs(part))),
        "mean_abs": float(np.mean(np.abs(part))),
        "rms": float(np.sqrt(np.mean(part * part))),
    }


def _shape_list(array: np.ndarray) -> list[int]:
    return [int(x) for x in array.shape]


def _run_once(
    *,
    vla,
    adapter: VLAJEPAPolicyAdapter,
    image: Image.Image,
    prompt: str,
    state: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    with torch.inference_mode():
        pytorch_out = vla.predict_action(batch_images=[[image]], instructions=[prompt], state=state)

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    with torch.inference_mode():
        triton_out, triton_timing = adapter._predict_full_shared_prefix(  # noqa: SLF001
            batch_images=[[image]],
            instructions=[prompt],
            state=state,
        )
    triton_out = dict(triton_out)
    triton_out["policy_timing"] = dict(triton_timing)
    return pytorch_out, triton_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT)
    parser.add_argument("--triton-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--state-json", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-dim", type=int, default=46)
    parser.add_argument("--action-dim", type=int, default=78)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--out-json", default="data/result/full_parity/sonic_full_parity.json")
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    image = Image.open(Path(args.image).expanduser()).convert("RGB")
    state = _load_state(args.state_json, state_dim=int(args.state_dim))

    vla = baseframework.from_pretrained(args.ckpt_path)
    if bool(args.use_bf16):
        vla = vla.to(torch.bfloat16)
    vla = vla.to(args.device).eval()

    adapter = VLAJEPAPolicyAdapter(
        vla,
        device=args.device,
        action_dim=int(args.action_dim),
        state_dim=int(args.state_dim),
        backend="triton",
        triton_manifest=args.triton_manifest,
        max_exec_steps=20,
        draft_checkpoint=None,
        tau_radius=0.25,
        verify_dist_dims=int(args.action_dim),
        t_list=[0.1, 0.05],
        draft_full_fallback=True,
        force_full_each_round=False,
        periodic_full_every_n_draft_rounds=1,
        enable_shared_prefix_full=True,
    )

    for idx in range(max(0, int(args.warmup_runs))):
        logging.info("warmup run %d/%d", idx + 1, int(args.warmup_runs))
        _run_once(vla=vla, adapter=adapter, image=image, prompt=args.prompt, state=state, seed=int(args.seed))

    pytorch_out, triton_out = _run_once(
        vla=vla,
        adapter=adapter,
        image=image,
        prompt=args.prompt,
        state=state,
        seed=int(args.seed),
    )
    pytorch_actions = np.asarray(pytorch_out["normalized_actions"], dtype=np.float32)
    triton_actions = np.asarray(triton_out["normalized_actions"], dtype=np.float32)
    if pytorch_actions.shape != triton_actions.shape:
        raise ValueError(f"shape mismatch: pytorch={pytorch_actions.shape} triton={triton_actions.shape}")

    diff = triton_actions - pytorch_actions
    result = {
        "ckpt_path": str(args.ckpt_path),
        "triton_manifest": str(args.triton_manifest),
        "image": str(args.image),
        "state_json": args.state_json,
        "prompt": str(args.prompt),
        "seed": int(args.seed),
        "warmup_runs": int(args.warmup_runs),
        "state_shape": _shape_list(state),
        "action_shape": _shape_list(pytorch_actions),
        "overall": _metrics(diff, 0, int(args.action_dim)),
        "motion_token_0_64": _metrics(diff, 0, 64),
        "left_hand_64_71": _metrics(diff, 64, 71),
        "right_hand_71_78": _metrics(diff, 71, 78),
        "triton_timing": triton_out.get("policy_timing", {}),
    }
    out_path = Path(args.out_json).expanduser()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
