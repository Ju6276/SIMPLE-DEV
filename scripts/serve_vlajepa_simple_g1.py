"""
Run a SIMPLE-compatible websocket inference server backed by a local VLA-JEPA checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _append_vlajepa_repo_to_path(vlajepa_root: Path) -> None:
    repo_str = str(vlajepa_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _unnormalize_actions(raw_actions: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    q01 = np.array(action_stats["q01"], dtype=np.float32)
    q99 = np.array(action_stats["q99"], dtype=np.float32)
    mask = np.array(
        action_stats.get("mask", np.ones_like(q01, dtype=bool)),
        dtype=bool,
    )
    clipped = np.clip(raw_actions, -1.0, 1.0)
    scaled = 0.5 * (clipped + 1.0) * (q99 - q01) + q01
    return np.where(mask, scaled, clipped).astype(np.float32)


class SimpleG1VlajepaPolicy:
    def __init__(self, ckpt_path: Path, device: str, use_bf16: bool):
        self._ckpt_path = ckpt_path
        self._device = torch.device(device)

        vlajepa_root = ckpt_path.parents[2]
        _append_vlajepa_repo_to_path(vlajepa_root)

        from starVLA.model.framework.base_framework import baseframework

        self._model = baseframework.from_pretrained(str(ckpt_path))
        if use_bf16:
            self._model = self._model.to(torch.bfloat16)
        self._model = self._model.to(self._device).eval()
        self._action_stats = self._model.norm_stats["g1_handover"]["action"]

    def predict_action(
        self,
        batch_images,
        instructions,
        state: np.ndarray,
        reset: bool = False,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        instruction = instructions[0]
        pil_image = batch_images[0][0]
        if isinstance(pil_image, np.ndarray):
            pil_image = Image.fromarray(pil_image.astype(np.uint8), mode="RGB")
        output = self._model.predict_action(
            batch_images=[[pil_image]],
            instructions=[instruction],
            state=np.asarray(state, dtype=np.float32),
        )
        normalized_actions = output["normalized_actions"]
        actions = _unnormalize_actions(normalized_actions, self._action_stats)
        return {"actions": actions[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-path",
        default="/home/d013/桌面/VLA-JEPA/checkpoints/CKPT/steps_50000_pytorch_model.pt",
    )
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--use-bf16", action="store_true")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    vlajepa_root = ckpt_path.parents[2]
    _append_vlajepa_repo_to_path(vlajepa_root)

    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    policy = SimpleG1VlajepaPolicy(
        ckpt_path=ckpt_path,
        device=device,
        use_bf16=args.use_bf16,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating SIMPLE VLA-JEPA server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "env": "simple_g1_handover",
            "checkpoint": str(ckpt_path),
        },
    )
    logging.info("server running ...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    for key in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        os.environ.pop(key, None)
    main()
