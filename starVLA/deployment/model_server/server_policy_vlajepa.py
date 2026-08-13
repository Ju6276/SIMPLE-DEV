import argparse
import contextlib
import logging
import os
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework


def _to_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[-1] == 4:
        return Image.fromarray(array, mode="RGBA").convert("RGB")
    if array.ndim == 3 and array.shape[-1] == 3:
        return Image.fromarray(array, mode="RGB")
    raise ValueError(f"Unsupported image shape for policy server: {array.shape}")


def _normalize_image_list(images):
    if isinstance(images, (Image.Image, np.ndarray)):
        return [_to_pil_rgb(images)]
    if isinstance(images, list):
        return [_to_pil_rgb(image) for image in images]
    raise ValueError(f"Unsupported image container for policy server: {type(images)}")


def _clone_cache_snapshot(snapshot: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    if snapshot is None:
        return None
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in snapshot.items()
    }


class VLAJEPAPolicyAdapter:
    """Adapt current websocket `examples` requests to VLA_JEPA.predict_action()."""

    def __init__(
        self,
        policy,
        *,
        device: str,
        action_dim: int,
        state_dim: int,
        backend: str,
        triton_manifest: str | None,
        max_exec_steps: int,
        draft_checkpoint: str | None,
        tau_radius: float,
        verify_dist_dims: int,
        t_list: list[float],
        draft_full_fallback: bool,
        force_full_each_round: bool,
        periodic_full_every_n_draft_rounds: int,
        draft_start_after_full_rounds: int,
        draft_min_accept_steps: int,
        enable_shared_prefix_full: bool,
    ):
        self.policy = policy
        self.device = str(device)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.backend = str(backend)
        self.triton_manifest = None if triton_manifest is None else Path(triton_manifest).expanduser().resolve()
        self.max_exec_steps = int(max_exec_steps)
        self.tau_radius = float(tau_radius)
        self.verify_dist_dims = int(verify_dist_dims)
        self.t_list = tuple(float(t) for t in t_list)
        self.draft_full_fallback = bool(draft_full_fallback)
        self.force_full_each_round = bool(force_full_each_round)
        self.periodic_full_every_n_draft_rounds = int(periodic_full_every_n_draft_rounds)
        self.draft_start_after_full_rounds = int(max(0, draft_start_after_full_rounds))
        self.draft_min_accept_steps = int(max(0, draft_min_accept_steps))
        self.enable_shared_prefix_full = bool(enable_shared_prefix_full)
        self.draft_enabled = bool(draft_checkpoint)
        self.pending_full_fallback = True
        self.draft_rounds_since_full = 0
        self.full_rounds_served = 0
        self.full_cache_snapshot: dict[str, torch.Tensor] | None = None
        self.draft_head = None
        self.draft_meta: dict[str, Any] = {}
        if draft_checkpoint:
            self._load_draft_head(draft_checkpoint)
        self.triton_runtime = None
        if self.backend == "triton":
            self._init_triton_runtime()

    def _sync_cuda(self) -> None:
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def _load_draft_head(self, draft_checkpoint: str) -> None:
        from deployment.model_server.draft_head import DraftChunkHead
        from deployment.model_server.draft_head import Qwen3VLDraftChunkHead

        path = Path(draft_checkpoint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Sonic draft checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu")
        if not isinstance(ckpt, dict) or "draft_head" not in ckpt:
            raise ValueError(f"draft checkpoint must be a dict with key 'draft_head': {path}")
        meta = dict(ckpt.get("meta", {}) or {})
        if meta.get("condition") != "pre_vlm_prefix_embs":
            raise ValueError(f"draft checkpoint condition must be pre_vlm_prefix_embs, got {meta.get('condition')!r}")
        if int(meta.get("state_dim", self.state_dim)) != self.state_dim:
            raise ValueError(f"draft state_dim mismatch: ckpt={meta.get('state_dim')} server={self.state_dim}")
        if int(meta.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError(f"draft action_dim mismatch: ckpt={meta.get('action_dim')} server={self.action_dim}")

        token_dim = int(meta.get("token_dim", meta.get("img_dim", meta.get("prefix_dim", 0))))
        if token_dim <= 0:
            raise ValueError(f"draft checkpoint missing token_dim/img_dim/prefix_dim metadata: {path}")
        draft_arch = str(meta.get("draft_arch", "qwen3_vl_text_block")).lower()
        draft_cls = Qwen3VLDraftChunkHead if draft_arch in {"qwen3_vl_text_block", "qwen3_vl_block", "qwen3"} else DraftChunkHead
        chunk_m = int(meta.get("chunk_m", 40))
        num_heads = int(meta.get("draft_num_heads", draft_cls._resolve_num_heads(token_dim)))
        num_kv_heads = int(meta.get("draft_num_kv_heads", 1))
        head_dim = int(meta.get("draft_head_dim", max(1, token_dim // max(1, num_heads))))
        draft_kwargs = {
            "img_dim": token_dim,
            "state_dim": self.state_dim,
            "out_dim": self.action_dim,
            "chunk_m": chunk_m,
            "hidden_dim": int(meta.get("hidden_dim", meta.get("draft_intermediate_size", 6144))),
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
        if draft_cls is Qwen3VLDraftChunkHead:
            draft_kwargs.update(
                {
                    "hidden_act": str(meta.get("draft_hidden_act", "silu")),
                    "max_position_embeddings": int(meta.get("draft_max_position_embeddings", 128000)),
                    "rms_norm_eps": float(meta.get("draft_rms_norm_eps", 1e-6)),
                    "rope_theta": float(meta.get("draft_rope_theta", 5000000.0)),
                    "rope_scaling": meta.get("draft_rope_scaling", None),
                    "attention_bias": bool(meta.get("draft_attention_bias", False)),
                    "attention_dropout": float(meta.get("draft_attention_dropout", 0.0)),
                }
            )
        head = draft_cls(**draft_kwargs)
        head.load_state_dict(ckpt["draft_head"], strict=True)
        self.draft_head = head.to(self.device).eval()
        self.draft_meta = meta
        logging.info("Loaded Sonic draft checkpoint %s chunk_m=%d action_dim=%d", path, chunk_m, self.action_dim)

    def _init_triton_runtime(self) -> None:
        import importlib.util

        runtime_path = Path(__file__).resolve().parent / "triton" / "jepa_triton_runtime.py"
        if not runtime_path.exists():
            raise FileNotFoundError(f"SonicStar Triton runtime not found: {runtime_path}")
        spec = importlib.util.spec_from_file_location("sonic_jepa_triton_runtime", runtime_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load JEPA Triton runtime from {runtime_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(self.policy, "prepare_draft_prefix"):
            setattr(self.policy, "prepare_draft_prefix", self._prepare_stock_prefix)
        self.triton_runtime = module.JEPATritonRuntime(
            model=self.policy,
            draft_head=self.draft_head,
            device=self.device,
            manifest_path=self.triton_manifest,
            verify_t_list=self.t_list,
        )
        logging.info("Initialized Sonic Triton backend: %s", self.triton_runtime.describe())

    @staticmethod
    def _prepare_qwen3_prefix_embeddings(qwen_interface: torch.nn.Module, batch_inputs: dict[str, torch.Tensor]):
        qwen_model = qwen_interface.model.model
        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs.get("attention_mask", None)
        pixel_values = batch_inputs.get("pixel_values", None)
        pixel_values_videos = batch_inputs.get("pixel_values_videos", None)
        image_grid_thw = batch_inputs.get("image_grid_thw", None)
        video_grid_thw = batch_inputs.get("video_grid_thw", None)
        device_type = "cuda" if input_ids.device.type == "cuda" else "cpu"
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_embeds, _ = qwen_model.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = qwen_model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            if pixel_values_videos is not None:
                video_embeds, _ = qwen_model.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = qwen_model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    video_features=video_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        if attention_mask is None:
            prefix_pad_masks = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.bool)
        else:
            prefix_pad_masks = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        prefix_att_masks = torch.zeros_like(prefix_pad_masks, dtype=torch.bool)
        return inputs_embeds, prefix_pad_masks, prefix_att_masks

    def _prepare_stock_prefix(self, *, batch_images, instructions):
        qwen_inputs = self.policy.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict={
                "{actions}": self.policy.replace_prompt,
                "{e_actions}": self.policy.embodied_replace_prompt,
            },
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._prepare_qwen3_prefix_embeddings(
            self.policy.qwen_vl_interface,
            qwen_inputs,
        )
        return {
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_att_masks": prefix_att_masks,
            "input_ids": qwen_inputs["input_ids"],
            "attention_mask": qwen_inputs.get("attention_mask"),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
            "video_grid_thw": qwen_inputs.get("video_grid_thw"),
        }

    def _run_qwen_language_from_prefix(self, prefix: dict[str, Any]) -> torch.Tensor:
        required = ("prefix_embs", "input_ids", "attention_mask")
        missing = [key for key in required if key not in prefix or prefix.get(key) is None]
        if missing:
            raise RuntimeError(f"shared JEPA prefix is missing full-path fields: {missing}")

        qwen_model = self.policy.qwen_vl_interface.model.model
        input_ids = prefix["input_ids"]
        attention_mask = prefix.get("attention_mask")
        attention_mask_tensor = attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        if isinstance(attention_mask_tensor, torch.Tensor) and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        position_ids, rope_deltas = qwen_model.get_rope_index(
            input_ids,
            prefix.get("image_grid_thw"),
            prefix.get("video_grid_thw"),
            attention_mask=attention_mask_tensor,
        )
        if hasattr(qwen_model, "rope_deltas"):
            qwen_model.rope_deltas = rope_deltas

        language_model = qwen_model.language_model
        final_norm = getattr(language_model, "norm", None)
        norm_handle = None
        if final_norm is not None:
            norm_handle = final_norm.register_forward_hook(lambda _module, inputs, _output: inputs[0])
        try:
            outputs = language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                inputs_embeds=prefix["prefix_embs"],
                cache_position=None,
                visual_pos_masks=prefix.get("visual_pos_masks"),
                deepstack_visual_embeds=prefix.get("deepstack_visual_embeds"),
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            if norm_handle is not None:
                norm_handle.remove()
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            return hidden_states[-1]
        return outputs.last_hidden_state

    def _predict_full_shared_prefix(self, *, batch_images, instructions, state: np.ndarray):
        if self.triton_runtime is None:
            raise RuntimeError("shared prefix full path requires Triton runtime")
        if len(batch_images) != 1:
            raise ValueError("Sonic Triton full path currently expects batch size 1")

        self._sync_cuda()
        total_t0 = time.perf_counter()
        enc_t0 = time.perf_counter()
        with torch.inference_mode():
            prefix = self.triton_runtime.prepare_policy_prefix(batch_images=batch_images, instructions=instructions)
        prefix_runtime_timing = dict(prefix.pop("_runtime_timing", {}) or {})
        self._sync_cuda()
        encoder_ms = float((time.perf_counter() - enc_t0) * 1000.0)

        embodied_action_indices = torch.isin(
            prefix["input_ids"],
            torch.tensor([self.policy.embodied_action_token_id], device=prefix["input_ids"].device),
        ).nonzero(as_tuple=True)
        autocast_qwen = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        prefill_t0 = time.perf_counter()
        with torch.inference_mode(), autocast_qwen:
            last_hidden = self._run_qwen_language_from_prefix(prefix)
            batch, _, hidden = last_hidden.shape
            embodied_action_tokens = last_hidden[
                embodied_action_indices[0],
                embodied_action_indices[1],
                :,
            ].view(batch, -1, hidden)
        self._sync_cuda()
        prefill_ms = float((time.perf_counter() - prefill_t0) * 1000.0)

        denoise_t0 = time.perf_counter()
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32).reshape(len(batch_images), 1, self.state_dim)).to(
            last_hidden.device,
            dtype=last_hidden.dtype,
        )
        with torch.inference_mode():
            pred_actions = self.triton_runtime.predict_actions_from_embodied(
                embodied_action_tokens=embodied_action_tokens,
                state=state_t,
            )
        self._sync_cuda()
        denoise_ms = float((time.perf_counter() - denoise_t0) * 1000.0)

        output = {
            "normalized_actions": pred_actions.to(dtype=torch.float32).detach().cpu().numpy(),
            "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy(),
        }
        timing = {
            "encoder_ms": encoder_ms,
            "vlm_prefill_ms": prefill_ms,
            "full_fallback_ms": denoise_ms,
            "total_ms": float((time.perf_counter() - total_t0) * 1000.0),
            "jepa_shared_prefix_full": 1.0,
        }
        timing.update(prefix_runtime_timing)
        if hasattr(self.triton_runtime, "describe"):
            desc = self.triton_runtime.describe()
            timing["triton_runtime_visual_compiled"] = 1.0 if desc.get("compiled_image_features") else 0.0
            timing["triton_runtime_action_compiled"] = 1.0 if desc.get("compiled_action_predict") else 0.0
        return output, timing

    def _should_run_full_round(self) -> bool:
        if not self._has_draft_path():
            return True
        if self.full_cache_snapshot is None or self.pending_full_fallback or self.force_full_each_round:
            return True
        if self.full_rounds_served < self.draft_start_after_full_rounds:
            return True
        if self.periodic_full_every_n_draft_rounds > 0:
            return self.draft_rounds_since_full >= self.periodic_full_every_n_draft_rounds
        return False

    def _has_draft_path(self) -> bool:
        if not self.draft_enabled:
            return False
        if self.draft_head is not None:
            return True
        return bool(self.triton_runtime is not None and self.triton_runtime.has_draft_runtime)

    @staticmethod
    def _compute_radius_prefix_acceptance(
        *,
        x0_draft: torch.Tensor,
        x0_hat: torch.Tensor,
        tau_radius: float,
        dist_dims: int,
        eval_h: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eval_h = int(min(int(x0_draft.shape[1]), max(1, int(eval_h))))
        eval_d = int(min(int(x0_draft.shape[2]), max(1, int(dist_dims))))
        diff = x0_hat[:, :, :eval_h, :eval_d].to(torch.float32) - x0_draft[:, None, :eval_h, :eval_d].to(torch.float32)
        norm_d = torch.tensor(float(eval_d), device=x0_draft.device, dtype=torch.float32).sqrt().clamp_min(1.0)
        dist = torch.linalg.vector_norm(diff, ord=2, dim=3).to(dtype=torch.float32) / norm_d
        ok = dist <= float(tau_radius)
        prefix_len_k = ok.to(dtype=torch.int64).cumprod(dim=2).sum(dim=2)
        return prefix_len_k.min(dim=1).values.to(dtype=torch.int64), dist

    @staticmethod
    def _stitch_radius_prefix_output(
        *,
        x0_draft: torch.Tensor,
        x0_tail: torch.Tensor,
        accepted_prefix_len: torch.Tensor,
    ) -> torch.Tensor:
        accepted_prefix_len = accepted_prefix_len.to(device=x0_draft.device, dtype=torch.int64)
        idx = torch.arange(int(x0_draft.shape[1]), device=x0_draft.device, dtype=torch.int64)[None, :]
        accept_mask = (idx < accepted_prefix_len[:, None])[:, :, None]
        return torch.where(accept_mask, x0_draft, x0_tail)

    def _run_verify_with_timing(
        self,
        *,
        embodied_action_tokens: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        action_model = self.policy.action_model
        model_dtype = next(action_model.parameters()).dtype
        embodied = embodied_action_tokens.to(self.device, dtype=model_dtype)
        state = state.to(self.device, dtype=model_dtype)
        draft = x0_draft.to(self.device, dtype=model_dtype)
        noise = noise.to(self.device, dtype=model_dtype)
        batch, horizon, _ = draft.shape

        self._sync_cuda()
        t0 = time.perf_counter()
        with torch.inference_mode():
            if self.triton_runtime is not None and hasattr(self.triton_runtime, "predict_verify"):
                x0_hat = self.triton_runtime.predict_verify(
                    embodied_action_tokens=embodied,
                    state=state,
                    noise=noise,
                    x0_draft=draft,
                )
            else:
                state_features = action_model.state_encoder(state) if getattr(action_model, "state_encoder", None) else None
                x0_hat_steps: list[torch.Tensor] = []
                for t_value in self.t_list:
                    t_float = float(max(0.0, min(1.0, float(t_value))))
                    noisy_trajectory = (1.0 - t_float) * noise + t_float * draft
                    t_discretized = int(t_float * int(action_model.num_timestep_buckets))
                    t_discretized = max(0, min(t_discretized, int(action_model.num_timestep_buckets) - 1))
                    timesteps = torch.full((batch,), t_discretized, device=self.device, dtype=torch.long)
                    action_features = action_model.action_encoder(noisy_trajectory, timesteps)
                    if getattr(action_model.config, "add_pos_embed", False):
                        pos_ids = torch.arange(horizon, dtype=torch.long, device=self.device)
                        action_features = action_features + action_model.position_embedding(pos_ids).unsqueeze(0)
                    future_tokens = action_model.future_tokens.weight.unsqueeze(0).expand(batch, -1, -1)
                    hidden_states = (
                        torch.cat((state_features, future_tokens, action_features), dim=1)
                        if state_features is not None
                        else torch.cat((future_tokens, action_features), dim=1)
                    )
                    model_output = action_model.model(
                        hidden_states=hidden_states,
                        encoder_hidden_states=embodied,
                        timestep=timesteps,
                    )
                    pred = action_model.action_decoder(model_output)
                    pred_velocity = pred[:, -horizon:]
                    x0_hat_steps.append((noisy_trajectory + (1.0 - t_float) * pred_velocity).to(torch.float32))
                x0_hat = torch.stack(x0_hat_steps, dim=1)
        self._sync_cuda()
        timing = {"action_verify_ms": float((time.perf_counter() - t0) * 1000.0)}
        if self.triton_runtime is not None and hasattr(self.triton_runtime, "describe"):
            timing["triton_runtime_verify_compiled"] = (
                1.0 if bool(self.triton_runtime.describe().get("compiled_action_verify", False)) else 0.0
            )
        return x0_hat, timing

    def _prepare_batch(self, examples):
        if not isinstance(examples, list) or not examples:
            raise ValueError("Expected non-empty `examples` list.")

        batch_images = []
        instructions = []
        states = []
        for example in examples:
            batch_images.append(_normalize_image_list(example["image"]))
            instructions.append(example["lang"])
            if "state" in example and example["state"] is not None:
                states.append(np.asarray(example["state"], dtype=np.float32))
        state = np.asarray(states, dtype=np.float32) if states else None
        if state is not None and state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state_dim={self.state_dim}, got state shape={state.shape}")
        return batch_images, instructions, state

    def _predict_full(self, examples, **kwargs):
        batch_images, instructions, state = self._prepare_batch(examples)
        self._sync_cuda()
        t0 = time.perf_counter()
        runtime_timing: dict[str, float] = {}
        if self.backend == "triton" and self.triton_runtime is not None and self.enable_shared_prefix_full:
            try:
                output, runtime_timing = self._predict_full_shared_prefix(
                    batch_images=batch_images,
                    instructions=instructions,
                    state=state,
                )
            except Exception:
                logging.exception("Sonic Triton full path failed; falling back to PyTorch predict_action")
                output = self.policy.predict_action(batch_images=batch_images, instructions=instructions, state=state, **kwargs)
        else:
            output = self.policy.predict_action(batch_images=batch_images, instructions=instructions, state=state, **kwargs)
        self._sync_cuda()
        total_ms = float((time.perf_counter() - t0) * 1000.0)
        embodied = np.asarray(output.get("embodied_action_tokens"), dtype=np.float32)
        if embodied.ndim == 3:
            self.full_cache_snapshot = {
                "embodied_action_tokens": torch.from_numpy(embodied).to(self.device, dtype=torch.float32),
                "state": torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device, dtype=torch.float32),
            }
            self.pending_full_fallback = False
            self.draft_rounds_since_full = 0
            self.full_rounds_served += 1
        output["policy_timing"] = {
            "route_type": "full",
            "sample_actions_ms": total_ms,
            "total_ms": total_ms,
            "encoder_ms": 0.0,
            "vlm_prefill_ms": total_ms,
            "draft_ms": 0.0,
            "action_verify_ms": 0.0,
            "full_fallback_ms": total_ms,
            "did_prefill": 1.0,
            "is_full_pipeline_round": 1.0,
            "used_full_fallback": 1.0,
            "accepted_prefix_len": float(min(int(output["normalized_actions"].shape[1]), self.max_exec_steps)),
            "accepted_prefix_len_mean": float(min(int(output["normalized_actions"].shape[1]), self.max_exec_steps)),
            "include_in_draft_accept_metrics": 0.0,
            "backend_triton": 1.0 if self.backend == "triton" else 0.0,
            "full_rounds_served": float(self.full_rounds_served),
        }
        output["policy_timing"].update(runtime_timing)
        output["policy_timing"]["sample_actions_ms"] = total_ms
        output["policy_timing"]["total_ms"] = total_ms
        return output

    def _predict_draft(self, examples, **kwargs):
        del kwargs
        if not self._has_draft_path() or self.full_cache_snapshot is None:
            return self._predict_full(examples)
        batch_images, instructions, state = self._prepare_batch(examples)
        if len(batch_images) != 1:
            raise ValueError("Sonic draft inference currently expects batch size 1")
        state_np = np.asarray(state, dtype=np.float32)
        state_flat = state_np.reshape(1, -1)
        state_3d = state_np.reshape(1, 1, -1)

        total_t0 = time.perf_counter()
        prefix_t0 = time.perf_counter()
        with torch.inference_mode():
            if self.triton_runtime is not None:
                prefix = self.triton_runtime.prepare_draft_prefix(batch_images=batch_images, instructions=instructions)
            else:
                prefix = self._prepare_stock_prefix(batch_images=batch_images, instructions=instructions)
            prefix_runtime_timing = dict(prefix.pop("_runtime_timing", {}) or {})
        self._sync_cuda()
        prefix_ms = float((time.perf_counter() - prefix_t0) * 1000.0)

        draft_t0 = time.perf_counter()
        with torch.inference_mode():
            chunk_m = (
                int(self.draft_head.chunk_m)
                if self.draft_head is not None
                else int(self.triton_runtime.draft_chunk_m or self.max_exec_steps)
            )
            last_actions = torch.zeros((1, chunk_m, self.action_dim), device=self.device, dtype=torch.float32)
            if self.triton_runtime is not None:
                x0_draft = self.triton_runtime.predict_draft(
                    prefix_embs=prefix["prefix_embs"].to(self.device),
                    prefix_pad_masks=prefix["prefix_pad_masks"].to(self.device),
                    prefix_att_masks=prefix["prefix_att_masks"].to(self.device),
                    robot_state=torch.from_numpy(state_flat).to(self.device, dtype=torch.float32),
                    last_actions=last_actions,
                )
            else:
                x0_draft = self.draft_head(
                    prefix_embs=prefix["prefix_embs"].to(self.device),
                    prefix_pad_masks=prefix["prefix_pad_masks"].to(self.device),
                    prefix_att_masks=prefix["prefix_att_masks"].to(self.device),
                    robot_state=torch.from_numpy(state_flat).to(self.device, dtype=torch.float32),
                    last_actions=last_actions,
                )
            x0_draft = x0_draft.to(self.device, dtype=torch.float32).clone()
        self._sync_cuda()
        draft_ms = float((time.perf_counter() - draft_t0) * 1000.0)

        noise = torch.randn_like(x0_draft)
        x0_hat, verify_timing = self._run_verify_with_timing(
            embodied_action_tokens=self.full_cache_snapshot["embodied_action_tokens"],
            state=torch.from_numpy(state_3d),
            noise=noise,
            x0_draft=x0_draft,
        )
        accepted_prefix_len, dist = self._compute_radius_prefix_acceptance(
            x0_draft=x0_draft,
            x0_hat=x0_hat,
            tau_radius=self.tau_radius,
            dist_dims=self.verify_dist_dims,
            eval_h=self.max_exec_steps,
        )
        raw_accepted_prefix_len = accepted_prefix_len.clone()
        rejected_by_min_accept = bool(
            self.draft_min_accept_steps > 0
            and (accepted_prefix_len < self.draft_min_accept_steps).any().item()
        )
        if rejected_by_min_accept:
            accepted_prefix_len = torch.zeros_like(accepted_prefix_len)
        should_schedule_full = bool((accepted_prefix_len <= 0).any().item()) and self.draft_full_fallback
        self.pending_full_fallback = should_schedule_full
        self.draft_rounds_since_full += 1
        x0_tail = x0_hat.mean(dim=1)
        x0_out = self._stitch_radius_prefix_output(
            x0_draft=x0_draft,
            x0_tail=x0_tail,
            accepted_prefix_len=accepted_prefix_len,
        )
        accepted = int(min(int(accepted_prefix_len.to(torch.float32).mean().round().item()), self.max_exec_steps))
        normalized = x0_out.detach().cpu().numpy().astype(np.float32)
        total_ms = float((time.perf_counter() - total_t0) * 1000.0)
        result = {
            "normalized_actions": normalized,
            "policy_timing": {
                "route_type": "draft",
                "sample_actions_ms": total_ms,
                "total_ms": total_ms,
                "encoder_ms": float(prefix_runtime_timing.get("triton_runtime_prefix_ms", prefix_ms)),
                "vlm_prefill_ms": 0.0,
                "draft_ms": draft_ms,
                "action_verify_ms": float(verify_timing.get("action_verify_ms", 0.0)),
                "full_fallback_ms": 0.0,
                "did_prefill": 0.0,
                "is_full_pipeline_round": 0.0,
                "used_full_fallback": 0.0,
                "accepted_prefix_len": float(accepted),
                "accepted_prefix_len_mean": float(accepted),
                "raw_accepted_prefix_len": float(
                    min(
                        int(raw_accepted_prefix_len.to(torch.float32).mean().round().item()),
                        self.max_exec_steps,
                    )
                ),
                "draft_rejected_by_min_accept": 1.0 if rejected_by_min_accept else 0.0,
                "draft_min_accept_steps": float(self.draft_min_accept_steps),
                "scheduled_full_fallback": 1.0 if should_schedule_full else 0.0,
                "radius_dist": float(dist.mean().item()),
                "radius_dist_mean": float(dist.mean().item()),
                "radius_dist_max": float(dist.max().item()),
                "dist_dims_effective": float(min(self.action_dim, max(1, int(self.verify_dist_dims)))),
                "include_in_draft_accept_metrics": 1.0,
                "backend_triton": 1.0 if self.backend == "triton" else 0.0,
                "full_rounds_served": float(self.full_rounds_served),
            },
        }
        result["policy_timing"].update(verify_timing)
        result["policy_timing"].update(prefix_runtime_timing)
        result["policy_timing"]["encoder_ms"] = float(prefix_runtime_timing.get("triton_runtime_prefix_ms", prefix_ms))
        if self.triton_runtime is not None and hasattr(self.triton_runtime, "describe"):
            desc = self.triton_runtime.describe()
            result["policy_timing"]["triton_runtime_draft_compiled"] = (
                1.0 if bool(desc.get("compiled_draft_forward", False)) else 0.0
            )
            result["policy_timing"]["triton_runtime_draft_source"] = str(desc.get("draft_runtime_source", "unknown"))
        return result

    def predict_action(self, examples, **kwargs):
        if self._should_run_full_round():
            return self._predict_full(examples, **kwargs)
        try:
            return self._predict_draft(examples, **kwargs)
        except Exception:
            if not self.draft_full_fallback:
                raise
            logging.exception("Sonic draft path failed; falling back to full VLA-JEPA path")
            output = self._predict_full(examples, **kwargs)
            output.setdefault("policy_timing", {})["route_type"] = "full_fallback"
            return output

    def warmup(
        self,
        *,
        runs: int,
        image_path: str | None,
        prompt: str,
        target_ms: float = 0.0,
        max_runs: int = 0,
    ) -> None:
        min_runs = int(runs)
        target_ms = float(target_ms)
        max_runs = int(max_runs)
        if min_runs <= 0 and target_ms <= 0.0:
            return
        min_runs = max(0, min_runs)
        max_runs = max(min_runs, max_runs if max_runs > 0 else min_runs)
        if image_path:
            image = Image.open(Path(image_path).expanduser()).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        state = np.zeros((1, self.state_dim), dtype=np.float32)
        examples = [{"image": [image], "lang": str(prompt), "state": state}]

        saved_cache = _clone_cache_snapshot(self.full_cache_snapshot)
        saved_pending_full_fallback = bool(self.pending_full_fallback)
        saved_draft_rounds_since_full = int(self.draft_rounds_since_full)
        saved_full_rounds_served = int(self.full_rounds_served)
        logging.info(
            "Starting Sonic policy warmup: min_runs=%d max_runs=%d target_ms=%.3f backend=%s draft=%s image=%s",
            min_runs,
            max_runs,
            target_ms,
            self.backend,
            "enabled" if self._has_draft_path() else "disabled",
            image_path or "<black>",
        )
        try:
            completed = 0
            while completed < max_runs:
                t0 = time.perf_counter()
                output = self.predict_action(examples)
                self._sync_cuda()
                completed += 1
                elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
                timing = dict(output.get("policy_timing", {}) or {})
                total_ms = float(timing.get("total_ms", elapsed_ms))
                logging.info(
                    "Sonic policy warmup %d/%d route=%s total_ms=%.3f elapsed_ms=%.3f full_compiled=%s draft_compiled=%s verify_compiled=%s",
                    completed,
                    max_runs,
                    timing.get("route_type", "unknown"),
                    total_ms,
                    elapsed_ms,
                    timing.get("triton_runtime_action_compiled", "n/a"),
                    timing.get("triton_runtime_draft_compiled", "n/a"),
                    timing.get("triton_runtime_verify_compiled", "n/a"),
                )
                if completed >= min_runs and target_ms > 0.0 and total_ms <= target_ms:
                    logging.info(
                        "Sonic policy warmup reached target: total_ms=%.3f <= %.3f after %d runs",
                        total_ms,
                        target_ms,
                        completed,
                    )
                    break
        finally:
            self.full_cache_snapshot = saved_cache
            self.pending_full_fallback = saved_pending_full_fallback
            self.draft_rounds_since_full = saved_draft_rounds_since_full
            self.full_rounds_served = saved_full_rounds_served
        logging.info("Sonic policy warmup complete; restored runtime cache state.")


def main(args) -> None:
    device = args.device or "cuda"
    vla = baseframework.from_pretrained(args.ckpt_path)

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to(device).eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating VLA_JEPA server (host: %s, ip: %s)", hostname, local_ip)

    policy_adapter = VLAJEPAPolicyAdapter(
        vla,
        device=device,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        backend=args.backend,
        triton_manifest=args.triton_manifest,
        max_exec_steps=args.max_exec_steps,
        draft_checkpoint=args.draft_checkpoint,
        tau_radius=args.tau_radius,
        verify_dist_dims=args.verify_dist_dims,
        t_list=args.t_list,
        draft_full_fallback=not args.no_draft_full_fallback,
        force_full_each_round=args.force_full_each_round,
        periodic_full_every_n_draft_rounds=args.periodic_full_every_n_draft_rounds,
        draft_start_after_full_rounds=args.draft_start_after_full_rounds,
        draft_min_accept_steps=args.draft_min_accept_steps,
        enable_shared_prefix_full=not args.disable_shared_prefix_full,
    )
    policy_adapter.warmup(
        runs=args.warmup_runs,
        image_path=args.warmup_image,
        prompt=args.warmup_prompt,
        target_ms=args.warmup_target_ms,
        max_runs=args.warmup_max_runs,
    )

    server = WebsocketPolicyServer(
        policy=policy_adapter,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={
            "env": "sonicstar",
            "framework": "VLA_JEPA",
            "action_dim": int(args.action_dim),
            "state_dim": int(args.state_dim),
            "draft_checkpoint": args.draft_checkpoint,
            "backend": args.backend,
            "triton_manifest": args.triton_manifest,
        },
    )
    logging.info("VLA_JEPA server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800)
    parser.add_argument("--backend", type=str, choices=["pytorch", "triton"], default="pytorch")
    parser.add_argument("--triton_manifest", type=str, default=None)
    parser.add_argument("--draft_checkpoint", type=str, default=None)
    parser.add_argument("--max_exec_steps", type=int, default=20)
    parser.add_argument("--tau_radius", type=float, default=0.25)
    parser.add_argument("--verify_dist_dims", type=int, default=64)
    parser.add_argument("--t_list", type=float, nargs="+", default=[0.1, 0.05])
    parser.add_argument("--action_dim", type=int, default=78)
    parser.add_argument("--state_dim", type=int, default=46)
    parser.add_argument("--no_draft_full_fallback", action="store_true")
    parser.add_argument("--force_full_each_round", action="store_true")
    parser.add_argument("--periodic_full_every_n_draft_rounds", type=int, default=1)
    parser.add_argument(
        "--draft_start_after_full_rounds",
        type=int,
        default=int(os.getenv("SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS", "0")),
    )
    parser.add_argument(
        "--draft_min_accept_steps",
        type=int,
        default=int(os.getenv("SONICSTAR_DRAFT_MIN_ACCEPT_STEPS", "1")),
    )
    parser.add_argument("--disable_shared_prefix_full", action="store_true")
    parser.add_argument("--warmup_runs", type=int, default=int(os.getenv("SONICSTAR_POLICY_WARMUP_RUNS", "0")))
    parser.add_argument(
        "--warmup_target_ms",
        type=float,
        default=float(os.getenv("SONICSTAR_POLICY_WARMUP_TARGET_MS", "0")),
    )
    parser.add_argument(
        "--warmup_max_runs",
        type=int,
        default=int(os.getenv("SONICSTAR_POLICY_WARMUP_MAX_RUNS", "0")),
    )
    parser.add_argument("--warmup_image", type=str, default=os.getenv("SONICSTAR_POLICY_WARMUP_IMAGE", ""))
    parser.add_argument(
        "--warmup_prompt",
        type=str,
        default=os.getenv("SONICSTAR_POLICY_WARMUP_PROMPT", "pick up the cylinder and throw it into the trash bin"),
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("DEBUG is enabled", flush=True)
    main(args)
