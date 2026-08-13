from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


class _DraftForward(torch.nn.Module):
    def __init__(self, draft_head: torch.nn.Module) -> None:
        super().__init__()
        self.draft_head = draft_head
        self.chunk_m = int(getattr(draft_head, "chunk_m", 0))

    def forward(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        robot_state: torch.Tensor,
        last_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.draft_head(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            robot_state=robot_state,
            last_actions=last_actions,
        )


class _ActionPredict(torch.nn.Module):
    def __init__(self, action_model: torch.nn.Module) -> None:
        super().__init__()
        self.action_model = action_model

    def forward(self, embodied_action_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.action_model.predict_action(embodied_action_tokens, state)


class _ActionVerify(torch.nn.Module):
    def __init__(self, action_model: torch.nn.Module, t_list: tuple[float, ...]) -> None:
        super().__init__()
        self.action_model = action_model
        buckets = int(getattr(action_model, "num_timestep_buckets"))
        self.t_values = tuple(float(max(0.0, min(1.0, float(t)))) for t in t_list)
        self.timestep_values = tuple(max(0, min(int(t * buckets), buckets - 1)) for t in self.t_values)

    def forward(
        self,
        embodied_action_tokens: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
    ) -> torch.Tensor:
        action_model = self.action_model
        batch, horizon, _ = x0_draft.shape
        state_features = (
            action_model.state_encoder(state)
            if getattr(action_model, "state_encoder", None) is not None
            else None
        )
        x0_hat_steps: list[torch.Tensor] = []
        for t_float, t_discretized in zip(self.t_values, self.timestep_values):
            noisy_trajectory = (1.0 - t_float) * noise + t_float * x0_draft
            timesteps = torch.full(
                (batch,),
                int(t_discretized),
                device=x0_draft.device,
                dtype=torch.long,
            )

            action_features = action_model.action_encoder(noisy_trajectory, timesteps)
            if getattr(action_model.config, "add_pos_embed", False):
                pos_ids = torch.arange(horizon, dtype=torch.long, device=x0_draft.device)
                action_features = action_features + action_model.position_embedding(pos_ids).unsqueeze(0)

            future_tokens = action_model.future_tokens.weight.unsqueeze(0).expand(batch, -1, -1)
            if state_features is not None:
                hidden_states = torch.cat((state_features, future_tokens, action_features), dim=1)
            else:
                hidden_states = torch.cat((future_tokens, action_features), dim=1)

            model_output = action_model.model(
                hidden_states=hidden_states,
                encoder_hidden_states=embodied_action_tokens,
                timestep=timesteps,
            )
            pred = action_model.action_decoder(model_output)
            pred_velocity = pred[:, -horizon:]
            x0_hat_steps.append(noisy_trajectory + (1.0 - t_float) * pred_velocity)
        return torch.stack(x0_hat_steps, dim=1).to(torch.float32)


class _QwenImageFeaturesForward(torch.nn.Module):
    def __init__(self, qwen_model: torch.nn.Module) -> None:
        super().__init__()
        self.qwen_model = qwen_model

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> tuple[torch.Tensor, Any]:
        image_embeds, deepstack_image_embeds = self.qwen_model.get_image_features(pixel_values, image_grid_thw)
        if not isinstance(image_embeds, torch.Tensor):
            image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds, deepstack_image_embeds


def _sync_if_cuda(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class JEPAPrefixRuntime:
    """Prepare JEPA draft prefix with cached prompt tokens and image-only processing.

    The stock VLA-JEPA path rebuilds the full Qwen chat template every policy call.
    For SIMPLE evaluation the instruction, action placeholders, image count, and
    resized image geometry are stable, so only pixel values need to change.
    """

    def __init__(self, *, model: torch.nn.Module, device: str | torch.device) -> None:
        self.model = model
        self.device = torch.device(device)
        self._cache: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
        self._image_features_forward: torch.nn.Module | None = None
        self._image_features_compiled = False
        self._image_features_warned = False
        self._process_images_on_device = os.environ.get("JEPA_PREFIX_PROCESS_IMAGE_ON_DEVICE", "1") != "0"
        self._skip_processor_resize = os.environ.get("JEPA_PREFIX_SKIP_PROCESSOR_RESIZE", "1") != "0"
        self._disable_processor_grouping = os.environ.get("JEPA_PREFIX_DISABLE_PROCESSOR_GROUPING", "1") != "0"
        self._exact_placeholder_scatter = os.environ.get("JEPA_PREFIX_EXACT_PLACEHOLDER_SCATTER", "1") != "0"

    def set_image_features_forward(self, module: torch.nn.Module | None, *, compiled: bool) -> None:
        self._image_features_forward = module
        self._image_features_compiled = bool(compiled)

    @staticmethod
    def _clone_static(batch_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        static: dict[str, torch.Tensor] = {}
        for key in ("input_ids", "attention_mask", "image_grid_thw", "video_grid_thw"):
            value = batch_inputs.get(key)
            if isinstance(value, torch.Tensor):
                static[key] = value.detach().clone()
        return static

    @staticmethod
    def _as_device_dict(data: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in data.items():
            out[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        return out

    def _resize_images(self, batch_images: list[list[Image.Image]]) -> list[list[Image.Image]]:
        train_obs_image_size = getattr(self.model.config.datasets.vla_data, "image_size", None)
        if not train_obs_image_size:
            return batch_images
        target = tuple(int(x) for x in train_obs_image_size)
        if len(target) != 2:
            return batch_images

        def _resize_one(img: Image.Image) -> Image.Image:
            return img if tuple(img.size) == target else img.resize(target)

        return [[_resize_one(img) for img in sample] for sample in batch_images]

    def _cache_key(self, *, batch_images: list[list[Image.Image]], instructions: list[str]) -> tuple[Any, ...]:
        image_sizes = tuple(tuple((int(img.size[0]), int(img.size[1])) for img in sample) for sample in batch_images)
        return (
            tuple(str(x) for x in instructions),
            tuple(len(sample) for sample in batch_images),
            image_sizes,
            str(getattr(self.model, "replace_prompt", "")),
            str(getattr(self.model, "embodied_replace_prompt", "")),
        )

    def _build_static_cache(
        self,
        *,
        batch_images: list[list[Image.Image]],
        instructions: list[str],
    ) -> dict[str, torch.Tensor]:
        qwen_inputs = self.model.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict={
                "{actions}": self.model.replace_prompt,
                "{e_actions}": self.model.embodied_replace_prompt,
            },
        )
        static = self._clone_static(dict(qwen_inputs))
        if isinstance(static.get("image_grid_thw"), torch.Tensor):
            static["image_grid_thw_cpu"] = static["image_grid_thw"].detach().cpu().clone()
        qwen_model = self.model.qwen_vl_interface.model.model
        input_ids = qwen_inputs["input_ids"]
        device_type = "cuda" if input_ids.device.type == "cuda" else "cpu"
        with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
            image_token_id = int(getattr(qwen_model.config, "image_token_id", 151655))
            visual_pos_mask = input_ids.eq(image_token_id)
            if bool(visual_pos_mask.any().item()):
                batch_idx, seq_idx = visual_pos_mask.nonzero(as_tuple=True)
                static["image_token_batch_idx"] = batch_idx.detach().clone()
                static["image_token_seq_idx"] = seq_idx.detach().clone()
                static["visual_pos_mask"] = visual_pos_mask.detach().clone()
        static["inputs_embeds_template"] = inputs_embeds.detach().clone()
        return static

    @staticmethod
    def _can_skip_processor_resize(
        *,
        batch_images: list[list[Image.Image]],
        image_processor: Any,
        static: dict[str, torch.Tensor],
    ) -> bool:
        static_grid = static.get("image_grid_thw_cpu", static.get("image_grid_thw"))
        if not isinstance(static_grid, torch.Tensor):
            return False
        patch_size = int(getattr(image_processor, "patch_size", 0) or 0)
        merge_size = int(getattr(image_processor, "merge_size", 0) or 0)
        temporal_patch_size = int(getattr(image_processor, "temporal_patch_size", 1) or 1)
        if patch_size <= 0 or merge_size <= 0 or temporal_patch_size <= 0:
            return False
        expected = []
        for sample in batch_images:
            for img in sample:
                width, height = int(img.size[0]), int(img.size[1])
                if height % (patch_size * merge_size) != 0 or width % (patch_size * merge_size) != 0:
                    return False
                expected.append([1, height // patch_size, width // patch_size])
        if len(expected) != int(static_grid.shape[0]):
            return False
        expected_t = torch.as_tensor(expected, dtype=static_grid.dtype)
        return bool(torch.equal(static_grid, expected_t))

    def _build_fast_inputs(
        self,
        *,
        batch_images: list[list[Image.Image]],
        static: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        if len(batch_images) != 1:
            raise ValueError("fast JEPA prefix runtime currently supports batch size 1")
        image_processor = self.model.qwen_vl_interface.processor.image_processor
        qwen_device = torch.device(self.model.qwen_vl_interface.model.device)
        processor_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if self._process_images_on_device:
            processor_kwargs["device"] = qwen_device
        if self._disable_processor_grouping:
            processor_kwargs["disable_grouping"] = True
        skip_resize = bool(self._skip_processor_resize) and self._can_skip_processor_resize(
            batch_images=batch_images,
            image_processor=image_processor,
            static=static,
        )
        if skip_resize:
            processor_kwargs["do_resize"] = False

        processor_t0 = time.perf_counter()
        visual_inputs = image_processor(images=batch_images[0], **processor_kwargs)
        processor_ms = (time.perf_counter() - processor_t0) * 1000.0

        h2d_t0 = time.perf_counter()
        pixel_values = visual_inputs["pixel_values"]
        if isinstance(pixel_values, torch.Tensor) and pixel_values.device != qwen_device:
            pixel_values = pixel_values.to(qwen_device, non_blocking=True)
        fast_inputs: dict[str, Any] = {"pixel_values": pixel_values}
        image_grid_thw = visual_inputs.get("image_grid_thw")
        if isinstance(image_grid_thw, torch.Tensor):
            static_grid = static.get("image_grid_thw")
            if isinstance(static_grid, torch.Tensor) and tuple(static_grid.shape) == tuple(image_grid_thw.shape):
                fast_inputs["image_grid_thw"] = static_grid
            elif static_grid is None:
                fast_inputs["image_grid_thw"] = image_grid_thw.to(qwen_device, non_blocking=True)
            else:
                raise ValueError(
                    "image_grid_thw changed; cached prompt token layout is no longer valid "
                    f"static={tuple(static_grid.shape)} current={tuple(image_grid_thw.shape)}"
                )
        h2d_ms = (time.perf_counter() - h2d_t0) * 1000.0
        fast_inputs["_runtime_timing"] = {
            "prefix_processor_ms": float(processor_ms),
            "prefix_h2d_ms": float(h2d_ms),
            "prefix_skip_resize": 1.0 if skip_resize else 0.0,
            "prefix_process_on_device": 1.0 if self._process_images_on_device else 0.0,
        }
        return fast_inputs

    def _get_image_features(
        self,
        qwen_model: torch.nn.Module,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        if self._image_features_forward is not None:
            try:
                return self._image_features_forward(pixel_values, image_grid_thw)
            except Exception as exc:
                if not self._image_features_warned:
                    logging.warning(
                        "Compiled Qwen visual feature path failed; falling back to eager get_image_features: %s",
                        exc,
                    )
                    self._image_features_warned = True
                self._image_features_forward = None
                self._image_features_compiled = False
        image_embeds, deepstack_image_embeds = qwen_model.get_image_features(pixel_values, image_grid_thw)
        if not isinstance(image_embeds, torch.Tensor):
            image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds, deepstack_image_embeds

    def _prepare_prefix_from_visual_inputs(
        self,
        *,
        visual_inputs: dict[str, Any],
        static: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        qwen_model = self.model.qwen_vl_interface.model.model
        qwen_device = self.model.qwen_vl_interface.model.device
        runtime_timing = dict(visual_inputs.pop("_runtime_timing", {}) or {})
        visual_inputs = self._as_device_dict(dict(visual_inputs), device=qwen_device)
        inputs_embeds = static["inputs_embeds_template"].clone()
        input_ids = static["input_ids"]
        pixel_values = visual_inputs.get("pixel_values", None)
        image_grid_thw = visual_inputs.get("image_grid_thw", static.get("image_grid_thw"))

        device_type = "cuda" if input_ids.device.type == "cuda" else "cpu"
        visual_pos_masks = None
        deepstack_visual_embeds = None
        with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            if pixel_values is not None:
                visual_t0 = time.perf_counter()
                image_embeds, deepstack_image_embeds = self._get_image_features(qwen_model, pixel_values, image_grid_thw)
                runtime_timing["prefix_visual_ms"] = float((time.perf_counter() - visual_t0) * 1000.0)
                scatter_t0 = time.perf_counter()
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                if bool(self._exact_placeholder_scatter):
                    image_mask, _ = qwen_model.get_placeholder_mask(
                        input_ids,
                        inputs_embeds=inputs_embeds,
                        image_features=image_embeds,
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                    visual_pos_masks = image_mask[..., 0]
                else:
                    image_batch_idx = static.get("image_token_batch_idx")
                    image_seq_idx = static.get("image_token_seq_idx")
                    if (
                        isinstance(image_batch_idx, torch.Tensor)
                        and isinstance(image_seq_idx, torch.Tensor)
                        and int(image_embeds.shape[0]) == int(image_batch_idx.numel())
                    ):
                        inputs_embeds[image_batch_idx, image_seq_idx, :] = image_embeds
                        visual_pos_masks = static.get("visual_pos_mask")
                    else:
                        image_mask, _ = qwen_model.get_placeholder_mask(
                            input_ids,
                            inputs_embeds=inputs_embeds,
                            image_features=image_embeds,
                        )
                        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                        visual_pos_masks = image_mask[..., 0]
                deepstack_visual_embeds = deepstack_image_embeds
                runtime_timing["prefix_scatter_ms"] = float((time.perf_counter() - scatter_t0) * 1000.0)

        attention_mask = static.get("attention_mask")
        if attention_mask is None:
            prefix_pad_masks = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.bool)
        else:
            prefix_pad_masks = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        prefix_att_masks = torch.zeros_like(prefix_pad_masks, dtype=torch.bool)
        return {
            "prefix_embs": inputs_embeds,
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_att_masks": prefix_att_masks,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": static.get("video_grid_thw"),
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": deepstack_visual_embeds,
            "_runtime_timing": runtime_timing,
        }

    def prepare_policy_prefix(
        self,
        *,
        batch_images: list[list[Image.Image]],
        instructions: list[str],
    ) -> dict[str, Any]:
        batch_images = self._resize_images(batch_images)
        cache_key = self._cache_key(batch_images=batch_images, instructions=instructions)
        cache_hit = cache_key in self._cache

        _sync_if_cuda(self.device)
        total_t0 = time.perf_counter()
        static_ms = 0.0
        image_t0 = time.perf_counter()
        if cache_hit:
            static = self._cache[cache_key]
        else:
            static_build_t0 = time.perf_counter()
            static = self._build_static_cache(batch_images=batch_images, instructions=instructions)
            self._cache[cache_key] = static
            static_ms = (time.perf_counter() - static_build_t0) * 1000.0
        qwen_inputs = self._build_fast_inputs(batch_images=batch_images, static=static)
        _sync_if_cuda(self.device)
        image_ms = (time.perf_counter() - image_t0) * 1000.0 - static_ms

        embed_t0 = time.perf_counter()
        prefix = self._prepare_prefix_from_visual_inputs(
            visual_inputs=qwen_inputs,
            static=static,
        )
        stage_timing = dict(prefix.pop("_runtime_timing", {}) or {})
        _sync_if_cuda(self.device)
        embed_ms = (time.perf_counter() - embed_t0) * 1000.0
        total_ms = (time.perf_counter() - total_t0) * 1000.0
        runtime_timing = {
            "triton_runtime_prefix_ms": float(total_ms),
            "prefix_static_ms": float(static_ms),
            "prefix_image_ms": float(max(0.0, image_ms)),
            "prefix_embed_ms": float(embed_ms),
            "prefix_cache_hit": 1.0 if cache_hit else 0.0,
            "prefix_visual_compiled": 1.0 if self._image_features_compiled else 0.0,
        }
        runtime_timing.update(stage_timing)
        prefix["_runtime_timing"] = runtime_timing
        return prefix

    def prepare_draft_prefix(
        self,
        *,
        batch_images: list[list[Image.Image]],
        instructions: list[str],
    ) -> dict[str, Any]:
        return self.prepare_policy_prefix(batch_images=batch_images, instructions=instructions)

    def reset_runtime_state(self) -> None:
        self._cache.clear()


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_float * torch.rsqrt(variance + float(eps))
    return (weight.to(device=x.device, dtype=torch.float32) * x_norm).to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = int(x.shape[-1] // 2)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if int(n_rep) == 1:
        return x
    batch, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, int(n_rep), seq_len, head_dim)
    return x.reshape(batch, kv_heads * int(n_rep), seq_len, head_dim)


class JEPADraftArtifactRuntime(torch.nn.Module):
    """Draft runtime backed directly by `draft_triton.pkl` tensors."""

    REQUIRED_KEYS = {
        "draft_state_in_proj_w",
        "draft_state_in_proj_b",
        "draft_action_queries",
        "draft_qkv_w",
        "draft_attn_o_w",
        "draft_ffn_gate_w",
        "draft_ffn_up_w",
        "draft_ffn_down_w",
        "draft_input_layernorm_w",
        "draft_post_attention_layernorm_w",
        "draft_action_head_w",
        "draft_action_head_b",
    }

    def __init__(self, artifact: dict[str, Any], *, device: str | torch.device) -> None:
        super().__init__()
        meta = dict(artifact.get("meta", {}) or {})
        missing = sorted(self.REQUIRED_KEYS.difference(artifact))
        if missing:
            raise KeyError(f"draft artifact missing required tensors: {missing}")
        if str(meta.get("draft_artifact_format", "")) != "spec_draft_triton_v2":
            raise ValueError(f"unsupported draft artifact format: {meta.get('draft_artifact_format')!r}")

        self.meta = meta
        self.chunk_m = int(meta["chunk_m"])
        self.state_dim = int(meta["state_dim"])
        self.out_dim = int(meta.get("out_dim", meta.get("action_dim", 36)))
        self.hidden_size = int(meta.get("img_dim", meta.get("token_dim", artifact["draft_action_queries"].shape[1])))
        self.num_heads = int(meta.get("draft_num_heads", 8))
        self.num_kv_heads = int(meta.get("draft_num_kv_heads", 1))
        self.head_dim = int(meta.get("draft_head_dim", self.hidden_size // max(1, self.num_heads)))
        self.num_key_value_groups = int(self.num_heads // max(1, self.num_kv_heads))
        self.rms_norm_eps = float(meta.get("draft_rms_norm_eps", 1e-6))
        self.rope_theta = float(meta.get("draft_rope_theta", 5000000.0))
        self.attention_scaling = self.head_dim**-0.5

        q_end = self.num_heads * self.head_dim
        k_end = q_end + self.num_kv_heads * self.head_dim
        total_qkv = q_end + 2 * self.num_kv_heads * self.head_dim
        if int(artifact["draft_qkv_w"].shape[0]) != total_qkv:
            raise ValueError(f"draft_qkv_w has incompatible shape {tuple(artifact['draft_qkv_w'].shape)}")
        self.q_end = int(q_end)
        self.k_end = int(k_end)

        device = torch.device(device)
        for key in sorted(self.REQUIRED_KEYS):
            tensor = torch.as_tensor(artifact[key], dtype=torch.float32, device=device).contiguous()
            self.register_buffer(key, tensor, persistent=False)
        q_norm = artifact.get("draft_attn_q_norm_w")
        k_norm = artifact.get("draft_attn_k_norm_w")
        if q_norm is None or k_norm is None:
            q_norm = torch.ones(self.head_dim, dtype=torch.float32)
            k_norm = torch.ones(self.head_dim, dtype=torch.float32)
        self.register_buffer("draft_attn_q_norm_w", torch.as_tensor(q_norm, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("draft_attn_k_norm_w", torch.as_tensor(k_norm, dtype=torch.float32, device=device), persistent=False)

        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @classmethod
    def from_path(cls, path: str | Path, *, device: str | torch.device) -> "JEPADraftArtifactRuntime":
        with Path(path).expanduser().open("rb") as handle:
            artifact = pickle.load(handle)
        if not isinstance(artifact, dict):
            raise ValueError(f"draft artifact must be a dict: {path}")
        return cls(artifact, device=device)

    @staticmethod
    def _make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
        cumsum = torch.cumsum(att_masks.to(dtype=torch.int64), dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    def _build_attention_mask(self, *, prefix_pad_masks: torch.Tensor, prefix_att_masks: torch.Tensor) -> torch.Tensor:
        batch, prefix_len = prefix_pad_masks.shape
        state_pad = torch.ones((batch, 1), device=prefix_pad_masks.device, dtype=torch.bool)
        state_att = torch.zeros((batch, 1), device=prefix_pad_masks.device, dtype=torch.bool)
        prefix_plus_state_pad = torch.cat([prefix_pad_masks.to(dtype=torch.bool), state_pad], dim=1)
        prefix_plus_state_att = torch.cat([prefix_att_masks.to(dtype=torch.bool), state_att], dim=1)
        prefix_mask = self._make_att_2d_masks(prefix_plus_state_pad, prefix_plus_state_att)

        total = int(prefix_len + 1 + self.chunk_m)
        mask = torch.zeros((batch, total, total), device=prefix_pad_masks.device, dtype=torch.bool)
        prefix_state_len = int(prefix_mask.shape[1])
        mask[:, :prefix_state_len, :prefix_state_len] = prefix_mask
        mask[:, prefix_state_len:, :prefix_state_len] = prefix_plus_state_pad[:, None, :]
        mask[:, prefix_state_len:, prefix_state_len:] = True
        return mask

    def _build_position_ids(self, *, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
        batch = int(prefix_pad_masks.shape[0])
        state_pad = torch.ones((batch, 1), device=prefix_pad_masks.device, dtype=torch.bool)
        query_pad = torch.ones((batch, int(self.chunk_m)), device=prefix_pad_masks.device, dtype=torch.bool)
        pad_mask = torch.cat([prefix_pad_masks.to(dtype=torch.bool), state_pad, query_pad], dim=1)
        return (torch.cumsum(pad_mask.to(dtype=torch.int64), dim=1) - 1).clamp_min(0)

    def _rotary_cos_sin(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bs,d->bsd", position_ids.to(torch.float32), self.inv_freq.to(position_ids.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(hidden_states.dtype), emb.sin().to(hidden_states.dtype)

    def forward(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        robot_state: torch.Tensor,
        last_actions: torch.Tensor,
    ) -> torch.Tensor:
        del last_actions
        if prefix_embs.ndim != 3:
            raise ValueError(f"expected prefix_embs to be (B,S,H), got {tuple(prefix_embs.shape)}")
        if int(prefix_embs.shape[2]) != int(self.hidden_size):
            raise ValueError(f"expected prefix hidden size {self.hidden_size}, got {int(prefix_embs.shape[2])}")
        if robot_state.ndim != 2 or int(robot_state.shape[1]) != int(self.state_dim):
            raise ValueError(f"expected robot_state to be (B,{self.state_dim}), got {tuple(robot_state.shape)}")

        batch = int(robot_state.shape[0])
        x_dtype = self.draft_qkv_w.dtype
        prefix_embs = prefix_embs.to(dtype=x_dtype)
        robot_state = robot_state.to(dtype=x_dtype)
        state_token = F.linear(robot_state, self.draft_state_in_proj_w, self.draft_state_in_proj_b)[:, None, :]
        query_tokens = self.draft_action_queries[None, :, :].expand(batch, -1, -1)
        hidden_states = torch.cat([prefix_embs, state_token, query_tokens], dim=1)

        residual = hidden_states
        normed = _rms_norm(hidden_states, self.draft_input_layernorm_w, self.rms_norm_eps)
        qkv = F.linear(normed, self.draft_qkv_w)
        query = qkv[..., : self.q_end].view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = qkv[..., self.q_end : self.k_end].view(batch, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = qkv[..., self.k_end :].view(batch, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query = _rms_norm(query, self.draft_attn_q_norm_w, self.rms_norm_eps)
        key = _rms_norm(key, self.draft_attn_k_norm_w, self.rms_norm_eps)

        position_ids = self._build_position_ids(prefix_pad_masks=prefix_pad_masks)
        cos, sin = self._rotary_cos_sin(hidden_states, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * float(self.attention_scaling)
        mask_2d = self._build_attention_mask(prefix_pad_masks=prefix_pad_masks, prefix_att_masks=prefix_att_masks)
        attention_mask = torch.where(
            mask_2d[:, None, :, :],
            torch.zeros((), device=hidden_states.device, dtype=attn_weights.dtype),
            torch.full((), torch.finfo(attn_weights.dtype).min, device=hidden_states.device, dtype=attn_weights.dtype),
        )
        attn_weights = attn_weights + attention_mask
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_probs, value).transpose(1, 2).contiguous().view(batch, -1, self.hidden_size)
        hidden_states = residual + F.linear(attn_output, self.draft_attn_o_w)

        residual = hidden_states
        normed = _rms_norm(hidden_states, self.draft_post_attention_layernorm_w, self.rms_norm_eps)
        mlp = F.silu(F.linear(normed, self.draft_ffn_gate_w)) * F.linear(normed, self.draft_ffn_up_w)
        hidden_states = residual + F.linear(mlp, self.draft_ffn_down_w)

        query_hidden = hidden_states[:, -int(self.chunk_m) :, :]
        return F.linear(query_hidden, self.draft_action_head_w, self.draft_action_head_b).to(torch.float32)


def _compile_module(
    module: torch.nn.Module,
    *,
    name: str,
    enabled: bool,
    options: dict[str, Any] | None = None,
    strict: bool = False,
) -> tuple[torch.nn.Module, bool]:
    module.eval()
    if not enabled:
        return module, False
    if not hasattr(torch, "compile"):
        logging.warning("torch.compile is unavailable; %s will run eager", name)
        return module, False

    try:
        compile_kwargs: dict[str, Any] = {"fullgraph": False}
        if options is None:
            compile_kwargs["mode"] = "max-autotune"
        else:
            compile_kwargs["options"] = options
        compiled = torch.compile(module, **compile_kwargs)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Failed to compile {name} with torch.compile") from exc
        logging.warning("Failed to compile %s with torch.compile; falling back to eager: %s", name, exc)
        return module, False
    return compiled, True


class JEPATritonRuntime:
    """First JEPA runtime for the Triton backend.

    VLA-JEPA full inference includes Python-heavy Qwen3-VL image/text preparation and a
    large PyTorch checkpoint. This runtime keeps that official path intact, while using
    TorchInductor/Triton for tensor-only subgraphs that are safe to compile now. The
    draft head is the main accelerated path in this first version.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        draft_head: torch.nn.Module | None,
        device: str | torch.device,
        manifest_path: str | Path | None = None,
        compile_full_action_head: bool = False,
        compile_draft_head: bool = True,
        verify_t_list: tuple[float, ...] | None = None,
        compile_verify_head: bool = True,
    ) -> None:
        self.model = model
        self.draft_head = draft_head
        self.device = torch.device(device)
        self.manifest_path = None if manifest_path is None else Path(manifest_path).expanduser().resolve()
        self.manifest: dict[str, Any] = self._load_manifest(self.manifest_path)
        self.full_artifact_path = self._resolve_manifest_artifact("full_artifact")
        self.draft_artifact_path = self._resolve_draft_artifact_path()
        self._draft_runtime_source = "none"
        self._prefix_runtime = JEPAPrefixRuntime(model=model, device=self.device)
        self._full_runtime_enabled = bool(self.manifest.get("full_checkpoint_triton_supported", False))
        compile_visual_encoder = os.environ.get("JEPA_COMPILE_QWEN_VISUAL", "1") != "0"

        self._compiled_language_model: torch.nn.Module | None = None
        self._compiled_language_model_enabled = False
        self._compiled_image_features: torch.nn.Module | None = None
        self._compiled_image_features_enabled = False
        try:
            qwen_model = self.model.qwen_vl_interface.model.model
            language_model = qwen_model.language_model
        except Exception:
            qwen_model = None
            language_model = None
        if qwen_model is not None:
            image_features = _QwenImageFeaturesForward(qwen_model).to(self.device)
            self._compiled_image_features, self._compiled_image_features_enabled = _compile_module(
                image_features,
                name="VLA-JEPA Qwen3-VL image features",
                enabled=bool(compile_visual_encoder),
            )
            self._prefix_runtime.set_image_features_forward(
                self._compiled_image_features,
                compiled=bool(self._compiled_image_features_enabled),
            )
        if language_model is not None:
            self._compiled_language_model, self._compiled_language_model_enabled = _compile_module(
                language_model.to(self.device),
                name="VLA-JEPA Qwen3-VL language_model",
                enabled=bool(self._full_runtime_enabled),
            )

        self._compiled_action_predict: torch.nn.Module | None = None
        self._compiled_action_predict_enabled = False
        action_model = getattr(model, "action_model", None)
        if action_model is not None:
            action_predict = _ActionPredict(action_model).to(self.device)
            self._compiled_action_predict, self._compiled_action_predict_enabled = _compile_module(
                action_predict,
                name="VLA-JEPA action_model.predict_action",
                enabled=bool(compile_full_action_head) or bool(self._full_runtime_enabled),
            )

        self._compiled_action_verify: torch.nn.Module | None = None
        self._compiled_action_verify_enabled = False
        self._action_verify_eager: torch.nn.Module | None = None
        self._action_verify_warned = False
        self._action_verify_checked = False
        compile_action_verify = os.environ.get("JEPA_COMPILE_ACTION_VERIFY", "1") != "0"
        if action_model is not None and verify_t_list:
            action_verify = _ActionVerify(action_model, tuple(float(t) for t in verify_t_list)).to(self.device)
            self._action_verify_eager = action_verify
            self._compiled_action_verify, self._compiled_action_verify_enabled = _compile_module(
                action_verify,
                name="VLA-JEPA action_model.verify",
                enabled=bool(compile_verify_head) and bool(compile_action_verify),
                options={"triton.cudagraphs": False},
                strict=True,
            )

        self._compiled_draft_forward: torch.nn.Module | None = None
        self._compiled_draft_forward_enabled = False
        draft_forward: torch.nn.Module | None = None
        if self.draft_artifact_path is not None:
            try:
                draft_forward = JEPADraftArtifactRuntime.from_path(self.draft_artifact_path, device=self.device)
                self._draft_runtime_source = "artifact"
            except Exception as exc:
                logging.warning(
                    "Failed to load JEPA draft artifact %s; falling back to PyTorch draft head: %s",
                    self.draft_artifact_path,
                    exc,
                )
        if draft_forward is None and draft_head is not None:
            draft_forward = _DraftForward(draft_head).to(self.device)
            self._draft_runtime_source = "pytorch_module"
        if draft_forward is not None:
            self._compiled_draft_forward, self._compiled_draft_forward_enabled = _compile_module(
                draft_forward.to(self.device),
                name=f"VLA-JEPA draft head ({self._draft_runtime_source})",
                enabled=bool(compile_draft_head),
            )

        logging.info(
            "Initialized JEPA Triton runtime manifest=%s draft_artifact=%s draft_source=%s "
            "full_artifact=%s visual_compiled=%s language_compiled=%s full_action_compiled=%s "
            "verify_compiled=%s draft_compiled=%s",
            self.manifest_path,
            self.draft_artifact_path,
            self._draft_runtime_source,
            self.full_artifact_path,
            self._compiled_image_features_enabled,
            self._compiled_language_model_enabled,
            self._compiled_action_predict_enabled,
            self._compiled_action_verify_enabled,
            self._compiled_draft_forward_enabled,
        )

    @staticmethod
    def _load_manifest(path: Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        if not path.exists():
            logging.warning("JEPA Triton manifest does not exist: %s", path)
            return {}
        try:
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("Failed to read JEPA Triton manifest %s: %s", path, exc)
            return {}
        if not bool(data.get("full_checkpoint_triton_supported", False)):
            logging.info(
                "JEPA manifest marks full_checkpoint_triton_supported=false; full path will use "
                "the official VLA-JEPA PyTorch model with compiled tensor subgraphs where available"
            )
        return data if isinstance(data, dict) else {}

    def _resolve_draft_artifact_path(self) -> Path | None:
        return self._resolve_manifest_artifact("draft_artifact")

    def _resolve_manifest_artifact(self, key: str) -> Path | None:
        artifact = self.manifest.get(key)
        if not artifact:
            return None
        artifact_path = Path(str(artifact)).expanduser()
        if artifact_path.is_absolute():
            return artifact_path.resolve()
        if self.manifest_path is None:
            return artifact_path.resolve()
        return (self.manifest_path.parent / artifact_path).resolve()

    def describe(self) -> dict[str, Any]:
        return {
            "runtime": "jepa_triton_runtime",
            "manifest_path": None if self.manifest_path is None else str(self.manifest_path),
            "runtime_family": self.manifest.get("runtime_family"),
            "full_checkpoint_triton_supported": bool(self.manifest.get("full_checkpoint_triton_supported", False)),
            "full_artifact_path": None if self.full_artifact_path is None else str(self.full_artifact_path),
            "compiled_image_features": bool(self._compiled_image_features_enabled),
            "compiled_language_model": bool(self._compiled_language_model_enabled),
            "compiled_action_predict": bool(self._compiled_action_predict_enabled),
            "compiled_action_verify": bool(self._compiled_action_verify_enabled),
            "compiled_draft_forward": bool(self._compiled_draft_forward_enabled),
            "draft_runtime_source": self._draft_runtime_source,
            "draft_artifact_path": None if self.draft_artifact_path is None else str(self.draft_artifact_path),
        }

    @property
    def has_draft_runtime(self) -> bool:
        return self._compiled_draft_forward is not None

    @property
    def draft_chunk_m(self) -> int | None:
        if self._compiled_draft_forward is None:
            return None
        module = getattr(self._compiled_draft_forward, "_orig_mod", self._compiled_draft_forward)
        return getattr(module, "chunk_m", None)

    def predict_full(
        self,
        *,
        batch_images: list[list[Image.Image]],
        instructions: list[str],
        state: list[Any],
    ) -> dict[str, Any]:
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.predict_action(
                batch_images=batch_images,
                instructions=instructions,
                state=state,
            )
        output = dict(output)
        output["_runtime_timing"] = {
            "triton_runtime_full_ms": float((time.perf_counter() - start) * 1000.0),
            "triton_runtime_full_compiled": 0.0,
        }
        return output

    def run_language_model(
        self,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        inputs_embeds: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        deepstack_visual_embeds: list[torch.Tensor] | None,
    ):
        if self._compiled_language_model is None:
            raise RuntimeError("JEPA Triton runtime was created without a Qwen language model")
        return self._compiled_language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            cache_position=None,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            output_hidden_states=True,
            return_dict=True,
        )

    def predict_actions_from_embodied(self, *, embodied_action_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self._compiled_action_predict is None:
            raise RuntimeError("JEPA Triton runtime was created without a compiled action predictor")
        return self._compiled_action_predict(embodied_action_tokens, state)

    def predict_verify(
        self,
        *,
        embodied_action_tokens: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
    ) -> torch.Tensor:
        if self._compiled_action_verify is None:
            raise RuntimeError("JEPA Triton runtime was created without an action verifier")
        try:
            compiled_out = self._compiled_action_verify(
                embodied_action_tokens,
                state,
                noise,
                x0_draft,
            )
            if (
                self._compiled_action_verify_enabled
                and self._action_verify_eager is not None
                and not self._action_verify_checked
                and os.environ.get("JEPA_VERIFY_COMPILED_PARITY_CHECK", "1") != "0"
            ):
                self._action_verify_checked = True
                eager_out = self._action_verify_eager(
                    embodied_action_tokens,
                    state,
                    noise,
                    x0_draft,
                )
                diff = (compiled_out.detach().to(torch.float32) - eager_out.detach().to(torch.float32)).abs()
                max_diff = float(diff.max().item()) if diff.numel() else 0.0
                rms_diff = float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() else 0.0
                tol = float(os.environ.get("JEPA_VERIFY_COMPILED_PARITY_TOL", "1e-3"))
                if max_diff > tol or rms_diff > tol:
                    raise RuntimeError(
                        f"Compiled JEPA verify path failed parity check "
                        f"max_diff={max_diff:.6g} rms_diff={rms_diff:.6g} tol={tol:.6g}"
                    )
            return compiled_out.clone()
        except Exception as exc:
            raise RuntimeError(f"Compiled JEPA verify path failed without eager fallback: {exc}") from exc

    def prepare_policy_prefix(self, *, batch_images: list[list[Image.Image]], instructions: list[str]) -> dict[str, Any]:
        try:
            return self._prefix_runtime.prepare_policy_prefix(batch_images=batch_images, instructions=instructions)
        except Exception as exc:
            logging.warning("Fast JEPA prefix runtime failed; falling back to stock VLA-JEPA prefix path: %s", exc)
            start = time.perf_counter()
            with torch.inference_mode():
                prefix = self.model.prepare_draft_prefix(batch_images=batch_images, instructions=instructions)
            prefix = dict(prefix)
            prefix["_runtime_timing"] = {
                "triton_runtime_prefix_ms": float((time.perf_counter() - start) * 1000.0),
                "prefix_fast_fallback": 1.0,
            }
            return prefix

    def prepare_draft_prefix(self, *, batch_images: list[list[Image.Image]], instructions: list[str]) -> dict[str, Any]:
        return self.prepare_policy_prefix(batch_images=batch_images, instructions=instructions)

    def predict_draft(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        robot_state: torch.Tensor,
        last_actions: torch.Tensor,
    ) -> torch.Tensor:
        if self._compiled_draft_forward is None:
            raise RuntimeError("JEPA Triton runtime was created without a draft head")
        with torch.inference_mode():
            return self._compiled_draft_forward(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                robot_state,
                last_actions,
            )

    def reset_runtime_state(self) -> None:
        if hasattr(self, "_prefix_runtime"):
            self._prefix_runtime.reset_runtime_state()
