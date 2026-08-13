from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from starVLA.model.modules.vla_conditioner.state_encoder import StateEncoder
from starVLA.model.modules.vla_conditioner.state_token_projector import StateTokenProjector


class QwenVLAConditioner(nn.Module):
    def __init__(
        self,
        qwen_vl_interface: nn.Module,
        state_dim: int,
        state_hidden_dim: int,
        state_feature_dim: int,
        num_state_tokens: int = 4,
        freeze_qwen: bool = True,
    ):
        super().__init__()
        self.qwen_vl_interface = qwen_vl_interface
        self.freeze_qwen = freeze_qwen
        self.num_state_tokens = num_state_tokens
        self.qwen_hidden_dim = int(qwen_vl_interface.model.config.hidden_size)
        self._generation_prompt_len: int | None = None

        self.state_encoder = (
            StateEncoder(state_dim, state_hidden_dim, state_feature_dim) if state_dim else None
        )
        self.state_token_projector = (
            StateTokenProjector(
                input_dim=state_feature_dim,
                qwen_hidden_dim=self.qwen_hidden_dim,
                num_tokens=num_state_tokens,
                hidden_dim=state_hidden_dim,
            )
            if state_dim
            else None
        )

        if freeze_qwen:
            self.qwen_vl_interface.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_qwen:
            self.qwen_vl_interface.eval()
        return self

    def _encode_state_tokens(
        self,
        state,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.state_encoder is None or self.state_token_projector is None:
            return None
        if state is None:
            raise ValueError("QwenVLAConditioner expects `state` when StateEncoder is enabled.")
        state_tensor = torch.as_tensor(np.array(state), device=device, dtype=dtype)
        state_features = self.state_encoder(state_tensor)
        return self.state_token_projector(state_features)

    def _get_generation_prompt_len(self) -> int:
        if self._generation_prompt_len is not None:
            return self._generation_prompt_len

        processor = self.qwen_vl_interface.processor
        dummy = [[{"role": "user", "content": [{"type": "text", "text": "x"}]}]]
        with_prompt = processor.apply_chat_template(
            dummy,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"]
        without_prompt = processor.apply_chat_template(
            dummy,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"]
        gen_len = int(with_prompt.shape[1] - without_prompt.shape[1])
        if gen_len <= 0:
            raise RuntimeError(
                f"Failed to measure Qwen generation prompt length "
                f"(with={with_prompt.shape[1]}, without={without_prompt.shape[1]})."
            )
        self._generation_prompt_len = gen_len
        return gen_len

    def _insert_state_before_generation_prompt(
        self,
        qwen_inputs,
        num_state_tokens: int,
    ):
        gen_len = self._get_generation_prompt_len()
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs["attention_mask"]
        batch_size, seq_len = input_ids.shape
        if seq_len < gen_len:
            raise ValueError(
                f"Sequence length {seq_len} is shorter than generation prompt length {gen_len}."
            )

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        content = input_ids[:, :-gen_len]
        gen_prompt = input_ids[:, -gen_len:]
        content_mask = attention_mask[:, :-gen_len]
        gen_mask = attention_mask[:, -gen_len:]

        placeholders = torch.full(
            (batch_size, num_state_tokens),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        state_mask = torch.ones(
            (batch_size, num_state_tokens),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        qwen_inputs["input_ids"] = torch.cat([content, placeholders, gen_prompt], dim=1)
        qwen_inputs["attention_mask"] = torch.cat([content_mask, state_mask, gen_mask], dim=1)

        if "token_type_ids" in qwen_inputs and qwen_inputs["token_type_ids"] is not None:
            token_type_ids = qwen_inputs["token_type_ids"]
            tt_content = token_type_ids[:, :-gen_len]
            tt_gen = token_type_ids[:, -gen_len:]
            tt_state = torch.zeros(
                (batch_size, num_state_tokens),
                dtype=token_type_ids.dtype,
                device=token_type_ids.device,
            )
            qwen_inputs["token_type_ids"] = torch.cat([tt_content, tt_state, tt_gen], dim=1)

        state_start = content.shape[1]
        return qwen_inputs, state_start

    def _inject_state_into_inputs_embeds(
        self,
        qwen_inputs,
        state_tokens: torch.Tensor,
        state_start: int,
    ):
        input_ids = qwen_inputs["input_ids"]
        embed_layer = self.qwen_vl_interface.model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).clone()
        num_tokens = state_tokens.shape[1]
        inputs_embeds[:, state_start : state_start + num_tokens, :] = state_tokens.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        qwen_inputs["inputs_embeds"] = inputs_embeds
        return qwen_inputs

    def _prepare_qwen_forward_kwargs(self, qwen_inputs):
        forward_kwargs = dict(qwen_inputs)
        if "inputs_embeds" not in forward_kwargs or forward_kwargs["inputs_embeds"] is None:
            return forward_kwargs

        input_ids = forward_kwargs.get("input_ids", None)
        attention_mask = forward_kwargs.get("attention_mask", None)
        image_grid_thw = forward_kwargs.get("image_grid_thw", None)
        video_grid_thw = forward_kwargs.get("video_grid_thw", None)

        if input_ids is not None:
            qwen_model = self.qwen_vl_interface.model.model
            position_ids, rope_deltas = qwen_model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            forward_kwargs["position_ids"] = position_ids
            qwen_model.rope_deltas = rope_deltas
            forward_kwargs.pop("input_ids", None)

        return forward_kwargs

    def _pool_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden[:, -1:, :]
        lengths = attention_mask.to(device=hidden.device).sum(dim=1).clamp_min(1).long() - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, lengths].unsqueeze(1)

    def forward(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state=None,
        prompt_template: str = "{instruction}",
    ) -> torch.Tensor:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_template=prompt_template,
            add_generation_prompt=True,
        )

        device = qwen_inputs["input_ids"].device
        model_dtype = next(self.qwen_vl_interface.model.parameters()).dtype
        state_tokens = self._encode_state_tokens(state, device=device, dtype=model_dtype)

        if state_tokens is not None:
            qwen_inputs, state_start = self._insert_state_before_generation_prompt(
                qwen_inputs, state_tokens.shape[1]
            )
            qwen_inputs = self._inject_state_into_inputs_embeds(
                qwen_inputs, state_tokens, state_start
            )

        forward_kwargs = self._prepare_qwen_forward_kwargs(qwen_inputs)
        qwenvl_outputs = self.qwen_vl_interface(
            **forward_kwargs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = qwenvl_outputs.hidden_states[-1]
        attention_mask = forward_kwargs.get("attention_mask", None)
        return self._pool_hidden(last_hidden, attention_mask)
