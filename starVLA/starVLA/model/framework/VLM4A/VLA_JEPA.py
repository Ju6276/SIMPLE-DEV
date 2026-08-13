from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoVideoProcessor

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import resolve_model_id_or_path
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.world_model.local_vjepa_encoder import (
    LocalVJEPAVideoProcessor,
    VJEPA2LocalEncoderModel,
    load_local_vjepa_encoder_checkpoint,
    resolve_local_vjepa_checkpoint,
)
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """Inference-compatible VLA_JEPA framework."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token,
        )

        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        vj_base_encoder = resolve_model_id_or_path(self.config.framework.vj2_model.base_encoder)
        local_vjepa_checkpoint = resolve_local_vjepa_checkpoint(vj_base_encoder)
        if local_vjepa_checkpoint is not None:
            self.vj_encoder = VJEPA2LocalEncoderModel(
                image_size=self.config.framework.vj2_model.get("image_size", 384),
                depth=self.config.framework.vj2_model.get("encoder_depth", 24),
                num_heads=self.config.framework.vj2_model.get("encoder_num_heads", 16),
            )
            load_local_vjepa_encoder_checkpoint(self.vj_encoder, local_vjepa_checkpoint)
            self.vj_processor = LocalVJEPAVideoProcessor(size=self.vj_encoder.config.image_size)
        else:
            processor_path = self.config.framework.vj2_model.get("processor_path", None)
            self.vj_encoder = AutoModel.from_pretrained(vj_base_encoder)
            self.vj_processor = AutoVideoProcessor.from_pretrained(str(processor_path or vj_base_encoder))

        tubelet_size = self.vj_encoder.config.tubelet_size
        num_video_views = self.config.framework.vj2_model.get("num_video_views", 1)
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames // tubelet_size,
            img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_encoder.config.hidden_size * num_video_views,
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )
        self.replace_prompt = "".join(
            [
                each * self.config.framework.vj2_model.num_action_tokens_per_timestep
                for each in action_tokens[: self.config.framework.vj2_model.num_frames // tubelet_size - 1]
            ]
        )
        self.embodied_replace_prompt = "".join(
            [embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction]
        )
        self.image_token_id = self.qwen_vl_interface.model.config.image_token_id
        self.spatial_merge_size = self.qwen_vl_interface.model.visual.spatial_merge_size
        self.vision_start_token_id = getattr(self.qwen_vl_interface.model.config, "vision_start_token_id", None)
        self.pad_token_id = self.qwen_vl_interface.processor.tokenizer.pad_token_id

    def _select_alignment_query(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        batch_idx: int,
        embodied_action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Prefer language-token pooling for V-L alignment; fall back to action-conditioned tokens."""
        text_mask = input_ids[batch_idx] != self.image_token_id
        if self.vision_start_token_id is not None:
            text_mask = text_mask & (input_ids[batch_idx] != self.vision_start_token_id)
        if self.pad_token_id is not None:
            text_mask = text_mask & (input_ids[batch_idx] != self.pad_token_id)
        if attention_mask is not None:
            text_mask = text_mask & attention_mask[batch_idx].bool()

        text_hidden = last_hidden[batch_idx][text_mask]
        if text_hidden.numel() > 0:
            return text_hidden.to(torch.float32).mean(dim=0)

        if embodied_action_tokens.shape[1] > 0:
            return embodied_action_tokens[batch_idx].to(torch.float32).mean(dim=0)

        return last_hidden[batch_idx].to(torch.float32).mean(dim=0)

    def _compute_visual_alignment_heatmaps(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        embodied_action_tokens: torch.Tensor,
    ) -> list[np.ndarray]:
        """Project fused action-conditioned language features back onto image tokens."""
        if image_grid_thw is None:
            return []

        heatmaps: list[np.ndarray] = []
        token_counts = (image_grid_thw.prod(-1) // (self.spatial_merge_size**2)).tolist()

        for batch_idx, token_count in enumerate(token_counts):
            image_mask = input_ids[batch_idx] == self.image_token_id
            image_hidden = last_hidden[batch_idx][image_mask]
            if image_hidden.shape[0] == 0:
                heatmaps.append(np.zeros((1, 1), dtype=np.float32))
                continue

            query = self._select_alignment_query(
                last_hidden=last_hidden,
                input_ids=input_ids,
                attention_mask=attention_mask,
                batch_idx=batch_idx,
                embodied_action_tokens=embodied_action_tokens,
            )
            image_hidden = image_hidden[:token_count].to(torch.float32)

            query = F.normalize(query, dim=0, eps=1e-6)
            image_hidden = F.normalize(image_hidden, dim=-1, eps=1e-6)
            scores = torch.matmul(image_hidden, query)

            score_min = scores.min()
            score_max = scores.max()
            if torch.isclose(score_max, score_min):
                scores = torch.zeros_like(scores)
            else:
                scores = (scores - score_min) / (score_max - score_min)

            grid_t, grid_h, grid_w = image_grid_thw[batch_idx].tolist()
            token_h = max(1, grid_h // self.spatial_merge_size)
            token_w = max(1, grid_w // self.spatial_merge_size)
            expected_tokens = max(1, grid_t * token_h * token_w)

            if expected_tokens != scores.numel():
                inferred_side = int(round(scores.numel() ** 0.5))
                token_h = inferred_side
                token_w = max(1, scores.numel() // max(1, inferred_side))

            heatmaps.append(scores.reshape(token_h, token_w).detach().cpu().numpy().astype(np.float32))

        return heatmaps

    def _compute_fallback_visual_heatmaps(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
    ) -> list[np.ndarray]:
        """Best-effort visual saliency fallback based on image-token activation strength."""
        heatmaps: list[np.ndarray] = []

        for batch_idx in range(last_hidden.shape[0]):
            image_mask = input_ids[batch_idx] == self.image_token_id
            image_hidden = last_hidden[batch_idx][image_mask].to(torch.float32)

            if image_hidden.shape[0] == 0:
                heatmaps.append(np.zeros((1, 1), dtype=np.float32))
                continue

            scores = image_hidden.norm(dim=-1)
            score_min = scores.min()
            score_max = scores.max()
            if torch.isclose(score_max, score_min):
                scores = torch.zeros_like(scores)
            else:
                scores = (scores - score_min) / (score_max - score_min)

            if image_grid_thw is not None and batch_idx < image_grid_thw.shape[0]:
                _, grid_h, grid_w = image_grid_thw[batch_idx].tolist()
                token_h = max(1, grid_h // self.spatial_merge_size)
                token_w = max(1, grid_w // self.spatial_merge_size)
            else:
                inferred_side = max(1, int(round(scores.numel() ** 0.5)))
                token_h = inferred_side
                token_w = max(1, scores.numel() // inferred_side)

            total_tokens = token_h * token_w
            if total_tokens != scores.numel():
                inferred_side = max(1, int(round(scores.numel() ** 0.5)))
                token_h = inferred_side
                token_w = max(1, scores.numel() // inferred_side)
                total_tokens = token_h * token_w

            scores = scores[:total_tokens]
            heatmaps.append(scores.reshape(token_h, token_w).detach().cpu().numpy().astype(np.float32))

        return heatmaps

    def expand_tokenizer(
        self,
        tokenizer: AutoTokenizer,
        special_action_token: str = "<|action_{}|>",
        max_action_tokens: int = 32,
        embodied_action_token: str = "<|embodied_action|>",
    ):
        action_tokens, action_token_ids = [], []
        for i in range(max_action_tokens):
            token = special_action_token.format(i)
            action_tokens.append(token)
            if token not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([token], special_tokens=True)
                if added == 0:
                    logger.warning("0 tokens added for %s; it may already exist.", token)
            action_token_ids.append(tokenizer.convert_tokens_to_ids(token))

        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning("0 tokens added for %s; it may already exist.", embodied_action_token)
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if embedding_size < len(tokenizer):
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        return action_tokens, action_token_ids, embodied_action_token_id

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        batch_videos = [example["video"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples] if "action" in examples[0] else None
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        batch_videos = np.stack(batch_videos).transpose(0, 1, 2, 5, 3, 4)

        if actions is not None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_replace_dict={"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
            )
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_replace_dict={"{actions}": self.replace_prompt},
                prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""),
            )

        action_indices = torch.isin(
            qwen_inputs["input_ids"], torch.tensor(self.action_token_ids, device=qwen_inputs["input_ids"].device)
        ).nonzero(as_tuple=True)
        embodied_action_indices = torch.isin(
            qwen_inputs["input_ids"],
            torch.tensor([self.embodied_action_token_id], device=qwen_inputs["input_ids"].device),
        ).nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            batch_size, _, hidden_size = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(batch_size, -1, hidden_size)
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(
                batch_size, -1, hidden_size
            )

            batch_size, num_views, frames, channels, height, width = batch_videos.shape
            batch_videos = batch_videos.reshape(batch_size * num_views, frames, channels, height, width)
            input_videos = []
            for i in range(batch_size * num_views):
                input_videos.append(
                    self.vj_processor(videos=batch_videos[i], return_tensors="pt")["pixel_values_videos"].to(
                        self.vj_encoder.device
                    )
                )
            input_videos = torch.cat(input_videos, dim=0)
            with torch.no_grad():
                video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=num_views, dim=0), dim=2)

            frames = frames // self.vj_encoder.config.tubelet_size
            input_states = video_embeddings[:, : video_embeddings.shape[1] // frames * (frames - 1), :]
            gt_states = video_embeddings[:, video_embeddings.shape[1] // frames :, :]
            predicted_states = self.vj_predictor(input_states, action_tokens)
            teacher_forcing_wm_loss = F.l1_loss(predicted_states, gt_states, reduction="mean")

        if "action" not in examples[0]:
            return {"wm_loss": teacher_forcing_wm_loss}

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]
            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            embodied_action_repeated = embodied_action_tokens.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(embodied_action_repeated, actions_target_repeated, state_repeated)
        return {"action_loss": action_loss, "wm_loss": teacher_forcing_wm_loss * 0.1}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict={"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt},
        )

        embodied_action_indices = torch.isin(
            qwen_inputs["input_ids"],
            torch.tensor([self.embodied_action_token_id], device=qwen_inputs["input_ids"].device),
        ).nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            batch_size, _, hidden_size = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(
                batch_size, -1, hidden_size
            )
            try:
                visual_alignment_heatmaps = self._compute_visual_alignment_heatmaps(
                    last_hidden=last_hidden,
                    input_ids=qwen_inputs["input_ids"],
                    attention_mask=qwen_inputs.get("attention_mask"),
                    image_grid_thw=qwen_inputs.get("image_grid_thw"),
                    embodied_action_tokens=embodied_action_tokens,
                )
            except Exception as exc:
                logger.warning("visual alignment heatmap generation failed: %s", exc)
                visual_alignment_heatmaps = self._compute_fallback_visual_heatmaps(
                    last_hidden=last_hidden,
                    input_ids=qwen_inputs["input_ids"],
                    image_grid_thw=qwen_inputs.get("image_grid_thw"),
                )

            if not visual_alignment_heatmaps:
                visual_alignment_heatmaps = self._compute_fallback_visual_heatmaps(
                    last_hidden=last_hidden,
                    input_ids=qwen_inputs["input_ids"],
                    image_grid_thw=qwen_inputs.get("image_grid_thw"),
                )

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(embodied_action_tokens, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {
            "normalized_actions": normalized_actions,
            "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy(),
            "visual_alignment_heatmaps": visual_alignment_heatmaps,
        }
