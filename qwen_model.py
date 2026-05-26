"""
Qwen3-VL-based reward model for preference learning.

Following the QwenLM/Qwen3-VL finetuning approach (see memer-policy/memer):
  - Vision encoder frozen, LLM trainable (optionally MLP too)
  - Gradient checkpointing for memory efficiency
  - bf16 precision

Variants:
  qwen / qwen_lora         — all frames in one VLM input, K reward heads
  qwen_open                — all frames in one VLM input, 1 reward head, axis in prompt
  qwen_discounted          — per-frame scoring (each frame pair independent), sum over time
  qwen_open_discounted     — per-frame + open axis
  qwen_open_cum            — open axis, single VLM pass with prompt first so every
                             frame can attend to it via the causal mask; pool the
                             reward head at each frame pair's last <|vision_end|>
                             token and sum the per-timestep rewards.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, AutoModelForImageTextToText


_TRAJECTORY_PROMPT = "What is the score for the trajectory?"
_FRAME_PROMPT = "What is the score for this frame?"


class QwenRewardModel(nn.Module):
    """
    Qwen3-VL reward model following the QwenLM finetuning approach.

    Freezes the vision encoder and optionally the MLP projector,
    training only the LLM backbone + reward heads (or LoRA adapters).

    When discounted=True, each frame pair (third_person + wrist) is scored
    independently and the per-frame scores are summed (like DiscountedRewardModel).

    Forward input:
        third_person: (B, T, 3, H, W)
        wrist:        (B, T, 3, H, W)

    Forward output:
        rewards: (B, K)
    """

    def __init__(
        self,
        num_preferences: int = 5,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        use_lora: bool = False,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        reward_sigmoid: bool = False,
        tune_vision: bool = False,
        tune_mlp: bool = True,
        tune_llm: bool = True,
        gradient_checkpointing: bool = True,
        max_pixels: int = 50176,
        min_pixels: int = 784,
        discounted: bool = False,
        open_cum: bool = False,
    ):
        super().__init__()
        self.num_preferences = num_preferences
        self.reward_sigmoid = reward_sigmoid
        self.use_lora = use_lora
        self.model_name = model_name
        self.discounted = discounted
        self.open_cum = open_cum
        if discounted and open_cum:
            raise ValueError("discounted and open_cum are mutually exclusive")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        # Use flash_attention_2 if available, otherwise fall back to sdpa
        # (PyTorch's SDPA uses flash-attention kernels internally when possible)
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        self.qwen = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        print(f"[QwenRewardModel] Using {attn_impl}")

        # Get hidden size from config before freezing/LoRA
        config = self.qwen.config
        if hasattr(config, "text_config"):
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            hidden_size = 2048  # Qwen3-VL 4B default
        self._hidden_size = hidden_size

        # ----- Freeze components (following QwenLM finetuning pattern) -----
        print("[QwenRewardModel] Freezing components...", flush=True)
        if not tune_vision:
            for name, param in self.qwen.named_parameters():
                if "visual" in name:
                    param.requires_grad = False

        if not tune_mlp:
            for name, param in self.qwen.named_parameters():
                if "merger" in name or "mlp" in name.split(".")[:3]:
                    param.requires_grad = False

        if not tune_llm:
            for name, param in self.qwen.named_parameters():
                if "model.layers" in name or "model.norm" in name or "model.embed" in name:
                    param.requires_grad = False

        print("[QwenRewardModel] Enabling gradient checkpointing...", flush=True)
        # Gradient checkpointing (critical for 4B model memory)
        if gradient_checkpointing:
            self.qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # ----- Apply LoRA if requested -----
        if use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.qwen = get_peft_model(self.qwen, lora_config)
            self.qwen.print_trainable_parameters()

        # Reward heads (float32 for stable training)
        self.reward_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_preferences)
        ])

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _valid_frame_indices(self, T: int, padding_mask_b: torch.Tensor = None) -> list[int]:
        """Return indices of non-padded frames."""
        if padding_mask_b is not None:
            valid = (~padding_mask_b).nonzero(as_tuple=True)[0]
            return valid.tolist() if len(valid) > 0 else [0]
        return list(range(T))

    def _prepare_inputs(
        self,
        third_person: torch.Tensor,
        wrist: torch.Tensor,
        padding_mask: torch.Tensor | None,
        axis_labels: list[str] | None = None,
    ) -> dict:
        """
        All frames in one VLM input per trajectory.
        Returns processor inputs with batch size B.
        """
        B, T = third_person.shape[:2]
        all_texts = []
        all_images = []

        for b in range(B):
            pm = padding_mask[b] if padding_mask is not None else None
            indices = self._valid_frame_indices(T, pm)

            images = []
            content = []
            for idx in indices:
                images.append(to_pil_image(third_person[b, idx].cpu()))
                content.append({"type": "image"})
                images.append(to_pil_image(wrist[b, idx].cpu()))
                content.append({"type": "image"})

            if axis_labels is not None:
                prompt = f"What is the score for '{axis_labels[b]}' in this trajectory?"
            else:
                prompt = _TRAJECTORY_PROMPT
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_texts.append(text)
            all_images.append(images)

        inputs = self.processor(
            text=all_texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
        )
        return inputs

    def _prepare_inputs_open_cum(
        self,
        third_person: torch.Tensor,
        wrist: torch.Tensor,
        padding_mask: torch.Tensor | None,
        axis_labels: list[str] | None = None,
    ) -> tuple[dict, list[int]]:
        """
        Prompt-first, all frames in one VLM input per trajectory.
        Putting the axis prompt before the images lets every frame attend to
        it through the causal mask. Returns processor inputs with batch size B
        and per-batch frame counts so we can find the per-pair pool positions.
        """
        B, T = third_person.shape[:2]
        all_texts = []
        all_images = []
        frame_counts = []

        for b in range(B):
            pm = padding_mask[b] if padding_mask is not None else None
            indices = self._valid_frame_indices(T, pm)
            frame_counts.append(len(indices))

            if axis_labels is not None:
                prompt = f"What is the score for '{axis_labels[b]}' in this trajectory?"
            else:
                prompt = _TRAJECTORY_PROMPT
            content = [{"type": "text", "text": prompt}]

            images = []
            for idx in indices:
                images.append(to_pil_image(third_person[b, idx].cpu()))
                content.append({"type": "image"})
                images.append(to_pil_image(wrist[b, idx].cpu()))
                content.append({"type": "image"})

            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_texts.append(text)
            all_images.append(images)

        inputs = self.processor(
            text=all_texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
        )
        return inputs, frame_counts

    def _prepare_inputs_discounted(
        self,
        third_person: torch.Tensor,
        wrist: torch.Tensor,
        padding_mask: torch.Tensor | None,
        axis_labels: list[str] | None = None,
    ) -> tuple[dict, list[int]]:
        """
        One VLM input per frame pair (2 images + prompt each).
        Returns processor inputs with batch size sum(frame_counts)
        and frame_counts so we can reassemble per-trajectory rewards.
        """
        B, T = third_person.shape[:2]
        all_texts = []
        all_images = []
        frame_counts = []

        for b in range(B):
            pm = padding_mask[b] if padding_mask is not None else None
            indices = self._valid_frame_indices(T, pm)
            frame_counts.append(len(indices))

            for idx in indices:
                images = [
                    to_pil_image(third_person[b, idx].cpu()),
                    to_pil_image(wrist[b, idx].cpu()),
                ]
                content = [
                    {"type": "image"},
                    {"type": "image"},
                ]
                if axis_labels is not None:
                    prompt = f"What is the score for '{axis_labels[b]}' in this frame?"
                else:
                    prompt = _FRAME_PROMPT
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                all_texts.append(text)
                all_images.append(images)

        inputs = self.processor(
            text=all_texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
        )
        return inputs, frame_counts

    # ------------------------------------------------------------------
    # Hidden state extraction
    # ------------------------------------------------------------------

    def _extract_pooled(self, inputs: dict, device: torch.device) -> torch.Tensor:
        """Run VLM forward and pool at last non-padding token. Returns (N, hidden)."""
        # One-shot diagnostic: walk to the actual decoder layer and check the
        # flag that GradientCheckpointingLayer.__call__ tests. This is what
        # really matters; top-level / model-level flags can be True while
        # individual layers still take the fallback path.
        if self.training and not getattr(self, "_gc_diag_logged", False):
            gc_top = getattr(self.qwen, "is_gradient_checkpointing", "?")
            qmodel = getattr(self.qwen, "model", None)
            qlm = getattr(qmodel, "language_model", None) if qmodel is not None else None
            qlayers = getattr(qlm, "layers", None) if qlm is not None else None
            layer0_gc = getattr(qlayers[0], "gradient_checkpointing", "?") if qlayers is not None else "no_layers"
            layer0_train = getattr(qlayers[0], "training", "?") if qlayers is not None else "?"
            layer0_func = type(getattr(qlayers[0], "_gradient_checkpointing_func", None)).__name__ if qlayers is not None else "?"
            print(
                f"[gc_diag] self.training={self.training} qwen.is_gc={gc_top} "
                f"layer0.gc={layer0_gc} layer0.training={layer0_train} "
                f"layer0._gc_func={layer0_func} n_layers={len(qlayers) if qlayers else '?'}",
                flush=True,
            )
            self._gc_diag_logged = True
        outputs = self.qwen(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        if outputs.hidden_states is None:
            raise RuntimeError(
                "Model did not return hidden_states. "
                "Ensure the model supports output_hidden_states=True."
            )
        last_hidden = outputs.hidden_states[-1]

        if "attention_mask" in inputs:
            seq_lengths = inputs["attention_mask"].sum(dim=1).long() - 1
            batch_idx = torch.arange(last_hidden.shape[0], device=device)
            pooled = last_hidden[batch_idx, seq_lengths]
        else:
            pooled = last_hidden[:, -1]

        return pooled.float()

    def _apply_reward_heads(self, pooled: torch.Tensor) -> torch.Tensor:
        """Apply reward heads: (N, hidden) -> (N, K)."""
        return torch.stack(
            [head(pooled).squeeze(-1) for head in self.reward_heads],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        third_person: torch.Tensor,
        wrist: torch.Tensor,
        padding_mask: torch.Tensor = None,
        axis_labels: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Encode a trajectory and output K reward scalars.

        Args:
            third_person: (B, T, 3, H, W)
            wrist:        (B, T, 3, H, W)
            padding_mask: (B, T) bool -- True = padded frame
            axis_labels:  list of B strings (for open models)
        Returns:
            rewards: (B, K)
        """
        device = next(self.parameters()).device

        # Ensure uint8 for PIL conversion
        with torch.no_grad():
            if third_person.dtype != torch.uint8:
                third_person = (third_person * 255).clamp(0, 255).to(torch.uint8)
            if wrist.dtype != torch.uint8:
                wrist = (wrist * 255).clamp(0, 255).to(torch.uint8)

        if self.open_cum:
            return self._forward_open_cum(third_person, wrist, padding_mask, axis_labels, device)
        if self.discounted:
            return self._forward_discounted(third_person, wrist, padding_mask, axis_labels, device)
        else:
            return self._forward_standard(third_person, wrist, padding_mask, axis_labels, device)

    def _forward_standard(self, third_person, wrist, padding_mask, axis_labels, device):
        """All frames in one VLM input → single pooled reward."""
        t0 = time.perf_counter()
        with torch.no_grad():
            inputs = self._prepare_inputs(third_person, wrist, padding_mask, axis_labels=axis_labels)
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
        self._last_prep_time = time.perf_counter() - t0

        pooled = self._extract_pooled(inputs, device)
        rewards = self._apply_reward_heads(pooled)  # (B, K)

        if self.reward_sigmoid:
            rewards = torch.sigmoid(rewards)
        return rewards

    def _forward_open_cum(self, third_person, wrist, padding_mask, axis_labels, device):
        """Prompt-first single VLM pass. Pool at each pair's last <|vision_end|>
        token, apply the reward head, and sum the per-timestep rewards.
        """
        B = third_person.shape[0]

        t0 = time.perf_counter()
        with torch.no_grad():
            inputs, frame_counts = self._prepare_inputs_open_cum(
                third_person, wrist, padding_mask, axis_labels=axis_labels,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
        self._last_prep_time = time.perf_counter() - t0

        outputs = self.qwen(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (B, S, H)

        vision_end_id = self.processor.vision_end_token_id
        input_ids = inputs["input_ids"]  # (B, S)

        rewards = torch.zeros(B, self.num_preferences, device=device,
                              dtype=last_hidden.dtype)
        for b in range(B):
            positions = (input_ids[b] == vision_end_id).nonzero(as_tuple=True)[0]
            # Each frame contributes 2 <|vision_end|> tokens (3rd-person, wrist);
            # the second one closes the pair, so take indices 1, 3, 5, ...
            pair_end_positions = positions[1::2]
            if len(pair_end_positions) == 0:
                continue
            pooled = last_hidden[b, pair_end_positions].float()  # (n_frames, hidden)
            per_frame_rewards = self._apply_reward_heads(pooled)  # (n_frames, K)
            rewards[b] = per_frame_rewards.sum(dim=0).to(rewards.dtype)

        rewards = rewards.float()
        if self.reward_sigmoid:
            rewards = torch.sigmoid(rewards)
        return rewards

    def _forward_discounted(self, third_person, wrist, padding_mask, axis_labels, device):
        """Each frame pair scored independently, then summed per trajectory."""
        B = third_person.shape[0]

        t0 = time.perf_counter()
        with torch.no_grad():
            inputs, frame_counts = self._prepare_inputs_discounted(
                third_person, wrist, padding_mask, axis_labels=axis_labels,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
        self._last_prep_time = time.perf_counter() - t0

        pooled = self._extract_pooled(inputs, device)  # (total_frames, hidden)
        per_frame_rewards = self._apply_reward_heads(pooled)  # (total_frames, K)

        # Sum per-frame rewards for each trajectory
        rewards = torch.zeros(B, self.num_preferences, device=device)
        offset = 0
        for b, n_frames in enumerate(frame_counts):
            rewards[b] = per_frame_rewards[offset:offset + n_frames].sum(dim=0)
            offset += n_frames

        if self.reward_sigmoid:
            rewards = torch.sigmoid(rewards)
        return rewards

    # ------------------------------------------------------------------
    # Per-frame inference (for visualization / paper figures)
    # ------------------------------------------------------------------

    def forward_per_frame(
        self,
        third_person: torch.Tensor,
        wrist: torch.Tensor,
        padding_mask: torch.Tensor = None,
        axis_labels: list[str] | None = None,
    ) -> torch.Tensor:
        """Per-pair reward predictions (no temporal sum).

        Only defined for discounted and open_cum modes — those are the only
        variants whose final score is a sum of per-pair scores. Returns
        (B, T, K) with NaN at padded positions.
        """
        if not (self.discounted or self.open_cum):
            raise NotImplementedError(
                "forward_per_frame is only defined for discounted and open_cum modes"
            )

        device = next(self.parameters()).device
        with torch.no_grad():
            if third_person.dtype != torch.uint8:
                third_person = (third_person * 255).clamp(0, 255).to(torch.uint8)
            if wrist.dtype != torch.uint8:
                wrist = (wrist * 255).clamp(0, 255).to(torch.uint8)

        if self.open_cum:
            return self._forward_per_frame_open_cum(
                third_person, wrist, padding_mask, axis_labels, device
            )
        return self._forward_per_frame_discounted(
            third_person, wrist, padding_mask, axis_labels, device
        )

    def _forward_per_frame_discounted(self, third_person, wrist, padding_mask, axis_labels, device):
        B, T = third_person.shape[:2]
        with torch.no_grad():
            inputs, frame_counts = self._prepare_inputs_discounted(
                third_person, wrist, padding_mask, axis_labels=axis_labels,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}

        pooled = self._extract_pooled(inputs, device)            # (sum_T, hidden)
        per_frame = self._apply_reward_heads(pooled)              # (sum_T, K)

        out = torch.full((B, T, self.num_preferences), float("nan"),
                         device=device, dtype=per_frame.dtype)
        offset = 0
        for b, n in enumerate(frame_counts):
            out[b, :n] = per_frame[offset:offset + n]
            offset += n
        return out

    def _forward_per_frame_open_cum(self, third_person, wrist, padding_mask, axis_labels, device):
        B, T = third_person.shape[:2]
        with torch.no_grad():
            inputs, frame_counts = self._prepare_inputs_open_cum(
                third_person, wrist, padding_mask, axis_labels=axis_labels,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}

        outputs = self.qwen(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]  # (B, S, H)
        input_ids = inputs["input_ids"]
        vision_end_id = self.processor.vision_end_token_id

        out = torch.full((B, T, self.num_preferences), float("nan"),
                         device=device, dtype=last_hidden.dtype)
        for b in range(B):
            positions = (input_ids[b] == vision_end_id).nonzero(as_tuple=True)[0]
            pair_ends = positions[1::2]
            if len(pair_ends) == 0:
                continue
            pooled = last_hidden[b, pair_ends].float()
            per_frame = self._apply_reward_heads(pooled)  # (n_frames, K)
            n = min(int(frame_counts[b]), per_frame.shape[0])
            out[b, :n] = per_frame[:n].to(out.dtype)
        return out.float()

    # ------------------------------------------------------------------
    # Checkpointing helpers
    # ------------------------------------------------------------------

    def get_checkpoint_state_dict(self) -> dict:
        """Return only trainable params to keep checkpoints small."""
        trainable_names = {name for name, p in self.named_parameters()
                           if p.requires_grad}
        return {k: v for k, v in self.state_dict().items()
                if k in trainable_names}

    def load_checkpoint_state_dict(self, state_dict: dict):
        """Load checkpoint with strict=False (base weights come from pretrained)."""
        self.load_state_dict(state_dict, strict=False)
