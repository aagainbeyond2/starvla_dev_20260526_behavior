# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Cosmos-Predict2 World Model Interface.

Wraps NVIDIA Cosmos-Predict2 (diffusion-based Video2World model) as a
world-model backend for starVLA action prediction frameworks.

Architecture:
  - T5EncoderModel: text instruction → text embeddings [B, L_text, 1024]
  - AutoencoderKLWan (VAE): observation images → video latents [B, C, T, H, W]
  - CosmosTransformer3DModel (DiT): 28-layer transformer, hidden_dim=4096
    Takes noised latents + text embeddings → denoised latents
    We extract intermediate hidden states for action-conditioning.

Key difference from VLM wrappers:
  - No chat template / processor — uses T5 for text, VAE for vision
  - Hidden states come from DiT blocks, not autoregressive LM
  - The `build_inputs` interface provides a clean world-model API
    that does not depend on VLM-specific naming conventions.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _CosmoPredict2_Interface(nn.Module):
    """
    World model wrapper for Cosmos-Predict2 (diffusers-based).

    Exposes a compatible interface with VLM wrappers so that framework
    code can swap VLM ↔ WM transparently. The key methods are:
      - forward(**kwargs) → model outputs with hidden_states
      - build_inputs(images, instructions) → dict of tensors
      - generate(**kwargs) → video generation (optional)

    Representation extraction strategy:
      We run a single DiT forward pass at noise level σ≈0 and register
      forward hooks to capture intermediate block outputs. These are
      concatenated/pooled to produce a [B, N_tokens, hidden_dim] tensor
      that the action head can consume — analogous to VLM hidden_states.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get(
            "base_wm",
            config.framework.get("qwenvl", {}).get("base_vlm", "nvidia/Cosmos-Predict2-2B-Video2World"),
        )
        self.config = config

        # Import diffusers components
        from diffusers import (
            AutoencoderKLWan,
            CosmosTransformer3DModel,
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import T5EncoderModel, T5TokenizerFast

        logger.info(f"Loading Cosmos-Predict2 from {model_name}")

        # Load components individually (Pipeline is not nn.Module; split loading enables per-component freeze/finetune)
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        self.transformer = CosmosTransformer3DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        # Use diffusers' VideoProcessor for image/video preprocessing (resize, normalize, etc.)
        from diffusers.video_processor import VideoProcessor
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Observation resolution for the (single) composed multi-view frame.
        # Route A: multiple camera views are tiled into ONE frame (see _encode_images),
        # so they are encoded as a single video timestep instead of a fake 3-frame clip.
        # Configurable via world_model.obs_resolution: [height, width].
        # 480x832 is the Cosmos-Predict2 pretrained resolution; spatial dims must be
        # multiples of vae_scale_factor_spatial. Default keeps ~16:9 to save VRAM.
        obs_res = wm_cfg.get("obs_resolution", [320, 576])
        self._obs_height = self._round_to_multiple(int(obs_res[0]), self.vae_scale_factor_spatial)
        self._obs_width = self._round_to_multiple(int(obs_res[1]), self.vae_scale_factor_spatial)

        # How multiple camera views are fed to the DiT backbone:
        #   - "tile"  (default): tile all views side-by-side into ONE composed frame,
        #     encode it as a single observation -> features [B, N, D].
        #   - "batch": treat each view as its own batch element, encode all views in a
        #     single DiT pass, then concatenate the per-view features along the TOKEN
        #     axis -> features [B, num_views * N, D]. Token count (and VRAM) scale with
        #     the number of views, since each view is encoded at full obs_resolution.
        #   - "cosmos3_mosaic": Cosmos3-style fixed three-view canvas. The head view is
        #     224x224 on top; left/right wrist views are 112x112 side-by-side below.
        #     The resulting single frame is 336x224 (H x W) -> features [B, N, D].
        self._view_fusion = str(wm_cfg.get("view_fusion", "tile")).lower()
        if self._view_fusion not in ("tile", "batch", "cosmos3_mosaic"):
            raise ValueError(
                "world_model.view_fusion must be 'tile', 'batch', or "
                f"'cosmos3_mosaic', got '{self._view_fusion}'."
            )
        if self._view_fusion == "cosmos3_mosaic" and (
            self._obs_height,
            self._obs_width,
        ) != (336, 224):
            raise ValueError(
                "world_model.view_fusion='cosmos3_mosaic' requires "
                "world_model.obs_resolution: [336, 224] (H x W), got "
                f"[{self._obs_height}, {self._obs_width}]."
            )

        # Freeze VAE and text encoder by default
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Expose config compatible with framework expectations
        # DiT: 16 heads × 128 dim = 2048
        self._hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

        # Create a config-like object for the framework to read hidden_size
        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

        # Hook storage for intermediate features
        self._intermediate_features = []
        self._hooks = []

        # Which transformer blocks to extract features from (-1 = last)
        extract_layers = wm_cfg.get("extract_layers", [-1])
        self._extract_layers = extract_layers
        self._register_hooks()

    @property
    def model(self):
        """Compatibility shim: framework code accesses self.qwen_vl_interface.model.config.hidden_size"""
        class _ModelShim:
            pass
        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    @staticmethod
    def _round_to_multiple(value: int, multiple: int) -> int:
        """Round value to the nearest positive multiple of `multiple` (VAE-friendly)."""
        if multiple <= 1:
            return max(1, value)
        return max(multiple, int(round(value / multiple)) * multiple)

    @staticmethod
    def _letterbox(img: Image.Image, target_h: int, target_w: int) -> Image.Image:
        """Aspect-ratio-preserving resize + center pad (letterbox).

        Scales `img` to fit inside (target_h, target_w) without distorting the
        aspect ratio, then pads the remaining border with black. This replaces
        the previous force-resize-to-square behaviour that stretched the image.
        """
        img = img.convert("RGB")
        src_w, src_h = img.size
        if src_w == 0 or src_h == 0:
            return Image.new("RGB", (target_w, target_h), (0, 0, 0))
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return canvas

    def _compose_views_single_frame(
        self, views: Sequence[Image.Image], target_h: int, target_w: int
    ) -> Image.Image:
        """Tile multiple camera views horizontally into a SINGLE frame.

        Route A: instead of stacking views along the temporal axis (which the
        VAE would compress and treat as motion), we lay them out side-by-side in
        the spatial dimension so all views survive as one observation timestep.
        Each view is letterboxed into its tile to preserve aspect ratio.
        """
        views = [v for v in views]
        num_views = len(views)
        if num_views == 0:
            return Image.new("RGB", (target_w, target_h), (0, 0, 0))
        if num_views == 1:
            return self._letterbox(views[0], target_h, target_w)

        base_tile_w = target_w // num_views
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        x_offset = 0
        for i, view in enumerate(views):
            # Give the last tile any remainder pixels so the row fills target_w exactly.
            tile_w = base_tile_w if i < num_views - 1 else target_w - base_tile_w * (num_views - 1)
            tile = self._letterbox(view, target_h, tile_w)
            canvas.paste(tile, (x_offset, 0))
            x_offset += tile_w
        return canvas

    @classmethod
    def _compose_cosmos3_mosaic(cls, views: Sequence[Image.Image]) -> Image.Image:
        """Build the fixed Cosmos3-style Behavior-Skill three-view observation.

        Input order is the dataset/model contract:
          0. head camera       -> 224x224, top
          1. left wrist       -> 112x112, bottom-left
          2. right wrist      -> 112x112, bottom-right

        Each view is resized with aspect-ratio-preserving letterbox padding. The
        returned RGB canvas is exactly 224 pixels wide by 336 pixels high.
        """
        views = list(views)
        if len(views) != 3:
            raise ValueError(
                "view_fusion='cosmos3_mosaic' requires exactly 3 views in "
                "[head, left_wrist, right_wrist] order, "
                f"but received {len(views)}."
            )

        canvas = Image.new("RGB", (224, 336), (0, 0, 0))
        canvas.paste(cls._letterbox(views[0], 224, 224), (0, 0))
        canvas.paste(cls._letterbox(views[1], 112, 112), (0, 224))
        canvas.paste(cls._letterbox(views[2], 112, 112), (112, 224))
        return canvas

    def _register_hooks(self):
        """Register forward hooks on selected transformer blocks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        num_blocks = len(self.transformer.transformer_blocks)
        for layer_idx in self._extract_layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_blocks + layer_idx
            if 0 <= actual_idx < num_blocks:
                block = self.transformer.transformer_blocks[actual_idx]
                hook = block.register_forward_hook(self._capture_hook)
                self._hooks.append(hook)

    def _capture_hook(self, module, input, output):
        """Capture intermediate transformer block output."""
        # DiT block output is a tuple; first element is hidden_states
        if isinstance(output, tuple):
            self._intermediate_features.append(output[0])
        else:
            self._intermediate_features.append(output)

    @staticmethod
    def _merge_views(feat: torch.Tensor, num_views: int) -> torch.Tensor:
        """Concatenate per-view features along the token axis.

        In "batch" view fusion the DiT runs on a sample-major flattened batch of
        shape [B * num_views, N, D]. This regroups the views per sample and
        concatenates them along the token dimension -> [B, num_views * N, D].
        For num_views == 1 (tile fusion) the tensor is returned unchanged.
        """
        if num_views <= 1:
            return feat
        bv, n_tokens, hidden = feat.shape
        if bv % num_views != 0:
            raise ValueError(
                f"_merge_views expected a sample-major batch of B*num_views rows, "
                f"but got batch={bv} which is not divisible by num_views={num_views}."
            )
        b = bv // num_views
        # [B*V, N, D] -> [B, V, N, D] -> [B, V*N, D] (view order preserved)
        return feat.reshape(b, num_views * n_tokens, hidden)

    @classmethod
    def _to_merged_tokens(cls, feat: torch.Tensor, num_views: int) -> torch.Tensor:
        """Normalize a raw DiT block output into merged per-view token features.

        This is the single source of truth for turning a captured backbone block
        output into the `[B, num_views * N, D]` representation the action heads
        expect. It performs two steps:

          1. Token-flatten: a 5D DiT activation `[B_eff, C, T, H, W]` is permuted
             to `[B_eff, T*H*W, C]`. Already-tokenized `[B_eff, N, D]` outputs are
             passed through unchanged.
          2. View-merge: in `view_fusion="batch"` the sample-major batch
             `[B*V, N, D]` is regrouped to `[B, V*N, D]` (token-axis concat). For
             `num_views == 1` (tile fusion) this is a no-op.

        Any consumer of the backbone features (single-layer GR00T/OFT *and* the
        layerwise PI head) must route through this method so they all agree on the
        `[B, V*N, D]` contract.
        """
        if feat.dim() == 5:
            b_eff, c, t, h, w = feat.shape
            feat = feat.permute(0, 2, 3, 4, 1).reshape(b_eff, t * h * w, c)
        return cls._merge_views(feat, num_views)

    def _encode_text(self, instructions, max_length=512):
        """Encode text instructions using T5."""
        device = next(self.text_encoder.parameters()).device
        text_inputs = self.tokenizer(
            instructions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state  # [B, L, 1024]

        return text_embeds, text_inputs.attention_mask

    def _encode_images(self, images, num_frames=None):
        """Encode observation images through the VAE to get latent tokens.

        Three multi-view fusion strategies are supported (world_model.view_fusion):

        - "tile" (default): the per-sample camera views are tiled side-by-side into
          ONE composed frame (see _compose_views_single_frame) with aspect-preserving
          letterbox padding, then VAE-encoded as a single observation timestep. The
          views are NOT mistaken for temporal motion and there is no aspect distortion.
          Effective batch == B, num_views == 1.

        - "batch": each camera view is letterboxed to the full obs_resolution and
          encoded as its OWN batch element. The flattened batch is laid out
          sample-major ([s0v0, s0v1, ..., s1v0, ...]) so the per-view features can be
          regrouped and concatenated along the token axis later. Effective batch ==
          B * num_views.

        - "cosmos3_mosaic": exactly three views in [head, left_wrist, right_wrist]
          order are composed into one 336x224 frame: a 224x224 head view above two
          112x112 wrist views. Effective batch == B, num_views == 1.

        Args:
            images: List (batch) of per-sample views. Each element is a list of
                PIL Images (one per camera view), or a single PIL Image.
            num_frames: Kept for API compatibility; unused (every view/frame is
                encoded as exactly one observation timestep).

        Returns:
            latents: [B_eff, C, T_latent, H/8, W/8] video latent tensor (T_latent == 1),
                where B_eff == B (tile) or B * num_views (batch)
            cond_frame_counts: list[int], all 1 (single observation frame per element)
            num_views: int, number of views fused per sample (1 for "tile")
        """
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        height, width = self._obs_height, self._obs_width

        batch_videos = []
        if self._view_fusion == "batch":
            # Each view -> one batch element, letterboxed to full resolution.
            num_views = None
            for sample_imgs in images:
                if not isinstance(sample_imgs, (list, tuple)):
                    sample_imgs = [sample_imgs]
                sample_imgs = list(sample_imgs)
                if num_views is None:
                    num_views = len(sample_imgs)
                elif len(sample_imgs) != num_views:
                    raise ValueError(
                        "view_fusion='batch' requires a consistent number of views per "
                        f"sample; got {len(sample_imgs)} vs {num_views}."
                    )
                for view in sample_imgs:
                    frame = self._letterbox(view, height, width)
                    video_tensor = self.video_processor.preprocess_video([frame], height=height, width=width)
                    video_tensor = video_tensor.to(device=device, dtype=dtype)  # [1, C, 1, H, W]
                    batch_videos.append(video_tensor.squeeze(0))  # [C, 1, H, W]
            num_views = num_views or 1
        else:
            # Compose each sample's views into a single frame, then preprocess
            # (normalize to [-1, 1]). The composed frame is already at the target
            # resolution so preprocess_video does not resize/stretch it.
            num_views = 1
            for sample_imgs in images:
                if not isinstance(sample_imgs, (list, tuple)):
                    sample_imgs = [sample_imgs]

                if self._view_fusion == "cosmos3_mosaic":
                    composed = self._compose_cosmos3_mosaic(sample_imgs)
                else:
                    composed = self._compose_views_single_frame(sample_imgs, height, width)
                video_tensor = self.video_processor.preprocess_video([composed], height=height, width=width)
                video_tensor = video_tensor.to(device=device, dtype=dtype)  # [1, C, 1, H, W]
                batch_videos.append(video_tensor.squeeze(0))  # [C, 1, H, W]

        # Single observation frame per batch element.
        cond_frame_counts = [1] * len(batch_videos)

        # Stack to [B_eff, C, 1, H, W]
        video = torch.stack(batch_videos, dim=0)

        with torch.no_grad():
            # Single composed frame -> T_latent = (1-1)//temporal+1 = 1 latent timestep.
            latents = self.vae.encode(video).latent_dist.sample()  # [B, 16, 1, H/8, W/8]

        # Normalize latents (matches official pipeline: prepare_latents) # TODO check if this normalization is actually needed
        if self.vae.config.latents_mean is not None:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype=latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype=latents.dtype)
            ) 
            sigma_data = self.scheduler.config.sigma_data
            latents = (latents - latents_mean) / latents_std * sigma_data

        # latents: [B_eff, C, T_latent, H/8, W/8]; cond_frame_counts: list[int]; num_views: int
        return latents, cond_frame_counts, num_views

    def build_inputs(self, images, instructions, **kwargs):
        """Build inputs for the DiT world model.

        Instead of chat templates (VLM), we:
        1. Encode text with T5
        2. Encode images with VAE
        3. Package for the DiT forward pass

        Returns:
            dict with keys matching what forward() expects
        """
        assert len(images) == len(instructions)

        # Ensure encoders are on the right device
        device = next(self.transformer.parameters()).device
        # self.text_encoder.to(device)
        # self.vae.to(device)

        text_embeds, text_mask = self._encode_text(instructions)
        latents, cond_frame_counts, num_views = self._encode_images(images)

        # In "batch" view fusion each view is its own batch element, so replicate the
        # per-sample text conditioning to match (sample-major: s0,s0,...,s1,s1,...).
        if num_views > 1:
            text_embeds = text_embeds.repeat_interleave(num_views, dim=0)
            text_mask = text_mask.repeat_interleave(num_views, dim=0)

        # Offload T5 and VAE to CPU to free VRAM for the transformer
        # self.text_encoder.to("cpu")
        # self.vae.to("cpu")
        # torch.cuda.empty_cache()

        # For feature extraction, use timestep=0 (clean / minimal noise)
        batch_size = latents.shape[0]
        device = latents.device
        _, _, t_lat, h_lat, w_lat = latents.shape # B, C, T_latent, H/8, W/8
        timestep = torch.zeros(batch_size, device=device, dtype=torch.long)
        # condition_mask: tells DiT which latent frames are reliable conditions (=1) vs to-be-generated (=0).
        # In Video2World, this separates input frames from predicted future frames.
        # Here (action prediction, not generation), we set timestep=0 + condition_mask on real frames
        # so DiT runs a near-clean forward pass for feature extraction, not actual denoising.
        # in_channels = 16 (latents) + 1 (condition_mask) = 17
        # Shape: [B, 1, T_latent, H_latent, W_latent]
        condition_mask = latents.new_zeros(batch_size, 1, t_lat, h_lat, w_lat)
        for i, n_cond in enumerate(cond_frame_counts):
            # Map pixel-frame count to latent-frame count
            n_cond_latent = (n_cond - 1) // self.vae_scale_factor_temporal + 1
            condition_mask[i, :, :n_cond_latent] = 1.0

        # padding_mask: concat_padding_mask=True adds 1 more channel → 18 total
        # Shape: [1, 1, H_orig, W_orig] — all zeros = no padding
        # Will be resized to latent spatial dims by the transformer
        padding_mask = latents.new_zeros(1, 1, h_lat, w_lat)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "attention_mask": text_mask,
            "condition_mask": condition_mask,
            "padding_mask": padding_mask,
            "_is_wm_input": True,
            "_num_views": num_views,
        }

    def forward(self, **kwargs):
        """Forward pass through the DiT transformer.

        Runs a single-step forward to extract rich spatiotemporal features.
        Returns an output object with .hidden_states for compatibility.
        """
        is_wm = kwargs.pop("_is_wm_input", False)
        output_hidden_states = kwargs.pop("output_hidden_states", False)
        return_dict = kwargs.pop("return_dict", True)
        num_views = int(kwargs.pop("_num_views", 1))
        kwargs.pop("output_attentions", None)

        # Clear feature buffer
        self._intermediate_features.clear()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.transformer(
                hidden_states=kwargs["hidden_states"],
                timestep=kwargs["timestep"],
                encoder_hidden_states=kwargs["encoder_hidden_states"],
                condition_mask=kwargs.get("condition_mask", None),
                padding_mask=kwargs.get("padding_mask", None),
            )

        # Build hidden_states tuple from captured intermediate features.
        # `_to_merged_tokens` token-flattens (5D -> [B, N, D]) and merges per-view
        # tokens ([B*V, N, D] -> [B, V*N, D]) so the action head sees [B, V*N, D].
        extracted = [self._to_merged_tokens(feat, num_views) for feat in self._intermediate_features]

        # If no hooks fired (shouldn't happen), use transformer output
        if not extracted:
            out = dit_output.sample if hasattr(dit_output, "sample") else dit_output
            extracted.append(self._to_merged_tokens(out, num_views))

        # Build compatible output object
        class _WMOutput:
            def __init__(self, hidden_states_tuple, loss=None):
                self.hidden_states = hidden_states_tuple
                self.loss = loss

        return _WMOutput(hidden_states_tuple=tuple(extracted))

    def generate(self, **kwargs):
        """Video generation (for world-model imagination / planning).

        This builds the full Cosmos2VideoToWorldPipeline on-the-fly.
        Not used during standard VLA training, but useful for visualization
        and planning-based approaches.
        """
        from diffusers import Cosmos2VideoToWorldPipeline

        pipe = Cosmos2VideoToWorldPipeline(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            safety_checker=None,
        )
        return pipe(**kwargs)
