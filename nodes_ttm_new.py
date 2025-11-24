"""
New TTM node set for Wan 2.2 in ComfyUI.

This module defines a clean, self-contained set of nodes that:
- Use the official Wan 2.2 high/low noise UNets loaded via the standard UNet loader.
- Prepare TTM conditioning (start image/video, motion signal, motion mask).
- Run a two-phase MoE sampler (high-noise then low-noise) using ComfyUI's samplers.

The goal is to keep this file independent from the legacy ttm_sampler/ttm_conditioning
implementations so we can iterate safely.
"""

from __future__ import annotations

from typing import Dict, Any

import torch

import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management as mm
import latent_preview
from comfy.samplers import KSAMPLER
import nodes
from .flow_scheduler import FLOW_SCHEDULER_NAME, sample_flowmatch


class WanTTM_ModelFromUNet:
    """Wrap two Wan2.2 UNets (high/low noise) into MoE-ready models.

    Inputs are expected to be the outputs of the official Wan2.2 UNet loader nodes
    (i.e. ComfyUI "MODEL"-like objects with get_model_object("model_sampling") and
    get_model_object("diffusion_model")).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_unet": ("MODEL", {"tooltip": "Wan2.2 high-noise UNet (14B I2V/T2V)."}),
                "low_unet": ("MODEL", {"tooltip": "Wan2.2 low-noise UNet (14B I2V/T2V)."}),
                "sigma_shift": ("FLOAT", {"default": 5.0, "min": -50.0, "max": 50.0, "step": 0.1}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "high_steps": ("INT", {"default": 6, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "INT",)
    RETURN_NAMES = ("model_high", "model_low", "switch_step",)
    FUNCTION = "execute"
    CATEGORY = "TTM/Wan2.2"

    @classmethod
    def execute(
        cls,
        high_unet,
        low_unet,
        sigma_shift: float,
        steps: int,
        high_steps: int,
    ):
        # Clone models so downstream nodes can safely mutate options
        model_high = high_unet.clone()
        model_low = low_unet.clone()

        # Configure Flow Matching shift using the same helper pattern as your old set_shift
        sampling_high = model_high.get_model_object("model_sampling")
        if sampling_high is not None and hasattr(sampling_high, "set_parameters"):
            sampling_high.set_parameters(shift=sigma_shift, multiplier=1000)
            model_high.add_object_patch("model_sampling", sampling_high)

        sampling_low = model_low.get_model_object("model_sampling")
        if sampling_low is not None and hasattr(sampling_low, "set_parameters"):
            sampling_low.set_parameters(shift=sigma_shift, multiplier=1000)
            model_low.add_object_patch("model_sampling", sampling_low)

        switch_step = min(max(high_steps, 1), steps)
        return (model_high, model_low, switch_step)


class WanTTM_Conditioning:
    """Prepare TTM-specific conditioning and an initial latent.

    This node intentionally keeps the conditioning format compatible with the
    official Wan2.2 workflows: you pass in positive/negative conditionings
    from the standard text encoder nodes, and we simply forward them.

    TTM-specific tensors (reference motion latent + mask + step range) are
    provided out-of-band via a dict so the sampler can consume them explicitly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {}),
                "start_image": ("IMAGE", {"tooltip": "Start frame (I2V) or first frame (V2V)."}),
                "motion_signal_video": ("IMAGE", {"tooltip": "Motion signal video (T,H,W,C or B,T,H,W,C)."}),
                "motion_signal_mask": ("IMAGE", {"tooltip": "Mask video aligned with motion_signal_video (B,H,W,C) - use grayscale or binary images."}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "target_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Output width in pixels (multiple of 8)."}),
                "target_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8, "tooltip": "Output height in pixels (multiple of 8)."}),
                "num_frames": ("INT", {"default": 21, "min": 1, "max": 512, "step": 1}),
                "ttm_start_step": ("INT", {"default": 3, "min": 0, "max": 255, "step": 1}),
                "ttm_end_step": ("INT", {"default": 7, "min": 1, "max": 255, "step": 1}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Values >= threshold are treated as motion (1.0)."}),
                "mask_dilate_px": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1, "tooltip": "Dilate mask by N latent pixels using max-pool to increase coverage."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "DICT",)
    RETURN_NAMES = (
        "high_positive", "high_negative",
        "low_positive", "low_negative",
        "latent", "ttm_params",
    )
    FUNCTION = "execute"
    CATEGORY = "TTM/Wan2.2"

    @classmethod
    def execute(
        cls,
        vae,
        start_image,
        motion_signal_video,
        motion_signal_mask,
        positive,
        negative,
        target_width: int,
        target_height: int,
        num_frames: int,
        ttm_start_step: int,
        ttm_end_step: int,
        mask_threshold: float,
        mask_dilate_px: int,
    ):
        device = mm.get_torch_device()

        target_width_orig = target_width
        target_height_orig = target_height

        def ensure_batched(video: torch.Tensor, label: str) -> torch.Tensor:
            if video.ndim == 4:
                return video.unsqueeze(0)
            if video.ndim == 5:
                return video
            raise ValueError(f"{label} must be 4D or 5D tensor, got {video.ndim}D")

        # --- Encode motion signal video -------------------------------------------------
        motion_video = ensure_batched(motion_signal_video, "motion_signal_video")
        print(f"[WanTTM Conditioning] motion_signal_video input shape: {tuple(motion_signal_video.shape)}")

        T = motion_video.shape[1]
        # Encode framewise: [B,T,H,W,C] -> [B*T,H,W,C]
        motion_flat = motion_video.reshape(-1, *motion_video.shape[-3:])
        print(f"[WanTTM Conditioning] motion tensor to VAE (BT,H,W,C): {tuple(motion_flat.shape)}")
        motion_encoded = vae.encode(motion_flat)
        if isinstance(motion_encoded, dict):
            motion_latent = motion_encoded["samples"]
        else:
            motion_latent = motion_encoded
        print(f"[WanTTM Conditioning] motion_latent after encode shape: {tuple(motion_latent.shape)}")
        
        # Handle both 4D [B*T,C,h,w] and 5D [B,C,T,h,w] returns
        B = motion_video.shape[0]
        if motion_latent.dim() == 5:
            # Already [B,C,T,h,w] or [1,C,T,h,w]
            h, w = motion_latent.shape[3:]
        elif motion_latent.dim() == 4:
            # [B*T,C,h,w] -> [B,T,C,h,w] -> [B,C,T,h,w]
            C = motion_latent.shape[1]
            h, w = motion_latent.shape[2:]
            motion_latent = motion_latent.reshape(B, T, C, h, w).permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(f"Unexpected motion_latent dim={motion_latent.dim()}, shape={motion_latent.shape}")

        # Temporal length in latent space (Wan uses stride 4 in time)
        latent_frames = (num_frames - 1) // 4 + 1
        if motion_latent.shape[2] > latent_frames:
            motion_latent = motion_latent[:, :, :latent_frames]
        elif motion_latent.shape[2] < latent_frames:
            pad = latent_frames - motion_latent.shape[2]
            motion_latent = torch.cat([motion_latent, motion_latent[:, :, :pad]], dim=2)

        # --- Encode start_image to get initial latent -----------------------------------
        start = ensure_batched(start_image, "start_image")
        print(f"[WanTTM Conditioning] start_image input shape: {tuple(start_image.shape)}")
        start_first = start[:, :1]
        start_flat = start_first.reshape(-1, *start_first.shape[-3:])
        print(f"[WanTTM Conditioning] start tensor to VAE (1,H,W,C): {tuple(start_flat.shape)}")
        start_encoded = vae.encode(start_flat)
        if isinstance(start_encoded, dict):
            start_latent = start_encoded["samples"]
        else:
            start_latent = start_encoded
        print(f"[WanTTM Conditioning] start_latent after encode shape: {tuple(start_latent.shape)}")
        
        # Handle both 4D [1,C,h,w] and 5D [1,C,1,h,w] returns
        if start_latent.dim() == 5:
            # Already has time dimension
            start_latent = start_latent
        elif start_latent.dim() == 4:
            # [1,C,h,w] -> [1,C,1,h,w]
            start_latent = start_latent.unsqueeze(2)
        else:
            raise ValueError(f"Unexpected start_latent dim={start_latent.dim()}, shape={start_latent.shape}")

        # Allocate latent: [B,16,latent_frames,h,w]
        batch_size = start_latent.shape[0]
        
        # TTM approach: Initialize with start_image for all frames
        # This provides a clean base for background regions
        latent = start_latent.to(device).expand(-1, -1, latent_frames, -1, -1).clone()
        
        print(f"[WanTTM Conditioning] Latent shape: {latent.shape} (all frames from start_image)")
        print(f"[WanTTM Conditioning] Note: TTM will replace frames 2-21 with noisy motion signal during sampling")

        # --- Process motion mask --------------------------------------------------------
        # motion_signal_mask: IMAGE [B,H,W,C] - convert to grayscale mask [T,H,W]
        print(f"[WanTTM Conditioning] motion_signal_mask input shape: {tuple(motion_signal_mask.shape)}")
        
        if motion_signal_mask.ndim == 4:
            # [B,H,W,C] -> grayscale [B,H,W]
            # Take max across channels or average to get mask intensity
            if motion_signal_mask.shape[-1] == 1:
                mask_t = motion_signal_mask.squeeze(-1)  # Already single channel
            else:
                # Take max across RGB channels (white regions in any channel)
                mask_t = motion_signal_mask.max(dim=-1)[0]
        elif motion_signal_mask.ndim == 3:
            # [B,H,W] - already grayscale
            mask_t = motion_signal_mask
        elif motion_signal_mask.ndim == 2:
            # [H,W] - single frame
            mask_t = motion_signal_mask.unsqueeze(0)
        else:
            raise ValueError(f"motion_signal_mask must be 2D/3D/4D IMAGE tensor, got {motion_signal_mask.ndim}D")
        
        mask_t = mask_t.to(torch.float32)
        if mask_t.max() > 1.0:
            mask_t = mask_t / 255.0
        mask_t = mask_t.clamp_(0.0, 1.0)
        print(f"[WanTTM Conditioning] mask_t shape after conversion: {tuple(mask_t.shape)} (should be [T,H,W])")

        # Temporal downsampling to match VAE latent frames
        # Wan VAE uses temporal stride 4: frame 0, then every 4th frame starting from 1
        # This matches TTM's convert_rgb_mask_to_latent_mask logic
        mask_T = mask_t.shape[0]
        vae_temporal_stride = 4
        
        # Sample frames: [0], [1], [5], [9], [13], ...
        mask0 = mask_t[0:1]  # First frame [1,H,W]
        if mask_T > 1:
            # Sample every vae_temporal_stride frames starting from frame 1
            mask_rest_indices = list(range(1, mask_T, vae_temporal_stride))
            if len(mask_rest_indices) > 0:
                mask_rest = mask_t[mask_rest_indices]  # [T'-1,H,W]
                mask_sampled = torch.cat([mask0, mask_rest], dim=0)  # [T',H,W]
            else:
                mask_sampled = mask0
        else:
            mask_sampled = mask0
        
        print(f"[WanTTM Conditioning] mask_t original frames: {mask_T}, sampled frames: {mask_sampled.shape[0]}, target latent_frames: {latent_frames}")
        
        # Check if mask is dynamic (changes across frames)
        if mask_T > 1:
            first_frame = mask_t[0]
            is_static = all(torch.allclose(mask_t[i], first_frame, atol=1e-6) for i in range(1, min(5, mask_T)))
            if is_static:
                print(f"[WanTTM Conditioning] WARNING: Mask appears to be STATIC (all frames identical)!")
                print(f"[WanTTM Conditioning] For proper TTM, mask should follow object motion across frames.")
            else:
                # Show mask statistics per frame to verify it's dynamic
                mask_means = [mask_t[i].mean().item() for i in range(0, min(10, mask_T), max(1, mask_T//10))]
                print(f"[WanTTM Conditioning] Mask is DYNAMIC - sample frame means: {[f'{m:.3f}' for m in mask_means]}")
        elif mask_T == 1:
            print(f"[WanTTM Conditioning] WARNING: Only 1 mask frame provided - mask will be static!")
        
        # Adjust to match latent_frames exactly
        mask_sampled_T = mask_sampled.shape[0]
        if mask_sampled_T > latent_frames:
            mask_sampled = mask_sampled[:latent_frames]
        elif mask_sampled_T < latent_frames:
            # Repeat last frames to reach target length
            pad = latent_frames - mask_sampled_T
            mask_sampled = torch.cat([mask_sampled, mask_sampled[-1:].repeat(pad, 1, 1)], dim=0)
        
        # Resize spatially to latent resolution [T,H,W] -> [T,1,H,W] -> [T,1,h,w]
        mask_t = mask_sampled.unsqueeze(1)  # [T,1,H,W]
        mask_t = torch.nn.functional.interpolate(
            mask_t,
            size=(h, w),
            mode="nearest",
        )  # [T,1,h,w]
        if mask_dilate_px > 0:
            pad = mask_dilate_px
            mask_flat = mask_t.view(-1, 1, h, w)
            mask_flat = torch.nn.functional.max_pool2d(
                torch.nn.functional.pad(mask_flat, (pad, pad, pad, pad), mode="replicate"),
                kernel_size=2 * mask_dilate_px + 1,
                stride=1,
            )
            mask_t = mask_flat.view(-1, 1, h, w)
        mask_t = (mask_t >= mask_threshold).float()
        
        print(f"[WanTTM Conditioning] Final mask shape: {tuple(mask_t.shape)}")

        # Final mask format [T,1,h,w]
        ttm_mask = mask_t

        # Prepare TTM params dict for the sampler
        # motion_latent: [B,C,T,h,w] -> reference: take batch 0
        ttm_reference_latents = motion_latent[0].detach()
        ttm_params = {
            "ttm_reference_latents": ttm_reference_latents,  # [C,T,h,w]
            "ttm_mask": ttm_mask.detach(),                  # [T,1,h,w]
            "ttm_start_step": int(ttm_start_step),
            "ttm_end_step": int(ttm_end_step),
        }

        # Use ComfyUI Wan's I2V mechanism: concat_latent_image + concat_mask
        # This is the standard way to lock first frame in Wan models
        
        # concat_latent_image: VAE latent of start_image expanded to all frames
        concat_latent_image = start_latent.to(device).expand(-1, -1, latent_frames, -1, -1).clone()
        
        # concat_mask: 0.0=locked (first frame), 1.0=free (rest frames)
        # Format must match ComfyUI Wan's expectation: [B, 1, T, H, W]
        concat_mask = torch.ones(1, 1, latent_frames, h, w, device=device, dtype=start_latent.dtype)
        concat_mask[:, :, :1] = 0.0  # Lock first frame
        
        # Add I2V conditioning
        conditioning_data = {
            "concat_latent_image": concat_latent_image.cpu(),
            "concat_mask": concat_mask.cpu()
        }
        high_positive = nodes.node_helpers.conditioning_set_values(positive, conditioning_data)
        high_negative = nodes.node_helpers.conditioning_set_values(negative, conditioning_data)
        low_positive = nodes.node_helpers.conditioning_set_values(positive, conditioning_data)
        low_negative = nodes.node_helpers.conditioning_set_values(negative, conditioning_data)
        
        latent_dict = {"samples": latent.cpu()}
        
        motion_mask_preview = mask_t.mean().item()
        print(f"[WanTTM Conditioning] Mask coverage (mean of latent mask): {motion_mask_preview:.4f}")
        print(f"[WanTTM Conditioning] Added I2V conditioning: concat_latent_image + concat_mask")
        print(f"[WanTTM Conditioning] concat_mask: frame1=0.0 (locked), frames2-21=1.0 (free)")
        
        return (
            high_positive, high_negative,
            low_positive, low_negative,
            latent_dict, ttm_params,
        )


class WanTTM_Sampler:
    """Two-phase MoE sampler with optional TTM support.

    Supports:
    - Standard Wan2.2 I2V (MoE sampling without TTM)
    - TTM-enhanced I2V (with temporal reference blending)
    
    Based on official WanMoeKSampler pattern.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL", {}),
                "model_low": ("MODEL", {}),
                "high_positive": ("CONDITIONING", {}),
                "high_negative": ("CONDITIONING", {}),
                "low_positive": ("CONDITIONING", {}),
                "low_negative": ("CONDITIONING", {}),
                "latent": ("LATENT", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "switch_step": ("INT", {"default": 6, "min": 1, "max": 128, "step": 1}),
                "cfg_high": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 25.0, "step": 0.1}),
                "cfg_low": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 25.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": FLOW_SCHEDULER_NAME}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "ttm_params": ("DICT", {}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "execute"
    CATEGORY = "TTM/Wan2.2"

    @classmethod
    def execute(
        cls,
        model_high,
        model_low,
        high_positive,
        high_negative,
        low_positive,
        low_negative,
        latent,
        seed: int,
        steps: int,
        switch_step: int,
        cfg_high: float,
        cfg_low: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        ttm_params: Dict[str, Any] = None,
    ):
        """Dual Clock Denoising with MoE support.
        
        Based on TTM paper: https://github.com/time-to-move/TTM
        - tweak_index (ttm_start_step): when background starts denoising
        - tstrong_index (ttm_end_step): when motion region starts denoising
        - In between: motion region is replaced with noisy motion signal each step
        """
        device = mm.get_torch_device()
        latent_image = latent["samples"].to(device)
        
        print(f"[WanTTM] Initial latent: {latent_image.shape}")

        sigmas = comfy.samplers.calculate_sigmas(
            model_high.get_model_object("model_sampling"),
            scheduler,
            steps
        ).to(device=device, dtype=latent_image.dtype)
        use_flow_sampler = scheduler == FLOW_SCHEDULER_NAME

        # Prepare TTM data (optional)
        if ttm_params is None:
            ttm_params = {}
        
        ttm_reference_latents = ttm_params.get("ttm_reference_latents")
        ttm_mask = ttm_params.get("ttm_mask")
        tweak_index = int(ttm_params.get("ttm_start_step", 0))  # When background starts
        tstrong_index = int(ttm_params.get("ttm_end_step", 0))  # When motion starts
        
        # Check if TTM is enabled
        ttm_enabled = (ttm_reference_latents is not None and 
                      ttm_mask is not None and 
                      tstrong_index > tweak_index)
        
        preview_callback = latent_preview.prepare_callback(model_high, steps)

        if ttm_enabled:
            print(f"[WanTTM] TTM Dual Clock enabled: tweak={tweak_index}, tstrong={tstrong_index}")
            
            # Prepare motion signal reference latents
            ref = ttm_reference_latents.to(device)
            if ref.dim() == 4:
                ref = ref.unsqueeze(0)
            
            ref = comfy.sample.fix_empty_latent_channels(model_high, ref)
            latent_format = model_high.get_model_object("latent_format")
            if latent_format is not None:
                ref = latent_format.process_in(ref)
            
            print(f"[WanTTM] Motion signal ref: {ref.shape}, range: [{ref.min():.3f}, {ref.max():.3f}]")
            
            # Prepare masks
            mask = ttm_mask.to(device)
            print(f"[WanTTM] Input ttm_mask shape: {mask.shape}")
            
            # Convert [T,1,h,w] -> [1,1,T,h,w] to match ref [B,C,T,h,w]
            if mask.dim() == 4:
                # [T,1,h,w] -> [1,T,1,h,w] -> [1,1,T,h,w]
                mask = mask.unsqueeze(0).movedim(1, 2)
            print(f"[WanTTM] Mask after reshape: {mask.shape}")
            
            # Expand to match ref dimensions [B,C,T,h,w]
            motion_mask = mask.expand(ref.shape[0], ref.shape[1], -1, -1, -1)
            background_mask = 1.0 - motion_mask
            
            # Debug mask info
            mask_nonzero = (motion_mask > 0.5).float().mean()
            print(f"[WanTTM] Motion mask shape: {motion_mask.shape}")
            print(f"[WanTTM] Motion mask: mean={motion_mask.mean():.3f}, range=[{motion_mask.min():.3f}, {motion_mask.max():.3f}]")
            print(f"[WanTTM] Motion coverage (>0.5): {mask_nonzero:.1%}")
            
            # Check if mask varies over time
            mask_per_frame = motion_mask[0, 0].mean(dim=(1, 2))
            print(f"[WanTTM] Mask mean per frame (all {len(mask_per_frame)} frames):")
            for i in range(0, len(mask_per_frame), max(1, len(mask_per_frame)//5)):
                print(f"[WanTTM]   Frame {i:2d}: {mask_per_frame[i]:.4f}")
            
            # Verify mask is dynamic
            if len(mask_per_frame) > 1:
                mask_variance = mask_per_frame.std().item()
                if mask_variance < 1e-6:
                    print(f"[WanTTM] WARNING: Mask is STATIC across frames (variance={mask_variance:.2e})")
            
            # Generate fixed noise for motion signal
            fixed_noise = comfy.sample.prepare_noise(ref, seed + 1, None).to(device=device, dtype=ref.dtype)

            sampling_high = model_high.get_model_object("model_sampling")

            def noisy_reference_at_step(step_index: int):
                if step_index < 0:
                    step_index = 0
                if step_index >= len(sigmas):
                    return ref
                sigma_value = sigmas[step_index]
                sigma_value = sigma_value.view(1)
                noise_clone = fixed_noise.clone()
                return sampling_high.noise_scaling(sigma_value, noise_clone, ref, max_denoise=False)
            
            # TTM Original Initialization:
            # ALL frames = motion_signal + noise (including frame 1)
            # Frame 1 will be locked by I2V concat_mask mechanism in conditioning
            if tweak_index >= 0 and tweak_index < len(sigmas):
                tweak_sigma = sigmas[tweak_index].item()
                print(f"[WanTTM] Initializing at tweak_index={tweak_index}, sigma={tweak_sigma:.3f}")

                # TTM original: entire latent = ref_latents + fixed_noise (Flow-aware)
                latent_image = noisy_reference_at_step(tweak_index)
                
                print(f"[WanTTM]   ALL frames initialized as: motion_signal + noise (sigma={tweak_sigma:.3f})")
                print(f"[WanTTM]   Frame 1: will be locked by I2V concat_mask mechanism")
                print(f"[WanTTM]   Frames 2-21: will denoise from motion_signal")
                print(f"[WanTTM]   During denoising [tweak={tweak_index}, tstrong={tstrong_index}):")
                print(f"[WanTTM]     - Background: denoises from motion_signal background")
                print(f"[WanTTM]     - Motion: replaced with noisy motion_signal (TTM dual clock)")
            else:
                print(f"[WanTTM] Warning: tweak_index={tweak_index} out of range, using original latent")
            
        else:
            print(f"[WanTTM] TTM disabled, running standard MoE sampling")
            ref = None
            fixed_noise = None
            motion_mask = None
            background_mask = None
            sampling_high = None
        
        # Prepare noise ONCE for both phases (critical!)
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds).to(device)
        noise_mask = latent.get("noise_mask")
        
        # Determine MoE switching point
        switch_step_clamped = min(max(switch_step, 0), steps - 1)
        has_phase_2 = switch_step_clamped < steps - 1
        
        print(f"[WanTTM] MoE switching at step {switch_step_clamped}")
        print(f"[WanTTM] Phase 1: HIGH model [0, {switch_step_clamped}], CFG={cfg_high}")
        if has_phase_2:
            print(f"[WanTTM] Phase 2: LOW model [{switch_step_clamped}, {steps}], CFG={cfg_low}")
        
        # TTM Dual Clock + preview callback
        step_offset = [0]  # Track global step across phases

        def sampler_callback(step, x0, x, total_steps_phase):
            global_step = step_offset[0] + step

            if ttm_enabled and tweak_index <= global_step < tstrong_index:
                next_step = min(global_step + 1, len(sigmas) - 1)
                noisy_ref = noisy_reference_at_step(next_step)
                x.mul_(background_mask).add_(noisy_ref * motion_mask)

                if global_step == tweak_index:
                    print(f"[WanTTM] TTM Dual Clock active: steps [{tweak_index}, {tstrong_index})")
                    print(f"[WanTTM]   Background: normal denoising from motion_signal")
                    print(f"[WanTTM]   Motion region: replaced with noisy motion_signal")
                    print(f"[WanTTM]   (Frame 1 locked by I2V concat_mask)")

            if preview_callback is not None:
                preview_callback(global_step, x0, x, steps)
        
        def run_flow_phase(model, cfg_scale, pos_cond, neg_cond, sigmas_tensor, disable_noise_flag, force_zero_end_flag, latent_in, phase_steps):
            phase_cb = None
            if sampler_callback is not None:
                phase_cb = lambda data: sampler_callback(data["i"], data["denoised"], data["x"], phase_steps)
            return sample_flowmatch(
                model=model,
                noise=noise,
                cfg=cfg_scale,
                sigmas=sigmas_tensor,
                positive=pos_cond,
                negative=neg_cond,
                latent_image=latent_in,
                disable_noise=disable_noise_flag,
                force_zero_end=force_zero_end_flag,
                noise_mask=noise_mask,
                callback=phase_cb,
                disable_pbar=False,
                seed=seed,
            )

        # Phase 1
        if use_flow_sampler:
            sigmas_high = sigmas[:switch_step_clamped + 1].clone()
            if not has_phase_2:
                sigmas_high = sigmas.clone()
            total_steps_phase = len(sigmas_high) - 1
            latent_image = run_flow_phase(
                model_high,
                cfg_high,
                high_positive,
                high_negative,
                sigmas_high,
                disable_noise_flag=has_phase_2,
                force_zero_end_flag=has_phase_2,
                latent_in=latent_image,
                phase_steps=total_steps_phase,
            )
        else:
            latent_image = comfy.sample.fix_empty_latent_channels(model_high, latent_image)
            latent_image = comfy.sample.sample(
                model_high, noise, steps, cfg_high, sampler_name, scheduler,
                high_positive, high_negative, latent_image,
                denoise=denoise,
                disable_noise=has_phase_2,
                start_step=0,
                last_step=switch_step_clamped,
                force_full_denoise=has_phase_2,
                noise_mask=noise_mask,
                callback=sampler_callback,
                disable_pbar=False,
                seed=seed
            )

        # Phase 2
        if has_phase_2:
            step_offset[0] = switch_step_clamped
            if use_flow_sampler:
                sigmas_low = sigmas[switch_step_clamped:].clone()
                total_steps_low = len(sigmas_low) - 1
                latent_image = run_flow_phase(
                    model_low,
                    cfg_low,
                    low_positive,
                    low_negative,
                    sigmas_low,
                    disable_noise_flag=True,
                    force_zero_end_flag=True,
                    latent_in=latent_image,
                    phase_steps=total_steps_low,
                )
            else:
                latent_image = comfy.sample.fix_empty_latent_channels(model_low, latent_image)
                latent_image = comfy.sample.sample(
                    model_low, noise, steps, cfg_low, sampler_name, scheduler,
                    low_positive, low_negative, latent_image,
                    denoise=denoise,
                    disable_noise=True,
                    start_step=switch_step_clamped,
                    last_step=steps,
                    force_full_denoise=False,
                    noise_mask=noise_mask,
                    callback=sampler_callback,
                    disable_pbar=False,
                    seed=seed
                )
        
        out = latent.copy()
        out["samples"] = latent_image
        print(f"[WanTTM] Sampling complete. Output shape: {latent_image.shape}")
        print(f"[WanTTM] Note: First frame is locked by I2V concat_mask mechanism")
        return (out,)
class WanTTM_Decode:
    """Debug wrapper for VAE decode with detailed logging.
    
    NOTE: You can use ComfyUI's standard VAEDecode node instead.
    This node is mainly for debugging and adds extra safety checks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {}),
                "samples": ("LATENT", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video_frames",)
    FUNCTION = "execute"
    CATEGORY = "TTM/Wan2.2"

    @classmethod
    def execute(cls, vae, samples):
        x = samples["samples"]  # [B,C,T,H,W]
        device = mm.get_torch_device()
        x = x.to(device)
        
        print(f"[WanTTM Decode] Input latent: {x.shape}, range: [{x.min():.3f}, {x.max():.3f}]")
        
        # Just use standard VAE decode
        decoded = vae.decode(x)
        
        if isinstance(decoded, dict):
            decoded = decoded["samples"]
        
        print(f"[WanTTM Decode] VAE output raw: {decoded.shape}, range: [{decoded.min():.3f}, {decoded.max():.3f}]")
        
        # Squeeze batch dimension if present: [1, T, H, W, C] -> [T, H, W, C]
        if decoded.dim() == 5 and decoded.shape[0] == 1:
            decoded = decoded.squeeze(0)
        
        print(f"[WanTTM Decode] Final output: {decoded.shape}")
        
        return (decoded,)


NODE_CLASS_MAPPINGS = {
    "WanTTM_ModelFromUNet": WanTTM_ModelFromUNet,
    "WanTTM_Conditioning_New": WanTTM_Conditioning,
    "WanTTM_Sampler_New": WanTTM_Sampler,
    "WanTTM_Decode_New": WanTTM_Decode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanTTM_ModelFromUNet": "Wan2.2: MoE Model Setup",
    "WanTTM_Conditioning_New": "Wan2.2: TTM Conditioning",
    "WanTTM_Sampler_New": "Wan2.2: MoE Sampler (TTM Optional)",
    "WanTTM_Decode_New": "Wan2.2: Video Decode",
}
