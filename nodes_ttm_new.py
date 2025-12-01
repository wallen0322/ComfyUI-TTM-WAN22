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
from .flow_scheduler import (
    FLOW_SCHEDULER_NAME, 
    sample_flowmatch_ttm, 
    add_noise_at_sigma,
    generate_wan_sigmas,
)

# Fallback Wan VAE latent stats (taken from official WanVideoVAE implementations)
WAN_LATENT_STATS = {
    16: {
        "mean": torch.tensor(
            [
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
            ],
            dtype=torch.float32,
        ).view(1, 16, 1, 1, 1),
        "std": torch.tensor(
            [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
            ],
            dtype=torch.float32,
        ).view(1, 16, 1, 1, 1),
    },
    48: {
        "mean": torch.tensor(
            [
                -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
                -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
                -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
                -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
                -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
                0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
            ],
            dtype=torch.float32,
        ).view(1, 48, 1, 1, 1),
        "std": torch.tensor(
            [
                0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
                0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
                0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
                0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
                0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
                0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
            ],
            dtype=torch.float32,
        ).view(1, 48, 1, 1, 1),
    },
}


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

    @staticmethod
    def _resolve_latent_stats(vae):
        """Best-effort search for Wan VAE latent mean/std on any wrapped object."""
        def expand(obj):
            attr_names = (
                "config", "vae", "model", "inner_model", "first_stage_model",
                "wrapped_model", "pipeline", "_model",
            )
            for name in attr_names:
                child = getattr(obj, name, None)
                if child is not None:
                    yield child

        seen = set()
        queue = []

        def enqueue(obj):
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            queue.append(obj)

        enqueue(vae)
        enqueue(getattr(vae, "latent_format", None))

        def extract_stats(obj):
            mean = getattr(obj, "latents_mean", None)
            std = getattr(obj, "latents_std", None)
            inv_std = getattr(obj, "latents_inv_std", None)

            if mean is None:
                mean = getattr(obj, "mean", None)
            if std is None:
                std = getattr(obj, "std", None)
            if inv_std is None:
                inv_std = getattr(obj, "inv_std", None)

            if std is None and inv_std is not None:
                try:
                    std_tensor = torch.as_tensor(inv_std, dtype=torch.float32)
                    std = torch.where(std_tensor != 0, 1.0 / std_tensor, torch.ones_like(std_tensor))
                except Exception:
                    std = None

            return mean, std

        latents_mean = None
        latents_std = None

        while queue and (latents_mean is None or latents_std is None):
            current = queue.pop(0)
            candidate_mean, candidate_std = extract_stats(current)
            if latents_mean is None and candidate_mean is not None:
                latents_mean = candidate_mean
            if latents_std is None and candidate_std is not None:
                latents_std = candidate_std
            for child in expand(current):
                enqueue(child)

        if latents_mean is None or latents_std is None:
            latent_channels = (
                getattr(vae, "latent_channels", None)
                or getattr(getattr(vae, "model", None), "z_dim", None)
                or getattr(vae, "z_dim", None)
            )
            if latent_channels is None and hasattr(vae, "get_latent_channels"):
                try:
                    latent_channels = vae.get_latent_channels()
                except Exception:
                    latent_channels = None
            if latent_channels is not None:
                stats = WAN_LATENT_STATS.get(int(latent_channels))
                if stats is not None:
                    latents_mean = stats["mean"]
                    latents_std = stats["std"]

        return latents_mean, latents_std

    @classmethod
    def _normalize_reference_latents(cls, latents: torch.Tensor, vae) -> torch.Tensor:
        mean_values, std_values = cls._resolve_latent_stats(vae)
        if mean_values is None or std_values is None:
            print("[WanTTM Conditioning] WARNING: VAE latents_mean/std not found, skipping ref latent normalization.")
            return latents

        mean_tensor = torch.as_tensor(mean_values, dtype=latents.dtype, device=latents.device)
        std_tensor = torch.as_tensor(std_values, dtype=latents.dtype, device=latents.device)
        if mean_tensor.numel() != latents.shape[1] or std_tensor.numel() != latents.shape[1]:
            print("[WanTTM Conditioning] WARNING: VAE latent stats have unexpected size, skipping normalization.")
            return latents

        mean_tensor = mean_tensor.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        std_tensor = std_tensor.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        std_tensor = torch.where(std_tensor == 0, torch.ones_like(std_tensor), std_tensor)
        inv_std = torch.reciprocal(std_tensor)
        latents = (latents - mean_tensor) * inv_std
        latents._wan_stats = (mean_tensor, std_tensor)
        return latents

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
                "high_noise_lock_first_latent": ("BOOLEAN", {"default": True, "tooltip": "高噪阶段是否锁定第一个 latent 帧（I2V 首帧锁定）。关闭则高噪完全不锁 start_image。"}),
                "low_noise_use_start_image": ("BOOLEAN", {"default": True, "tooltip": "低噪阶段首帧是否也使用 start_image 作为 concat image；关闭则整段用中灰。"}),
                "low_noise_lock_first_frame": ("BOOLEAN", {"default": True, "tooltip": "低噪阶段首帧 concat_mask 是否为 0（锁定）。关闭则低噪所有帧 mask=1 完全自由生成。"}),
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
        high_noise_lock_first_latent: bool,
        low_noise_use_start_image: bool,
        low_noise_lock_first_frame: bool,
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
        # Official TTM: motion_signal_video is encoded to get ref_latents
        # The VAE has temporal stride 4, so T frames -> (T-1)//4 + 1 latent frames
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
            # Already [B,C,T',h,w] where T' is VAE-encoded temporal dimension
            h, w = motion_latent.shape[3:]
            T_encoded = motion_latent.shape[2]
        elif motion_latent.dim() == 4:
            # [B*T,C,h,w] -> [B,T,C,h,w] -> [B,C,T,h,w]
            # But VAE may have temporal stride, so T might be different
            C = motion_latent.shape[1]
            h, w = motion_latent.shape[2:]
            # VAE temporal stride: if input is T frames, output is (T-1)//4 + 1 frames
            T_encoded = (T - 1) // 4 + 1
            if motion_latent.shape[0] == B * T:
                # VAE didn't apply temporal stride, reshape normally
                motion_latent = motion_latent.reshape(B, T, C, h, w).permute(0, 2, 1, 3, 4).contiguous()
                T_encoded = T
            else:
                # VAE applied temporal stride, need to handle differently
                # Assume VAE output is [B*T_encoded, C, h, w]
                motion_latent = motion_latent.reshape(B, T_encoded, C, h, w).permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(f"Unexpected motion_latent dim={motion_latent.dim()}, shape={motion_latent.shape}")

        # Temporal length in latent space (Wan uses stride 4 in time)
        # This is the target number of latent frames for the output video
        latent_frames = (num_frames - 1) // 4 + 1
        print(f"[WanTTM Conditioning] motion_latent encoded frames: {T_encoded}, target latent_frames: {latent_frames}")
        
        # Align motion_latent to target latent_frames
        if motion_latent.shape[2] > latent_frames:
            # Trim to target length
            motion_latent = motion_latent[:, :, :latent_frames]
            print(f"[WanTTM Conditioning] Trimmed motion_latent from {T_encoded} to {latent_frames} frames")
        elif motion_latent.shape[2] < latent_frames:
            # Pad to target length (repeat last frame)
            pad = latent_frames - motion_latent.shape[2]
            motion_latent = torch.cat([motion_latent, motion_latent[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)
            print(f"[WanTTM Conditioning] Padded motion_latent from {T_encoded} to {latent_frames} frames")

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
        
        # Official TTM logic:
        # - Initial latent will be REPLACED in sampler with noisy ref_latents (from motion_signal_video)
        # - start_image is used via I2V concat_latent_image mechanism to lock first frame
        # - For now, create a placeholder with correct shape (will be replaced in sampler if TTM enabled)
        latent = start_latent.to(device).expand(-1, -1, latent_frames, -1, -1).clone()
        
        print(f"[WanTTM Conditioning] Latent shape: {latent.shape}")
        print(f"[WanTTM Conditioning] Note: If TTM enabled, initial latent will be replaced with noisy motion_signal in sampler")
        print(f"[WanTTM Conditioning] Start image locked via I2V concat_latent_image + concat_mask mechanism")

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
        # motion_latent: [B,C,T,h,w] -> reference: take batch 0, squeeze to [C,T,h,w]
        # NOTE: Don't normalize here - sampler will use latent_format.process_in for consistency
        ref_latents = motion_latent[:1].detach()
        ttm_reference_latents = ref_latents.squeeze(0).contiguous()

        # Ensure TTM reference and mask spatial resolution match the base latent.
        # WAN I2V in ComfyUI often uses higher spatial resolution for the start
        # image than the motion signal video. If we don't upsample the TTM
        # reference/mask to the latent resolution, TTM injection will fail with
        # shape mismatches when blending into the sampler's latent `x`.
        latent_h, latent_w = latent.shape[-2:]
        ref_h, ref_w = ttm_reference_latents.shape[-2:]
        mask_h, mask_w = ttm_mask.shape[-2:]

        if (ref_h, ref_w) != (latent_h, latent_w):
            import torch.nn.functional as F
            print(f"[WanTTM Conditioning] Upscaling TTM reference from {(ref_h, ref_w)} to {(latent_h, latent_w)}")
            # ttm_reference_latents: [C,T,h,w] -> [1,T,C,h,w] -> [T,C,h,w]
            C_ref, T_ref = ttm_reference_latents.shape[0], ttm_reference_latents.shape[1]
            ref_5d = ttm_reference_latents.unsqueeze(0)  # [1,C,T,h,w]
            ref_flat = ref_5d.permute(0, 2, 1, 3, 4).reshape(T_ref, C_ref, ref_h, ref_w)
            ref_flat = F.interpolate(ref_flat, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
            ref_5d = ref_flat.reshape(1, T_ref, C_ref, latent_h, latent_w).permute(0, 2, 1, 3, 4)
            ttm_reference_latents = ref_5d.squeeze(0).contiguous()

        if (mask_h, mask_w) != (latent_h, latent_w):
            import torch.nn.functional as F
            print(f"[WanTTM Conditioning] Upscaling TTM mask from {(mask_h, mask_w)} to {(latent_h, latent_w)}")
            T_mask = ttm_mask.shape[0]
            mask_flat = ttm_mask.reshape(T_mask, 1, mask_h, mask_w)
            mask_flat = F.interpolate(mask_flat, size=(latent_h, latent_w), mode="nearest")
            ttm_mask = mask_flat
            mask_h, mask_w = latent_h, latent_w

        print(f"[WanTTM Conditioning] Reference latents: {ttm_reference_latents.shape}")
        print(f"[WanTTM Conditioning] Reference range: [{ttm_reference_latents.min():.3f}, {ttm_reference_latents.max():.3f}]")
        print(f"[WanTTM Conditioning] Mask coverage (mean of latent mask): {ttm_mask.mean().item():.4f}")

        ttm_params = {
            "ttm_reference_latents": ttm_reference_latents,  # [C,T,h,w] (aligned to latent)
            "ttm_mask": ttm_mask.detach(),                   # [T,1,h,w] (aligned to latent)
            "ttm_start_step": int(ttm_start_step),
            "ttm_end_step": int(ttm_end_step),
        }

        # Use ComfyUI Wan's I2V mechanism: concat_latent_image + concat_mask
        # 高噪 / 低噪分开注入：
        #   - 高噪：整段都看到 start_image 的 latent（更强的起始约束）；
        #   - 低噪：只在首帧看到 start_image，后面帧看到的是 0.5 的灰底，避免整段在低噪被拉亮。

        # --- 高噪阶段 concat ---
        concat_latent_image_high = start_latent.to(device).expand(-1, -1, latent_frames, -1, -1).clone()

        # --- 低噪阶段 concat ---
        # 构造一段 image_low：
        #   - 如果 low_noise_use_start_image=True：第 0 帧 = start_image，其它帧 = 0.5 灰；
        #   - 如果为 False：整段都是 0.5 灰，只靠文本引导，完全不看 start_image。
        image_low = torch.ones(
            (num_frames, target_height_orig, target_width_orig, 3),
            device=device, dtype=start_image.dtype
        ) * 0.5
        if low_noise_use_start_image:
            # start 已经是保证 batch 维度的张量，这里只取第一帧
            image_low[0:1] = start[0, :1, :, :, :3]
        concat_latent_image_low = vae.encode(image_low[:, :, :, :3])

        # 处理 VAE 可能返回的 4D/5D 形状，统一成 [B,C,T,H,W]
        if isinstance(concat_latent_image_low, dict):
            concat_latent_image_low = concat_latent_image_low["samples"]
        if concat_latent_image_low.dim() == 4:
            # [T,C,H,W] or [1*T,C,H,W] 情况：压成 [1,C,T,H,W]
            if concat_latent_image_low.shape[0] == num_frames:
                concat_latent_image_low = concat_latent_image_low.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        # 若已是 [B,C,T,H,W] 则直接使用

        # --- concat_mask：0.0=锁定，1.0=自由 ---
        # 高噪：可选是否锁定第一个 latent 帧。
        concat_mask_high = torch.ones(1, 1, latent_frames, h, w, device=device, dtype=start_latent.dtype)
        if high_noise_lock_first_latent:
            # I2V 标准行为：第一个 latent 帧锁定到 start_image，其它帧自由生成。
            concat_mask_high[:, :, :1] = 0.0

        # 低噪：可选首帧是否继续锁定；
        #   - low_noise_lock_first_frame=True：跟高噪一样，0 帧锁定，其它帧自由；
        #   - False：所有帧 mask=1，真正意义上的“全程自由生成”。
        concat_mask_low = torch.ones(1, 1, latent_frames, h, w, device=device, dtype=start_latent.dtype)
        if low_noise_lock_first_frame:
            concat_mask_low[:, :, :1] = 0.0

        # Add I2V conditioning（高噪 / 低噪各自用自己的 concat_latent_image + concat_mask）
        conditioning_high = {
            "concat_latent_image": concat_latent_image_high.cpu(),
            "concat_mask": concat_mask_high.cpu()
        }
        conditioning_low = {
            "concat_latent_image": concat_latent_image_low.cpu(),
            "concat_mask": concat_mask_low.cpu()
        }
        high_positive = nodes.node_helpers.conditioning_set_values(positive, conditioning_high)
        high_negative = nodes.node_helpers.conditioning_set_values(negative, conditioning_high)
        low_positive = nodes.node_helpers.conditioning_set_values(positive, conditioning_low)
        low_negative = nodes.node_helpers.conditioning_set_values(negative, conditioning_low)
        
        latent_dict = {"samples": latent.cpu()}
        
        motion_mask_preview = mask_t.mean().item()
        print(f"[WanTTM Conditioning] Mask coverage (mean of latent mask): {motion_mask_preview:.4f}")
        print(f"[WanTTM Conditioning] Added I2V conditioning: HIGH/LOW split concat_latent_image + separate masks")
        print(f"[WanTTM Conditioning] HIGH concat: all frames = start_image latent")
        if low_noise_use_start_image:
            print(f"[WanTTM Conditioning] LOW concat: frame0 = start_image latent, others = gray(0.5)")
        else:
            print(f"[WanTTM Conditioning] LOW concat: all frames = gray(0.5)")
        print(f"[WanTTM Conditioning] HIGH mask: frame0=0.0 (locked), frames1-{latent_frames-1}=1.0 (free)")
        if low_noise_lock_first_frame:
            print(f"[WanTTM Conditioning] LOW mask: frame0=0.0 (locked), frames1-{latent_frames-1}=1.0 (free)")
        else:
            print(f"[WanTTM Conditioning] LOW mask: all frames = 1.0 (free)")
        
        return (
            high_positive, high_negative,
            low_positive, low_negative,
            latent_dict, ttm_params,
        )


class WanTTM_Sampler:
    """Two-phase MoE sampler with TTM Dual Clock support.

    Supports:
    - Standard Wan2.2 I2V (MoE sampling without TTM)
    - TTM-enhanced I2V (with temporal reference blending)
    
    Key design: Pure sigma-based TTM implementation
    - Uses sigma directly for noisy reference calculation
    - No timestep conversion needed
    - Aligned with official TTM dual-clock logic
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
                "skip_low_phase": ("BOOLEAN", {"default": False}),
                "use_high_for_low": ("BOOLEAN", {"default": False}),
                "ttm_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "TTM 注入强度（0=不注入，1=完全对齐参考运动）。"}),
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
        skip_low_phase: bool = False,
        use_high_for_low: bool = False,
        ttm_strength: float = 1.0,
    ):
        """TTM Dual Clock Denoising with MoE support.
        
        TTM Algorithm (sigma-based):
        1. Initialize: latent = (1 - sigma_start) * ref + sigma_start * fixed_noise
        2. For each step in [tweak_index, tstrong_index):
           - Normal Euler step: x = x + noise_pred * dt
           - TTM injection: x = x * bg_mask + noisy_ref * motion_mask
           - where noisy_ref = (1 - sigma_next) * ref + sigma_next * fixed_noise
        3. After tstrong_index: normal denoising (no injection)
        
        This is a pure sigma implementation - no timestep conversion needed.
        """
        device = mm.get_torch_device()
        latent_image = latent["samples"].to(device)
        
        print(f"[WanTTM] === Starting TTM Sampler ===")
        print(f"[WanTTM] Initial latent shape: {latent_image.shape}")
        if skip_low_phase:
            print(f"[WanTTM] skip_low_phase=True -> will skip LOW phase")
        if use_high_for_low:
            print(f"[WanTTM] use_high_for_low=True -> reuse HIGH model in low phase")
        if skip_low_phase:
            print(f"[WanTTM] skip_low_phase=True -> will run HIGH phase only, no LOW refinement")
        print(f"[WanTTM] TTM strength (alpha): {ttm_strength:.3f}")

        # Generate sigma schedule using model's parameters
        sampling_high = model_high.get_model_object("model_sampling")
        sigmas = generate_wan_sigmas(sampling_high, steps).to(device=device, dtype=latent_image.dtype)
        
        print(f"[WanTTM] Sigma schedule ({len(sigmas)} values):")
        print(f"[WanTTM]   First 5: {[f'{s:.4f}' for s in sigmas[:5].tolist()]}")
        print(f"[WanTTM]   Last 5: {[f'{s:.4f}' for s in sigmas[-5:].tolist()]}")
        
        use_flow_sampler = scheduler == FLOW_SCHEDULER_NAME

        # Parse TTM parameters
        if ttm_params is None:
            ttm_params = {}
        
        ttm_reference_latents = ttm_params.get("ttm_reference_latents")
        ttm_mask = ttm_params.get("ttm_mask")
        tweak_index = int(ttm_params.get("ttm_start_step", 0))
        tstrong_index = int(ttm_params.get("ttm_end_step", steps))
        
        # Validate TTM indices
        # CRITICAL: TTM must complete BEFORE MoE switch to low model
        # Low model cannot handle high-noise TTM injection!
        if tstrong_index > steps:
            print(f"[WanTTM] WARNING: ttm_end_step {tstrong_index} > steps {steps}, clamping")
            tstrong_index = steps
        if tstrong_index > switch_step:
            print(f"[WanTTM] WARNING: ttm_end_step {tstrong_index} > switch_step {switch_step}")
            print(f"[WanTTM] TTM must complete before MoE switch! Clamping tstrong to {switch_step}")
            tstrong_index = switch_step
        if tweak_index < 0:
            tweak_index = 0
        if tweak_index >= steps:
            print(f"[WanTTM] WARNING: ttm_start_step {tweak_index} >= steps, disabling TTM")
            tweak_index = steps
        if tweak_index >= tstrong_index:
            print(f"[WanTTM] WARNING: tweak_index >= tstrong_index, disabling TTM")
            tweak_index = tstrong_index
        
        ttm_enabled = (
            ttm_reference_latents is not None and 
            ttm_mask is not None and 
            tstrong_index > tweak_index
        )
        
        preview_callback = latent_preview.prepare_callback(model_high, steps)
        
        # Prepare TTM components
        ref = None
        motion_mask = None
        background_mask = None
        fixed_noise = None  # 保留占位，不再用于前景“自由噪声”
        sampling_high = None
        
        if ttm_enabled:
            print(f"[WanTTM] TTM Dual Clock ENABLED")
            print(f"[WanTTM]   tweak_index (background starts): {tweak_index}")
            print(f"[WanTTM]   tstrong_index (motion stops): {tstrong_index}")
            print(f"[WanTTM]   switch_step (MoE boundary): {switch_step}")
            print(f"[WanTTM]   TTM active steps: [{tweak_index}, {tstrong_index}) - HIGH model only!")
            
            # Prepare reference latents（作为“目标运动引导”而不是“带噪起点”）
            ref = ttm_reference_latents.to(device)
            if ref.dim() == 4:
                ref = ref.unsqueeze(0)
            
            # Process through model's latent format
            ref = comfy.sample.fix_empty_latent_channels(model_high, ref)
            latent_format = model_high.get_model_object("latent_format")
            if latent_format is not None:
                ref = latent_format.process_in(ref)
            
            # Get model_sampling（目前仅用于 debug / 可能的缩放，前景不再注入自由噪声）
            sampling_high = model_high.get_model_object("model_sampling")
            
            print(f"[WanTTM] Reference latent (clean guide): {ref.shape}, range: [{ref.min():.3f}, {ref.max():.3f}]")
            
            # Prepare mask: [T,1,h,w] -> [1,C,T,h,w]
            mask = ttm_mask.to(device)
            if mask.dim() == 4:
                # [T,1,h,w] -> [1,1,T,h,w]
                mask = mask.permute(1, 0, 2, 3).unsqueeze(0)
            
            # Expand to match ref dimensions
            motion_mask = mask.expand(ref.shape[0], ref.shape[1], -1, -1, -1).to(ref.dtype)
            background_mask = 1.0 - motion_mask
            
            # Mask statistics
            mask_coverage = (motion_mask > 0.5).float().mean().item()
            print(f"[WanTTM] Motion mask: {motion_mask.shape}, coverage: {mask_coverage:.1%}")
            
            # Check temporal variation
            mask_per_frame = motion_mask[0, 0].mean(dim=(1, 2))
            if mask_per_frame.std() < 1e-6:
                print(f"[WanTTM] WARNING: Mask appears STATIC across frames!")
            else:
                print(f"[WanTTM] Mask is dynamic, per-frame mean range: [{mask_per_frame.min():.3f}, {mask_per_frame.max():.3f}]")
            
            # 不再为前景生成单独的“自由噪声”初始状态，而是把 ref 当作“目标运动轨迹”。
            # latent_image 仍由 I2V + 文本决定，TTM 只在运动区域做插值引导。
            sigma_init = sigmas[tweak_index].item()
            print(f"[WanTTM] Prepared clean TTM reference at sigma={sigma_init:.4f} (latent_image left unchanged)")
        else:
            print(f"[WanTTM] TTM DISABLED, running standard MoE sampling")
        
        # Prepare sampling noise
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds).to(device)
        noise_mask = latent.get("noise_mask")
        
        # MoE phase configuration
        switch_step_clamped = min(max(switch_step, 1), steps)
        has_phase_2 = (switch_step_clamped < steps) and (not skip_low_phase)
        
        print(f"[WanTTM] MoE Configuration:")
        print(f"[WanTTM]   Phase 1 (HIGH): steps 0-{switch_step_clamped-1}, CFG={cfg_high}")
        if has_phase_2:
            print(f"[WanTTM]   Phase 2 (LOW): steps {switch_step_clamped}-{steps-1}, CFG={cfg_low}")
        
        # State tracking for callback
        global_step_counter = [0]
        current_sigmas = [sigmas]  # Will be updated per phase

        def ttm_callback(data: Dict[str, Any]):
            """TTM Dual Clock callback - pure sigma-based implementation.
            
            Called after each Euler step with:
            - data["i"]: step index within current phase
            - data["x"]: latent after Euler step (modifiable in-place)
            - data["sigma_next"]: next sigma value (use this for noisy ref)
            - data["denoised"]: model prediction (for preview)
            """
            step_in_phase = data["i"]
            x = data["x"]
            sigma_next = data["sigma_next"]
            denoised = data.get("denoised", x)
            
            global_step = global_step_counter[0]
            
            # TTM injection：只在 tweak_index 这一跳注入一次，引导到参考运动轨迹；
            # 后面的所有步（包括低噪阶段）完全按照普通 I2V 继续去噪。
            if ttm_enabled and global_step == tweak_index:
                # 计算当前步的 sigma（仅用于日志打印）
                sigma_val = float(sigma_next) if not torch.is_tensor(sigma_next) else sigma_next.item()
                # 在 ComfyUI 语义下：mask=1 表示“自由生成区域”，不应该在低噪阶段持续往里塞参考。
                # 这里只在高噪早期打一针，把运动区域拉到 ref 附近，后面全部交给正常 FlowMatch I2V。
                # 这里把 ref 视为“干净的运动引导”，不再叠加随机噪声，只在采样过程中插值靠近 ref。
                guide_ref = ref

                guide_ref = guide_ref.to(device=x.device, dtype=x.dtype)
                bg_mask = background_mask.to(device=x.device, dtype=x.dtype)
                mot_mask = motion_mask.to(device=x.device, dtype=x.dtype)

                # 单次注入强度：alpha 表示 motion 区域拉向 ref 的比例；1.0=完全对齐，0.5=折中。
                alpha = max(0.0, min(float(ttm_strength), 1.0))
                alpha_tensor = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)

                # Debug log
                print(f"[WanTTM] TTM single-shot injection at step {global_step}")
                print(f"[WanTTM]   sigma_next={sigma_val:.4f}")
                print(f"[WanTTM]   guide_ref range: [{guide_ref.min():.3f}, {guide_ref.max():.3f}]")
                print(f"[WanTTM]   alpha={alpha:.3f}")

                # Apply dual clock（单次注入版本）：
                #   motion 区域：x = (1-alpha)*x + alpha*guide_ref
                #   background 区域：保持 x 不变
                mot_alpha = mot_mask * alpha_tensor
                bg_effective = bg_mask + mot_mask * (1.0 - alpha_tensor)
                x.mul_(bg_effective).add_(guide_ref * mot_alpha)
            
            # Preview callback
            if preview_callback is not None:
                preview_callback(global_step, denoised, x, steps)
            
            global_step_counter[0] += 1

        def run_flow_phase(model, cfg, pos, neg, phase_sigmas, disable_noise, latent_in,
                           skip_final_scaling: bool = False):
            """Run a single MoE phase with FlowMatch sampling.

            Args:
                skip_final_scaling: If True, skip inverse_noise_scaling at the end.
                                    Use this for Phase 1 when Phase 2 follows.
            """
            current_sigmas[0] = phase_sigmas
            return sample_flowmatch_ttm(
                model=model,
                noise=noise,
                cfg=cfg,
                sigmas=phase_sigmas,
                positive=pos,
                negative=neg,
                latent_image=latent_in,
                disable_noise=disable_noise,
                force_zero_end=True,
                skip_final_scaling=skip_final_scaling,
                noise_mask=noise_mask,
                callback=ttm_callback,
                disable_pbar=False,
                seed=seed,
            )

        # Execute sampling
        if use_flow_sampler:
            # Phase 1: High-noise model
            if ttm_enabled and tweak_index > 0:
                # Skip sigmas before tweak_index (already initialized there)
                sigmas_p1 = sigmas[tweak_index:switch_step_clamped + 1].clone()
                global_step_counter[0] = tweak_index
            else:
                sigmas_p1 = sigmas[:switch_step_clamped + 1].clone()
                global_step_counter[0] = 0
            
            if not has_phase_2:
                # Single phase: use all remaining sigmas
                if ttm_enabled and tweak_index > 0:
                    sigmas_p1 = sigmas[tweak_index:].clone()
                else:
                    sigmas_p1 = sigmas.clone()
            
            print(f"[WanTTM] Phase 1: {len(sigmas_p1)-1} steps, sigmas [{sigmas_p1[0]:.4f} -> {sigmas_p1[-1]:.4f}]")

            # FIX: Phase 1 should only disable_noise if TTM has pre-initialized the latent.
            # 当 has_phase_2=True 但 ttm_enabled=False 时，Phase 1 仍然需要从 noise 初始化！
            disable_noise_p1 = ttm_enabled
            # 有 Phase 2 时，Phase 1 不能做 inverse_noise_scaling，否则会把 latent 放回输出空间导致数值放大。
            skip_scaling_p1 = has_phase_2

            print(f"[WanTTM] Phase 1 config: disable_noise={disable_noise_p1}, skip_final_scaling={skip_scaling_p1}")
            print(f"[WanTTM] Phase 1 input latent range: [{latent_image.min():.3f}, {latent_image.max():.3f}]")

            latent_image = run_flow_phase(
                model_high, cfg_high, high_positive, high_negative,
                sigmas_p1, disable_noise=disable_noise_p1, latent_in=latent_image,
                skip_final_scaling=skip_scaling_p1,
            )

            print(f"[WanTTM] Phase 1 output latent range: [{latent_image.min():.3f}, {latent_image.max():.3f}]")
            
            # Phase 2: Low-noise model（继续沿用高噪阶段的 latent，保持轨迹连续，
            # 但不再做任何 TTM 注入，只受文本 + I2V concat_latent_image 影响）。
            if has_phase_2:
                sigmas_p2 = sigmas[switch_step_clamped:].clone()
                print(f"[WanTTM] Phase 2: {len(sigmas_p2)-1} steps, sigmas [{sigmas_p2[0]:.4f} -> {sigmas_p2[-1]:.4f}]")
                print(f"[WanTTM] Phase 2: TTM disabled, normal denoising with {'HIGH' if use_high_for_low else 'LOW'} model")
                print(f"[WanTTM] Phase 2 input latent range: [{latent_image.min():.3f}, {latent_image.max():.3f}]")

                # Phase 2: disable_noise=True（接着 Phase 1 输出继续），skip_final_scaling=False（最终阶段要做一次 inverse_noise_scaling）
                latent_image = run_flow_phase(
                    model_high if use_high_for_low else model_low,
                    cfg_low, low_positive, low_negative,
                    sigmas_p2, disable_noise=True, latent_in=latent_image,
                    skip_final_scaling=False,
                )
        else:
            # Fallback: use standard ComfyUI sampler
            print(f"[WanTTM] Using standard sampler: {sampler_name}")
            latent_image = comfy.sample.fix_empty_latent_channels(model_high, latent_image)

            # NOTE:
            # Standard ComfyUI samplers expect a callback with signature
            #   callback(step, x0, x, total_steps)
            # while our TTM callback is tailored for FlowMatch Euler and
            # receives a single dict with rich context (sigmas, etc.).
            # Trying to wire TTM into the non‑flow path easily leads to
            # mismatched expectations and incorrect sigma handling.
            #
            # To keep behaviour predictable (and avoid runtime errors),
            # we currently DISABLE TTM when the user selects a non‑flow
            # scheduler, and only use the preview callback here.
            if ttm_enabled:
                print(
                    "[WanTTM] WARNING: TTM is only supported with the "
                    f"'{FLOW_SCHEDULER_NAME}' scheduler. Running WITHOUT TTM "
                    "for this sampler."
                )

            latent_image = comfy.sample.sample(
                model_high, noise, steps, cfg_high, sampler_name, scheduler,
                high_positive, high_negative, latent_image,
                denoise=denoise,
                disable_noise=has_phase_2,
                start_step=0,
                last_step=switch_step_clamped,
                force_full_denoise=has_phase_2,
                noise_mask=noise_mask,
                callback=preview_callback,
                disable_pbar=False,
                seed=seed
            )

            if has_phase_2:
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
                    callback=None,
                    disable_pbar=False,
                    seed=seed
                )

        out = latent.copy()
        out["samples"] = latent_image
        
        print(f"[WanTTM] === Sampling Complete ===")
        print(f"[WanTTM] Output shape: {latent_image.shape}")
        print(f"[WanTTM] Output range: [{latent_image.min():.3f}, {latent_image.max():.3f}]")
        
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
