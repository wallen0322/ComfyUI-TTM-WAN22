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

# FlowMatch scheduler implementation (aligned with KJ's WanVideoWrapper)
sigma_fn = lambda t: t.neg().exp()
t_fn = lambda sigma: sigma.log().neg()
phi1_fn = lambda t: torch.expm1(t) / t
phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t

class FlowMatchSchedulerResMultistep:
    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002, extra_one_step=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(num_inference_steps)
        self.prev_model_output = None
        self.old_sigma_next = None

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, sigmas=None):
        if self.extra_one_step:
            sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        full_sigmas = torch.linspace(self.sigma_max, self.sigma_min, self.num_train_timesteps)
        ss = len(full_sigmas) / num_inference_steps
        if sigmas is None:
            sigmas = []
            for x in range(num_inference_steps):
                idx = int(round(x * ss))
                sigmas.append(float(full_sigmas[idx]))
            sigmas.append(0.0)
        self.sigmas = torch.FloatTensor(sigmas)
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        # Store all timesteps for TTM (same as KJ's all_timesteps)
        self.all_timesteps = self.timesteps.clone()

    def add_noise(self, original_samples, noise, timestep):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

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
        
        # Official TTM: Initial latent comes from motion_signal (ref_latents), not start_image
        # The start_image is locked via I2V concat_latent_image + concat_mask mechanism
        # So we initialize latent with motion_latent shape, but actual initialization happens in sampler
        # For now, we create a placeholder - the sampler will replace it with noisy ref_latents
        latent = start_latent.to(device).expand(-1, -1, latent_frames, -1, -1).clone()
        
        print(f"[WanTTM Conditioning] Latent shape: {latent.shape} (placeholder - will be replaced by noisy motion_signal in sampler)")
        print(f"[WanTTM Conditioning] Note: Start image is locked via I2V concat_latent_image + concat_mask mechanism")

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
        ref_latents = motion_latent[:1].detach()
        ref_latents = cls._normalize_reference_latents(ref_latents, vae)
        ref_stats = getattr(ref_latents, "_wan_stats", None)
        ttm_reference_latents = ref_latents.squeeze(0).contiguous()
        ttm_params = {
            "ttm_reference_latents": ttm_reference_latents,  # [C,T,h,w]
            "ttm_mask": ttm_mask.detach(),                  # [T,1,h,w]
            "ttm_start_step": int(ttm_start_step),
            "ttm_end_step": int(ttm_end_step),
            "ttm_ref_stats": ref_stats,
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

        # Construct Wan FlowMatch scheduler (aligned with KJ's implementation)
        sampling_high = model_high.get_model_object("model_sampling")
        shift = getattr(sampling_high, "flow_shift", 3.0)
        sigma_max = getattr(sampling_high, "sigma_max", 1.0)
        sigma_min = getattr(sampling_high, "sigma_min", 0.003 / 1.002)
        train_steps = getattr(sampling_high, "flow_train_steps", 1000)
        
        # Initialize FlowMatch scheduler (same as KJ's WanVideoWrapper)
        flow_scheduler = FlowMatchSchedulerResMultistep(
            num_inference_steps=steps,
            num_train_timesteps=int(train_steps),
            shift=float(shift),
            sigma_max=float(sigma_max),
            sigma_min=float(sigma_min),
            extra_one_step=False
        )
        flow_scheduler.sigmas = flow_scheduler.sigmas.to(device=device, dtype=latent_image.dtype)
        flow_scheduler.timesteps = flow_scheduler.timesteps.to(device=device, dtype=latent_image.dtype)
        sigmas = flow_scheduler.sigmas
        print(f"[WanTTM] Flow sigmas ({len(sigmas)}): {sigmas.tolist()}")
        print(f"[WanTTM] Flow timesteps ({len(flow_scheduler.timesteps)}): {flow_scheduler.timesteps.tolist()}")
        use_flow_sampler = scheduler == FLOW_SCHEDULER_NAME

        # Prepare TTM data (optional)
        if ttm_params is None:
            ttm_params = {}
        
        ttm_reference_latents = ttm_params.get("ttm_reference_latents")
        ttm_mask = ttm_params.get("ttm_mask")
        tweak_index = int(ttm_params.get("ttm_start_step", 0))  # When background starts
        tstrong_index = int(ttm_params.get("ttm_end_step", steps))  # When motion stops
        if tstrong_index > steps:
            print(f"[WanTTM] WARNING: ttm_end_step {tstrong_index} exceeds total steps {steps}. Clamping to {steps}.")
            tstrong_index = steps
        
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
            motion_mask = mask.expand(ref.shape[0], ref.shape[1], -1, -1, -1).to(ref)
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
            
            # Note: TTM initialization will happen after noise is prepared (see below)
            
        else:
            print(f"[WanTTM] TTM disabled, running standard MoE sampling")
            ref = None
            motion_mask = None
            background_mask = None
        
        # Prepare noise ONCE for both phases (critical!)
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds).to(device)
        noise_mask = latent.get("noise_mask")
        
        # TTM initialization: 
        # - Initial latent comes from start_image (via I2V concat_latent_image mechanism)
        # - Motion signal is used as reference for dynamic guidance during sampling
        # - We prepare fixed_noise for TTM injection, but don't replace initial latent
        # Official implementation replaces latent with noisy ref_latents, but that causes issues
        # because motion_signal may contain full frames (jelly effect), not just motion info
        if ttm_enabled and tweak_index >= 0 and tweak_index < len(flow_scheduler.timesteps):
            # Generate fixed_noise for TTM (same shape as ref_latents)
            # This will be used during sampling to inject noisy motion_signal in motion regions
            import torch
            fixed_noise = torch.randn_like(ref)
            # Store fixed_noise for TTM injection during sampling
            ttm_fixed_noise = fixed_noise
            print(f"[WanTTM] TTM enabled: motion_signal will be injected during sampling via mask")
            print(f"[WanTTM] Initial latent from start_image (locked via I2V concat_mask)")
            print(f"[WanTTM] Motion guidance active from step {tweak_index} to {tstrong_index}")
        else:
            ttm_fixed_noise = None
        
        # Determine MoE switching point
        switch_step_clamped = min(max(switch_step, 0), steps - 1)
        has_phase_2 = switch_step_clamped < steps - 1
        
        print(f"[WanTTM] MoE switching at step {switch_step_clamped}")
        print(f"[WanTTM] Phase 1: HIGH model [0, {switch_step_clamped}], CFG={cfg_high}")
        if has_phase_2:
            print(f"[WanTTM] Phase 2: LOW model [{switch_step_clamped}, {steps}], CFG={cfg_low}")
        
        # TTM Dual Clock + preview callback
        # Official implementation: loop starts from tweak_index, so we need to adjust step indexing
        step_offset = [0]  # Track global step across phases
        phase_start_step = [0]  # Track where current phase starts in the full timestep array

        def sampler_callback(*args, **kwargs):
            """Unified callback that handles both FlowMatch (dict) and KSampler (tuple) formats.
            
            Official TTM logic:
            - Loop: for i, t in enumerate(timesteps[tweak_index:])
            - i=0 corresponds to timesteps[tweak_index] (first step after initialization)
            - TTM injection uses: timesteps[i+tweak_index+1] (next timestep)
            - Condition: (i+tweak_index) < tstrong_index
            """
            # FlowMatch format: callback({"i": i, "x": x, ...})
            if len(args) == 1 and isinstance(args[0], dict):
                data = args[0]
                step = data["i"]  # Loop index within current phase (0-based)
                x = data["x"]  # This is the latent AFTER step (in-place modifiable)
                x0_approx = data.get("denoised", x)  # noise_pred for preview
            # KSampler format: callback(step, x0, x, total_steps)
            elif len(args) == 4:
                step, x0_approx, x, _ = args
            else:
                return  # Unknown format, skip
            
            # Calculate absolute step index in the full timestep array
            # phase_start_step[0] is where current phase starts (0 for phase 1, switch_step for phase 2)
            # step is the loop index within current phase (0-based)
            # For TTM: we need to know if we're in the TTM range [tweak_index, tstrong_index)
            absolute_step = phase_start_step[0] + step
            global_step = step_offset[0] + step

            # TTM injection happens AFTER scheduler.step, using NEXT step's timestep
            # TTM mechanism: 
            # - Background regions: keep denoised result (normal denoising)
            # - Motion regions: replace with noisy motion_signal (dynamic guidance)
            # - First frame: locked via I2V concat_mask mechanism
            if ttm_enabled and tweak_index <= absolute_step < tstrong_index:
                # Calculate next timestep index (same as official: i+tweak_index+1)
                next_timestep_idx = absolute_step + 1
                if next_timestep_idx < len(flow_scheduler.all_timesteps):
                    # Use FlowMatch scheduler's add_noise with the correct timestep (same as official)
                    timestep_value = flow_scheduler.all_timesteps[next_timestep_idx].item()
                    timestep_tensor = torch.tensor([timestep_value], device=ref.device, dtype=torch.long).view(1)
                    # Flatten temporal dim for add_noise: [B,C,T,H,W] -> [B*T,C,H,W]
                    B, C, T, H, W = ref.shape
                    ref_flat = ref.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
                    # Use fixed_noise (same as official, not the sampling noise)
                    fixed_noise_flat = ttm_fixed_noise.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
                    timestep_expanded = timestep_tensor.expand(B * T).long()
                    noisy_flat = flow_scheduler.add_noise(ref_flat, fixed_noise_flat, timestep_expanded)
                    noisy_ref = noisy_flat.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous().to(x.device).to(x.dtype)
                else:
                    # Fallback to clean reference (same as official's else branch)
                    noisy_ref = ref.to(x.device).to(x.dtype)
                
                # Apply TTM dual clock: background keeps denoised, motion gets noisy reference
                # Same as official: latents = latents * background_mask + noisy_latents * motion_mask
                # Ensure masks are on correct device and dtype
                bg_mask = background_mask.to(device=x.device, dtype=x.dtype)
                mot_mask = motion_mask.to(device=x.device, dtype=x.dtype)
                
                # Debug: check mask coverage
                if absolute_step == tweak_index:
                    motion_coverage = (mot_mask > 0.5).float().mean().item()
                    print(f"[WanTTM] TTM Dual Clock active: steps [{tweak_index}, {tstrong_index})")
                    print(f"[WanTTM]   Motion mask coverage: {motion_coverage:.1%}")
                    print(f"[WanTTM]   Background: normal denoising (keeps denoised result)")
                    print(f"[WanTTM]   Motion region: replaced with noisy motion_signal (timestep={flow_scheduler.all_timesteps[next_timestep_idx].item():.1f})")
                    print(f"[WanTTM]   (Frame 1 locked by I2V concat_mask)")
                
                # Apply mask: x = x * bg_mask + noisy_ref * mot_mask
                x.mul_(bg_mask).add_(noisy_ref * mot_mask)

            if preview_callback is not None:
                preview_callback(global_step, x0_approx, x, steps)
        
        def run_flow_phase(model, cfg_scale, pos_cond, neg_cond, sigmas_tensor, disable_noise_flag, force_zero_end_flag, latent_in, phase_steps, enable_callback):
            phase_cb = None
            if sampler_callback is not None and enable_callback:
                phase_cb = sampler_callback  # Direct pass, no wrapper needed
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
        # Official TTM: if tweak_index > 0, loop starts from tweak_index, not 0
        # Official: for i, t in enumerate(timesteps[tweak_index:])
        #   - i=0 corresponds to timesteps[tweak_index] (first step after initialization)
        #   - The loop processes timesteps[tweak_index] to the end
        # In our case: sample_flowmatch processes sigmas[0] to sigmas[-1]
        #   - If tweak_index > 0, we need to start from sigmas[tweak_index]
        #   - But initialization already uses timesteps[tweak_index], so the first loop step
        #     should denoise from timesteps[tweak_index] to the next timestep
        if use_flow_sampler:
            # For TTM: if tweak_index > 0, we skip the first tweak_index sigmas
            # This matches official: loop starts from timesteps[tweak_index]
            if ttm_enabled and tweak_index > 0:
                # Start from tweak_index: only process sigmas from tweak_index onwards
                # Note: sigmas[tweak_index] corresponds to the timestep used in initialization
                sigmas_high = sigmas[tweak_index:switch_step_clamped + 1].clone()
                phase_start_step[0] = tweak_index
            else:
                sigmas_high = sigmas[:switch_step_clamped + 1].clone()
                phase_start_step[0] = 0
            if not has_phase_2:
                if ttm_enabled and tweak_index > 0:
                    sigmas_high = sigmas[tweak_index:].clone()
                else:
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
                enable_callback=True,
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
                phase_start_step[0] = switch_step_clamped
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
                    enable_callback=ttm_enabled,
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
                    callback=sampler_callback if ttm_enabled else None,
                    disable_pbar=False,
                    seed=seed
                )
        
        if ttm_enabled:
            ref_stats = ttm_params.get("ttm_ref_stats")
            if ref_stats is not None:
                mean_tensor, std_tensor = ref_stats
                if mean_tensor is not None and std_tensor is not None:
                    mean_tensor = mean_tensor.to(device=latent_image.device, dtype=latent_image.dtype)
                    std_tensor = std_tensor.to(device=latent_image.device, dtype=latent_image.dtype)
                    std_tensor = torch.where(std_tensor == 0, torch.ones_like(std_tensor), std_tensor)
                    local_motion_mask = motion_mask.to(device=latent_image.device, dtype=latent_image.dtype)
                    denorm = latent_image * std_tensor + mean_tensor
                    latent_image = latent_image * (1.0 - local_motion_mask) + denorm * local_motion_mask

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
