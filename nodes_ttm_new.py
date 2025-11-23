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
from comfy.samplers import KSAMPLER


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
                "motion_signal_mask": ("MASK", {"tooltip": "Mask video aligned with motion_signal_video (T,H,W)."}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "target_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Output width in pixels (multiple of 8)."}),
                "target_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8, "tooltip": "Output height in pixels (multiple of 8)."}),
                "num_frames": ("INT", {"default": 21, "min": 1, "max": 512, "step": 1}),
                "ttm_start_step": ("INT", {"default": 1, "min": 0, "max": 255, "step": 1}),
                "ttm_end_step": ("INT", {"default": 4, "min": 1, "max": 255, "step": 1}),
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

        # Allocate latent: [B,16,latent_frames,h,w] as Wan expects 16 base latent channels
        batch_size = start_latent.shape[0]
        latent = torch.zeros(batch_size, 16, latent_frames, h, w, device=device)
        # Put start frame into first latent frame
        latent[:, :, :1] = start_latent.to(device)

        # --- Process motion mask --------------------------------------------------------
        # motion_signal_mask: [T,H,W] or [H,W]
        if motion_signal_mask.ndim == 2:
            mask_t = motion_signal_mask.unsqueeze(0)  # [1,H,W]
        elif motion_signal_mask.ndim == 3:
            mask_t = motion_signal_mask
        else:
            raise ValueError(f"motion_signal_mask must be 2D or 3D, got {motion_signal_mask.ndim}D")

        # Resize to latent spatial size and to latent_frames temporally
        mask_T = mask_t.shape[0]
        mask_t = mask_t.unsqueeze(1)  # [T,1,H,W]
        mask_t = torch.nn.functional.interpolate(
            mask_t,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )  # [T,1,h,w]
        if mask_T > latent_frames:
            mask_t = mask_t[:latent_frames]
        elif mask_T < latent_frames:
            pad = latent_frames - mask_T
            mask_t = torch.cat([mask_t, mask_t[:pad]], dim=0)

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

        # For now, we forward the original conditioning lists unchanged
        high_positive = positive
        high_negative = negative
        low_positive = positive
        low_negative = negative

        latent_dict = {"samples": latent.cpu()}
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
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {}),
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
        latent_image = latent["samples"]
        
        print(f"[WanTTM] Initial latent: {latent_image.shape}")

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
            if mask.dim() == 4:
                mask = mask.unsqueeze(0).movedim(1, 2)
            motion_mask = mask.expand(ref.shape[0], ref.shape[1], -1, -1, -1)
            background_mask = 1.0 - motion_mask
            
            # Debug mask info
            mask_nonzero = (motion_mask > 0.5).float().mean()
            print(f"[WanTTM] Motion mask shape: {motion_mask.shape}")
            print(f"[WanTTM] Motion mask: mean={motion_mask.mean():.3f}, range=[{motion_mask.min():.3f}, {motion_mask.max():.3f}]")
            print(f"[WanTTM] Motion coverage (>0.5): {mask_nonzero:.1%}")
            
            # Generate fixed noise for motion signal
            fixed_noise = comfy.sample.prepare_noise(ref, seed + 1, None).to(device)
            
            # Get scheduler and sigmas for initialization
            sigmas = comfy.samplers.calculate_sigmas(
                model_high.get_model_object("model_sampling"),
                scheduler,
                steps
            ).to(device)
            
            # Initialize: motion region with noisy signal, background with random noise
            if tweak_index >= 0 and tweak_index < len(sigmas):
                tweak_sigma = sigmas[tweak_index]
                print(f"[WanTTM] Initializing at tweak_index={tweak_index}, sigma={tweak_sigma:.3f}")
                
                # Prepare random noise for background
                batch_inds = latent.get("batch_index", None)
                random_noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds).to(device)
                noisy_background = latent_image + random_noise * tweak_sigma
                
                # Prepare noisy motion signal
                noisy_ref = ref + fixed_noise * tweak_sigma
                
                # Combine: background random noise + motion noisy signal
                latent_image = noisy_background * background_mask + noisy_ref * motion_mask
                print(f"[WanTTM]   Background: random noise")
                print(f"[WanTTM]   Motion region: noisy motion signal")
            else:
                print(f"[WanTTM] Warning: tweak_index={tweak_index} out of range, using original latent")
            
        else:
            print(f"[WanTTM] TTM disabled, running standard MoE sampling")
            ref = None
            fixed_noise = None
            motion_mask = None
            background_mask = None
        
        # Prepare noise ONCE for both phases (critical!)
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds).to(device)
        noise_mask = latent.get("noise_mask")
        
        # Get sigmas for TTM
        sigmas = comfy.samplers.calculate_sigmas(
            model_high.get_model_object("model_sampling"),
            scheduler,
            steps
        ).to(device)
        
        # Determine MoE switching point
        switch_step_clamped = min(max(switch_step, 0), steps - 1)
        has_phase_2 = switch_step_clamped < steps - 1
        
        print(f"[WanTTM] MoE switching at step {switch_step_clamped}")
        print(f"[WanTTM] Phase 1: HIGH model [0, {switch_step_clamped}], CFG={cfg_high}")
        if has_phase_2:
            print(f"[WanTTM] Phase 2: LOW model [{switch_step_clamped}, {steps}], CFG={cfg_low}")
        
        # TTM Dual Clock callback
        step_offset = [0]  # Track global step across phases
        
        def ttm_callback(step, x0, x, total_steps):
            """TTM Dual Clock: replace motion region with noisy signal after each step."""
            global_step = step_offset[0] + step
            
            # Apply TTM replacement if in dual clock range
            if ttm_enabled and tweak_index <= global_step < tstrong_index:
                # Get next timestep for noisy motion signal
                next_step = global_step + 1
                if next_step < len(sigmas):
                    # Use scheduler.add_noise to properly add noise
                    # In comfy, we approximate with direct noise addition
                    next_sigma = sigmas[next_step]
                    noisy_ref = ref + fixed_noise * next_sigma
                    
                    # Replace motion region in x (the latent after scheduler.step)
                    # This modifies the tensor in-place
                    x.mul_(background_mask).add_(noisy_ref * motion_mask)
                    
                    if global_step == tweak_index:
                        print(f"[WanTTM] TTM Dual Clock active: steps [{tweak_index}, {tstrong_index})")
                        print(f"[WanTTM]   Background: normal denoising")
                        print(f"[WanTTM]   Motion region: replaced with noisy signal")
            
            return x
        
        # Phase 1: HIGH noise model (following official WanMoeKSampler)
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
            callback=ttm_callback if ttm_enabled else None,
            disable_pbar=False,
            seed=seed
        )
        
        # Phase 2: LOW noise model (if needed)
        if has_phase_2:
            step_offset[0] = switch_step_clamped  # Update offset for Phase 2
            
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
                callback=ttm_callback if ttm_enabled else None,
                disable_pbar=False,
                seed=seed
            )
        
        out = latent.copy()
        out["samples"] = latent_image
        print(f"[WanTTM] Sampling complete. Output shape: {latent_image.shape}")
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
