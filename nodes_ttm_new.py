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
    """Two-phase MoE sampler with TTM support.

    This is a clean baseline implementation that:
    - Uses ComfyUI's sigma/timestep calculation (same as official Wan2.2 workflows).
    - Runs high-noise then low-noise phases with a manual loop.
    - Applies TTM blending after each step in the specified range.

    NOTE: This is a first-pass implementation focused on correctness and stability,
    not micro-optimisation.
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
                "ttm_params": ("DICT", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "switch_step": ("INT", {"default": 6, "min": 1, "max": 128, "step": 1}),
                "cfg_high": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 25.0, "step": 0.1}),
                "cfg_low": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 25.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
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
        ttm_params: Dict[str, Any],
        seed: int,
        steps: int,
        switch_step: int,
        cfg_high: float,
        cfg_low: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
    ):
        device = mm.get_torch_device()
        latent_image = latent["samples"].to(device)

        # Prepare noise with correct shape
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds).to(device)

        # Get sampling object and sigmas from the HIGH model (official way)
        sampling_high = model_high.get_model_object("model_sampling")
        if sampling_high is None:
            raise RuntimeError("model_high has no 'model_sampling' object")

        sigmas = comfy.samplers.calculate_sigmas(sampling_high, scheduler, steps).to(device)
        sigmas = sigmas[int(steps * (1 - denoise)):]
        timesteps = sampling_high.timestep(sigmas).to(device)

        # Simple Euler scheduler wrapper
        class SimpleScheduler:
            def __init__(self, sigmas_, timesteps_):
                self.sigmas = sigmas_
                self.timesteps = timesteps_

            def step(self, noise_pred, t, x):
                # Locate index of t in timesteps
                idx = (self.timesteps == t).nonzero(as_tuple=True)[0][0]
                sigma = self.sigmas[idx]
                if idx + 1 < len(self.sigmas):
                    dt = self.sigmas[idx + 1] - sigma
                else:
                    dt = -sigma
                return x + noise_pred * dt

        scheduler_obj = SimpleScheduler(sigmas, timesteps)

        # Manually pad latent channels to 36 for Wan2.2 (16 real + 20 zero)
        print(f"[WanTTM] latent_image before padding: {latent_image.shape}")
        if latent_image.shape[1] == 16:
            B, C, T, H, W = latent_image.shape
            pad_channels = 36 - C
            pad = torch.zeros(B, pad_channels, T, H, W, device=device, dtype=latent_image.dtype)
            latent_image = torch.cat([latent_image, pad], dim=1)
            print(f"[WanTTM] latent_image after padding: {latent_image.shape}")
        
        # Rebuild noise to match padded latent
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds).to(device)
        print(f"[WanTTM] noise shape: {noise.shape}")

        # Extract TTM params
        ttm_reference_latents = ttm_params.get("ttm_reference_latents")  # [C,T,H,W]
        ttm_mask = ttm_params.get("ttm_mask")  # [T,1,H,W]
        ttm_start_step = int(ttm_params.get("ttm_start_step", 0))
        ttm_end_step = int(ttm_params.get("ttm_end_step", steps))

        if ttm_reference_latents is not None:
            print(
                f"[WanTTM] TTM ref: {tuple(ttm_reference_latents.shape)}, "
                f"mask: {tuple(ttm_mask.shape)}, range: [{ttm_start_step}, {ttm_end_step})"
            )

        # Broadcast mask to latent shape: [B,C,T,H,W]
        if ttm_reference_latents is not None and ttm_mask is not None:
            ref = ttm_reference_latents.to(device)  # [C,T,H,W]
            # Add batch dim and match channels
            if ref.dim() == 4:
                ref = ref.unsqueeze(0)  # [1,C,T,H,W]
            if ref.shape[1] != latent_image.shape[1]:
                # Pad or truncate channels to match latent
                if ref.shape[1] < latent_image.shape[1]:
                    pad_channels = latent_image.shape[1] - ref.shape[1]
                    pad = torch.zeros(
                        ref.shape[0], pad_channels, *ref.shape[2:],
                        device=ref.device, dtype=ref.dtype
                    )
                    ref = torch.cat([ref, pad], dim=1)
                else:
                    ref = ref[:, :latent_image.shape[1]]

            # Mask: [T,1,H,W] -> [1,1,T,H,W] -> [1,C,T,H,W]
            mask = ttm_mask.to(device)
            if mask.dim() != 4:
                raise ValueError(f"ttm_mask must be [T,1,H,W], got {tuple(mask.shape)}")
            mask = mask.unsqueeze(0).movedim(1, 2)  # [1,1,T,H,W]
            mask = mask.expand(ref.shape[0], ref.shape[1], -1, -1, -1)  # [1,C,T,H,W]
        else:
            ref = None
            mask = None

        # Extract conditioning tensors - ComfyUI format: [[tensor, dict], ...]
        def extract_cond(cond):
            if isinstance(cond, list) and len(cond) > 0:
                c0 = cond[0]
                if isinstance(c0, (list, tuple)) and len(c0) > 0:
                    tensor = c0[0]
                    if isinstance(tensor, torch.Tensor):
                        return tensor.to(device)
            raise ValueError(f"Invalid conditioning format: {type(cond)}")

        pos_high = extract_cond(high_positive)
        neg_high = extract_cond(high_negative)
        pos_low = extract_cond(low_positive)
        neg_low = extract_cond(low_negative)
        print(f"[WanTTM] Conditioning shapes: pos={pos_high.shape}, neg={neg_high.shape}")

        # Main sampling loop ---------------------------------------------------------
        # Initialize x with noise at starting sigma (Flow Matching: x_t = (1-t)*clean + t*noise)
        t_start = timesteps[0].float()
        if t_start.dim() == 0:
            t_start_norm = t_start / 1000.0 if t_start > 1.0 else t_start
        else:
            t_start_norm = (t_start / 1000.0).view(-1, 1, 1, 1, 1)
        x = (1.0 - t_start_norm) * latent_image + t_start_norm * noise
        
        switch_step_clamped = min(max(switch_step, 1), steps)
        print(f"[WanTTM] steps={steps}, switch_step={switch_step_clamped}, start t={t_start.item():.4f}")

        # Get model dtypes once
        model_high_dtype = next(model_high.get_model_object("diffusion_model").parameters()).dtype
        model_low_dtype = next(model_low.get_model_object("diffusion_model").parameters()).dtype
        print(f"[WanTTM] Model dtypes: HIGH={model_high_dtype}, LOW={model_low_dtype}")

        for idx in range(len(timesteps)):
            step_index = idx
            
            # Select model based on step
            if step_index < switch_step_clamped:
                model = model_high.get_model_object("diffusion_model")
                pos, neg, cfg = pos_high, neg_high, float(cfg_high)
                phase = "HIGH"
                target_dtype = model_high_dtype
            else:
                model = model_low.get_model_object("diffusion_model")
                pos, neg, cfg = pos_low, neg_low, float(cfg_low)
                phase = "LOW"
                target_dtype = model_low_dtype

            t = timesteps[idx]
            # Ensure timestep has batch dimension [B]
            if t.dim() == 0:
                t_batch = t.unsqueeze(0).expand(x.shape[0])
            else:
                t_batch = t

            print(f"[WanTTM] Step {step_index+1}/{len(timesteps)} | phase={phase} | t={t.item():.1f}")
            
            # Ensure all tensors have matching dtype
            x_in = x.to(dtype=target_dtype)
            pos_in = pos.to(dtype=target_dtype)
            neg_in = neg.to(dtype=target_dtype)
            
            with torch.no_grad():
                eps_cond = model(x=x_in, timestep=t_batch, context=pos_in)
                eps_uncond = model(x=x_in, timestep=t_batch, context=neg_in)
                eps = eps_uncond + cfg * (eps_cond - eps_uncond)
            
            # Free intermediate tensors
            del eps_cond, eps_uncond, x_in, pos_in, neg_in

            # Pad eps to match x channels (model outputs 16, but x has 36)
            if eps.shape[1] < x.shape[1]:
                pad_channels = x.shape[1] - eps.shape[1]
                eps_pad = torch.zeros(
                    eps.shape[0], pad_channels, *eps.shape[2:],
                    device=eps.device, dtype=eps.dtype
                )
                eps = torch.cat([eps, eps_pad], dim=1)
            
            # Euler step (ensure eps dtype matches x)
            eps = eps.to(dtype=x.dtype)
            x = scheduler_obj.step(eps, t, x)
            del eps

            # TTM blending in the specified range
            if ref is not None and mask is not None and (ttm_start_step <= step_index < ttm_end_step):
                if idx + 1 < len(timesteps):
                    next_t = timesteps[idx + 1]
                else:
                    next_t = t
                ref_noisy = ref
                # Flow Matching style: x_t = (1 - t)*ref + t*noise
                # Here we re-use add_noise helper from above logic
                # but with the same sigmas/timesteps domain
                t_scalar = next_t
                if t_scalar.dim() == 0:
                    tt = t_scalar.float() / 1000.0 if t_scalar > 1.0 else t_scalar.float()
                else:
                    tt = t_scalar.float() / 1000.0
                    tt = tt.view(-1, 1, 1, 1, 1)
                ref_noisy = (1.0 - tt) * ref + tt * noise
                x = x * (1.0 - mask) + ref_noisy * mask
                print(f"[WanTTM] Applied TTM blending at step {step_index+1}")

        print(f"[WanTTM] Sampling complete. Output shape: {x.shape}")
        out = {"samples": x}
        return (out,)
class WanTTM_Decode:
    """Decode Wan2.2 video latents to frames using the given VAE."""

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

        # Decode frame by frame via VAE (simple and robust)
        B, C, T, H, W = x.shape
        print(f"[WanTTM Decode] Input latent shape: {x.shape}")
        
        # Strip padding channels (36 -> 16 for Wan VAE)
        if C > 16:
            x = x[:, :16, :, :, :]
            print(f"[WanTTM Decode] Stripped to 16 channels: {x.shape}")
        
        C = x.shape[1]
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        decoded = vae.decode(x_flat)["samples"]  # [B*T,3,H,W]
        decoded = decoded.reshape(B, T, 3, H, W).permute(0, 1, 3, 4, 2).contiguous()  # [B,T,H,W,C]

        # For now we return only batch 0
        video_frames = decoded[0]
        print(f"[WanTTM Decode] Output video shape: {video_frames.shape}")
        return (video_frames,)


NODE_CLASS_MAPPINGS = {
    "WanTTM_ModelFromUNet": WanTTM_ModelFromUNet,
    "WanTTM_Conditioning_New": WanTTM_Conditioning,
    "WanTTM_Sampler_New": WanTTM_Sampler,
    "WanTTM_Decode_New": WanTTM_Decode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanTTM_ModelFromUNet": "Wan2.2 TTM: Model From UNet",
    "WanTTM_Conditioning_New": "Wan2.2 TTM: Conditioning",
    "WanTTM_Sampler_New": "Wan2.2 TTM: Sampler",
    "WanTTM_Decode_New": "Wan2.2 TTM: Decode",
}
