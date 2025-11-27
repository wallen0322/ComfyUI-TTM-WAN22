"""
Clean TTM implementation extracted from KJ's WanVideoWrapper.

Core principle from KJ:
1. Initialize latent with noisy reference at ttm_start_step
2. Loop from ttm_start_step, apply euler step
3. After each step, if step < ttm_end_step:
   - Compute noisy_ref at next_timestep
   - Blend: latent = latent * (1-mask) + noisy_ref * mask

Key formula (Flow Match):
  add_noise(clean, noise, timestep) = (1 - t/1000) * clean + (t/1000) * noise
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Callable
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

import comfy.model_management as mm
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_patcher
import latent_preview
import nodes
from comfy.k_diffusion import sampling as k_diffusion_sampling

def add_noise_timestep(
    original: torch.Tensor,
    noise: torch.Tensor, 
    timestep: torch.Tensor
) -> torch.Tensor:
    """Add noise using Flow Match formula with timestep in [0, 1000].
    
    This is the exact formula from KJ/diffusers:
    noisy = (1 - t/1000) * original + (t/1000) * noise
    """
    t = timestep.float() / 1000.0
    # Reshape t for broadcasting
    while t.dim() < original.dim():
        t = t.unsqueeze(-1)
    return (1.0 - t) * original + t * noise


def get_sigmas(num_steps: int, device: torch.device) -> torch.Tensor:
    """Generate sigmas for Flow Match sampling.
    
    Returns sigmas from 1.0 down to 0.0.
    For num_steps=8: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]
    """
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return sigmas


class WanTTM_PrepareLatents:
    """Prepare TTM reference latents and mask.
    
    Supports two modes:
    1. Direct LATENT input (recommended): Use pre-encoded latents from Wan VAE Encode
    2. IMAGE input: Will try to encode via VAE (may have frame limits)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_mask": ("IMAGE", {"tooltip": "Motion mask video [T,H,W,C]"}),
            },
            "optional": {
                "reference_latents": ("LATENT", {"tooltip": "Pre-encoded reference latents (recommended)"}),
                "reference_video": ("IMAGE", {"tooltip": "Reference video (will be encoded if no latents provided)"}),
                "vae": ("VAE", {"tooltip": "VAE for encoding (required if using reference_video)"}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "MASK",)
    RETURN_NAMES = ("reference_latents", "motion_mask",)
    FUNCTION = "execute"
    CATEGORY = "TTM"
    
    @classmethod
    def execute(cls, motion_mask, reference_latents=None, reference_video=None, vae=None):
        device = mm.get_torch_device()
        
        # Get latents from input
        if reference_latents is not None:
            # Direct latent input (recommended)
            latents = reference_latents["samples"].clone()
            print(f"[TTM Prepare] Using pre-encoded latents: {latents.shape}")
        elif reference_video is not None and vae is not None:
            # Encode video via VAE
            video = reference_video.clone()
            print(f"[TTM Prepare] Encoding video: {video.shape}")
            
            T = video.shape[0]
            video_for_vae = video  # [T,H,W,C]
            
            with torch.no_grad():
                latents = vae.encode(video_for_vae.to(device))
            
            print(f"[TTM Prepare] VAE output: {latents.shape}")
            
            # Handle different VAE output formats
            if latents.dim() == 4:
                latents = latents.permute(1, 0, 2, 3).unsqueeze(0)
            elif latents.dim() == 5:
                if latents.shape[0] == T:
                    latents = latents.squeeze(2).permute(1, 0, 2, 3).unsqueeze(0)
        else:
            raise ValueError("Must provide either reference_latents or (reference_video + vae)")
        
        # Ensure 5D format [B,C,T,H,W]
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        
        print(f"[TTM Prepare] Reference latents shape: {latents.shape}")
        
        # Get latent spatial size
        _, C, T_lat, lh, lw = latents.shape
        
        # Process mask
        mask = motion_mask.clone()
        print(f"[TTM Prepare] Input mask shape: {mask.shape}")
        
        # Ensure [T,H,W] format
        if mask.dim() == 4:
            mask = mask[..., 0]  # [T,H,W,C] -> [T,H,W]
        
        T_mask = mask.shape[0]
        
        # Match mask frames to latent frames (Wan VAE has 4x temporal compression)
        if T_mask != T_lat:
            print(f"[TTM Prepare] Downsampling mask from {T_mask} to {T_lat} frames (4x temporal compression)")
            # Temporal downsample: take every 4th frame, or use interpolation
            # Simple approach: stride sampling to match latent frame count
            indices = torch.linspace(0, T_mask - 1, T_lat).long()
            mask = mask[indices]
        
        # Resize mask to latent spatial size
        mask = mask.unsqueeze(1).float()  # [T,1,H,W]
        mask = F.interpolate(mask, size=(lh, lw), mode='bilinear', align_corners=False)
        mask = mask.squeeze(1)  # [T,h,w]
        
        # Binarize
        mask = (mask > 0.5).float()
        
        print(f"[TTM Prepare] Output mask shape: {mask.shape}")
        print(f"[TTM Prepare] Motion mask coverage: {mask.mean():.3f}")
        
        return ({"samples": latents.cpu()}, mask.cpu())


class TTMSampler:
    """Custom sampler with TTM support.
    
    This sampler wraps the euler sampler and applies TTM injection after each step.
    """
    
    def __init__(self, ref_latent, motion_mask, noise, ttm_start_step, ttm_end_step):
        self.ref = ref_latent
        self.motion_mask = motion_mask
        self.background_mask = 1.0 - motion_mask
        self.noise = noise
        self.ttm_start_step = ttm_start_step
        self.ttm_end_step = ttm_end_step
        self.current_step = 0
    
    def sample(self, model_wrap, sigmas, extra_args, callback, disable_pbar):
        """Custom sampling with TTM injection."""
        # Get initial latent from extra_args or generate
        x = extra_args.get('latent_image', None)
        if x is None:
            x = extra_args.get('noise', None)
        
        # Initialize with noisy reference if starting from step 0
        if self.ttm_start_step == 0:
            sigma_init = sigmas[0].item()
            x = add_noise_at_sigma(self.ref, self.noise, sigma_init)
            print(f"[TTM] Initialized with noisy ref at sigma={sigma_init:.4f}")
        
        total_steps = len(sigmas) - 1
        self.current_step = 0
        
        for i in k_diffusion_sampling.trange(total_steps, disable=disable_pbar):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Standard euler step
            denoised = model_wrap(x, sigma.unsqueeze(0) * x.new_ones([x.shape[0]]))
            d = (x - denoised) / sigma
            dt = sigma_next - sigma
            x = x + d * dt
            
            # TTM injection: after euler step
            if self.ttm_start_step <= self.current_step < self.ttm_end_step:
                # Compute noisy reference at next sigma
                noisy_ref = add_noise_at_sigma(self.ref, self.noise, sigma_next.item())
                # Blend
                x = x * self.background_mask + noisy_ref * self.motion_mask
                print(f"[TTM] Step {self.current_step}: injected at sigma_next={sigma_next.item():.4f}")
            
            self.current_step += 1
            
            if callback is not None:
                callback({'x': x, 'denoised': denoised, 'i': i, 'sigma': sigma, 'sigma_next': sigma_next})
        
        return x


def add_noise_at_sigma(clean: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add noise using Flow Match formula.
    
    noisy = (1 - sigma) * clean + sigma * noise
    """
    return (1.0 - sigma) * clean + sigma * noise


class WanTTM_Sampler_Clean:
    """Clean TTM sampler using ComfyUI's infrastructure.
    
    This sampler uses a custom TTMSampler that:
    1. Has full control over the sampling loop
    2. Applies TTM injection AFTER each euler step
    3. Works with ComfyUI's standard model loading
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "ttm_start_step": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Step to start TTM (background starts denoising)"}),
                "ttm_end_step": ("INT", {"default": 7, "min": 1, "max": 100, "tooltip": "Step to end TTM (motion starts normal denoising)"}),
            },
            "optional": {
                "reference_latents": ("LATENT", {"tooltip": "TTM reference latents (optional, enables TTM when provided)"}),
                "motion_mask": ("MASK", {"tooltip": "Motion mask (optional, required if reference_latents provided)"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "TTM"
    
    @classmethod
    def execute(
        cls,
        model,
        positive,
        negative,
        latent,
        seed: int,
        steps: int,
        cfg: float,
        ttm_start_step: int,
        ttm_end_step: int,
        reference_latents=None,
        motion_mask=None,
    ):
        device = mm.get_torch_device()
        
        # Check if TTM is enabled
        ttm_enabled = reference_latents is not None and motion_mask is not None
        
        print(f"\n[TTM Clean] === Starting Sampling ===")
        print(f"[TTM Clean] Steps: {steps}, CFG: {cfg}")
        print(f"[TTM Clean] TTM: {'ENABLED' if ttm_enabled else 'DISABLED'}")
        if ttm_enabled:
            print(f"[TTM Clean] TTM range: [{ttm_start_step}, {ttm_end_step})")
        print(f"[TTM Clean] Seed: {seed}")
        
        # Get latent samples
        x = latent["samples"].clone()
        
        if ttm_enabled:
            ref = reference_latents["samples"].clone()
            mask = motion_mask.clone()
        else:
            ref = None
            mask = None
        
        print(f"[TTM Clean] Input latent shape: {x.shape}")
        
        if ttm_enabled:
            print(f"[TTM Clean] Reference latent shape: {ref.shape}")
            print(f"[TTM Clean] Motion mask shape: {mask.shape}")
            
            # Prepare mask for broadcasting
            # Input mask: [T,h,w] or [B,T,h,w]
            if mask.dim() == 3:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,h,w]
            elif mask.dim() == 4:
                mask = mask.unsqueeze(1)  # [B,1,T,h,w]
            
            # Expand to match latent shape [B,C,T,h,w]
            if mask.shape[1] == 1 and ref.shape[1] > 1:
                mask = mask.expand(-1, ref.shape[1], -1, -1, -1)
            
            print(f"[TTM Clean] Expanded mask shape: {mask.shape}")
            print(f"[TTM Clean] Motion mask coverage: {mask.mean():.4f}")
        
        # Generate sigmas
        sigmas = get_sigmas(steps, device)
        print(f"[TTM Clean] Sigmas: first={sigmas[0]:.4f}, last={sigmas[-1]:.4f}")
        
        # Generate fixed noise
        generator = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn(x.shape, generator=generator, device='cpu')
        
        # Use ComfyUI's sample infrastructure
        mm.load_model_gpu(model)
        
        # Create callback for preview
        preview_callback = latent_preview.prepare_callback(model, steps)
        
        if not ttm_enabled:
            # Standard sampling without TTM - use ComfyUI's built-in sampler
            print(f"[TTM Clean] Using standard euler sampling")
            result = comfy.sample.sample(
                model,
                noise,
                steps,
                cfg,
                "euler",
                "normal",
                positive,
                negative,
                x,
                denoise=1.0,
                disable_noise=False,
                start_step=0,
                last_step=steps,
                force_full_denoise=True,
                noise_mask=None,
                callback=preview_callback,
                seed=seed,
            )
        else:
            # TTM sampling with manual loop
            # Validate TTM indices
            ttm_start_step = max(0, min(ttm_start_step, steps - 1))
            ttm_end_step = max(ttm_start_step + 1, min(ttm_end_step, steps))
            print(f"[TTM Clean] Validated TTM range: [{ttm_start_step}, {ttm_end_step})")
            
            # Move to device
            ref = ref.to(device)
            mask = mask.to(device)
            noise = noise.to(device)
            
            # Manual sampling with TTM
            result = cls._sample_with_ttm(
                model, positive, negative, x, ref, mask, noise,
                sigmas, cfg, ttm_start_step, ttm_end_step, preview_callback
            )
        
        print(f"[TTM Clean] === Sampling Complete ===")
        print(f"[TTM Clean] Output latent range: [{result.min():.3f}, {result.max():.3f}]")
        
        return ({"samples": result.cpu()},)
    
    @classmethod
    def _sample_with_ttm(
        cls,
        model,
        positive,
        negative,
        latent,
        ref,
        motion_mask,
        noise,
        sigmas,
        cfg,
        ttm_start_step,
        ttm_end_step,
        callback,
    ):
        """Use ComfyUI's sampler with TTM callback injection."""
        device = mm.get_torch_device()
        
        print(f"[TTM] Latent shape: {latent.shape}, Ref shape: {ref.shape}")
        
        # Ensure ref matches latent spatial dimensions
        # latent: [B,C,T,H,W], ref: [B,C,T,H,W]
        if ref.shape[-2:] != latent.shape[-2:]:
            print(f"[TTM] Resizing ref from {ref.shape[-2:]} to {latent.shape[-2:]}")
            B, C, T, H_ref, W_ref = ref.shape
            H_lat, W_lat = latent.shape[-2:]
            # Reshape for interpolation: [B*C*T, 1, H, W]
            ref_flat = ref.view(-1, 1, H_ref, W_ref)
            ref_flat = F.interpolate(ref_flat, size=(H_lat, W_lat), mode='bilinear', align_corners=False)
            ref = ref_flat.view(B, C, T, H_lat, W_lat)
            print(f"[TTM] Ref resized to: {ref.shape}")
        
        # Also resize mask if needed
        if motion_mask.dim() == 3:
            # [T, H, W]
            if motion_mask.shape[-2:] != latent.shape[-2:]:
                print(f"[TTM] Resizing mask from {motion_mask.shape[-2:]} to {latent.shape[-2:]}")
                T_m = motion_mask.shape[0]
                H_lat, W_lat = latent.shape[-2:]
                mask_4d = motion_mask.unsqueeze(1)  # [T, 1, H, W]
                mask_4d = F.interpolate(mask_4d, size=(H_lat, W_lat), mode='bilinear', align_corners=False)
                motion_mask = mask_4d.squeeze(1)  # [T, H, W]
        
        # Regenerate noise to match latent shape (in case ref had different shape)
        if noise.shape != latent.shape:
            print(f"[TTM] Regenerating noise to match latent shape: {latent.shape}")
            noise = torch.randn_like(latent)
        
        # Move TTM components to device
        ref = ref.to(device)
        motion_mask = motion_mask.to(device)
        noise = noise.to(device)
        
        # Ensure mask has correct shape for broadcasting [1,C,T,H,W]
        if motion_mask.dim() == 3:
            motion_mask = motion_mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,H,W]
        if motion_mask.shape[1] == 1 and ref.shape[1] > 1:
            motion_mask = motion_mask.expand(-1, ref.shape[1], -1, -1, -1)
        
        background_mask = 1.0 - motion_mask
        
        print(f"[TTM] Mask shape after expand: {motion_mask.shape}")
        
        # TTM state
        step_counter = [0]
        
        # Create TTM callback wrapper
        def ttm_callback(step, x, x_denoised, total_steps):
            current_step = step_counter[0]
            
            # Apply TTM injection after each step
            if ttm_start_step <= current_step < ttm_end_step:
                sigma_next = sigmas[current_step + 1].item() if current_step + 1 < len(sigmas) else 0.0
                
                # Compute noisy reference at next sigma
                noisy_ref = add_noise_at_sigma(ref, noise, sigma_next)
                
                # Blend in-place
                x.mul_(background_mask).add_(noisy_ref * motion_mask)
                
                print(f"[TTM] Step {current_step}: injected at sigma_next={sigma_next:.4f}")
            
            step_counter[0] += 1
            
            # Preview callback
            if callback is not None:
                try:
                    if x.dim() == 5:
                        preview_x = x[:, :, x.shape[2] // 2, :, :]
                    else:
                        preview_x = x
                    callback(step, preview_x, x_denoised, total_steps)
                except Exception:
                    pass
        
        # Initialize latent with noisy reference if TTM starts at step 0
        latent_in = latent.to(device)
        if ttm_start_step == 0:
            sigma_init = sigmas[0].item()
            # Use expanded ref for initialization
            latent_in = add_noise_at_sigma(ref, noise, sigma_init)
            print(f"[TTM] Initialized with noisy ref at sigma={sigma_init:.4f}")
        
        # Use ComfyUI's standard sample function
        samples = comfy.sample.sample(
            model,
            noise.to('cpu'),  # ComfyUI expects CPU noise
            len(sigmas) - 1,  # steps
            cfg,
            "euler",
            "normal",
            positive,
            negative,
            latent_in.to('cpu'),  # latent tensor, not dict
            denoise=1.0,
            disable_noise=True,  # We already added noise
            start_step=0,
            last_step=len(sigmas) - 1,
            force_full_denoise=True,
            noise_mask=None,
            callback=ttm_callback,
            seed=0,
        )
        
        # samples is a tensor, return it
        return samples


# Node registration
NODE_CLASS_MAPPINGS = {
    "WanTTM_PrepareLatents": WanTTM_PrepareLatents,
    "WanTTM_Sampler_Clean": WanTTM_Sampler_Clean,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanTTM_PrepareLatents": "TTM: Prepare Latents",
    "WanTTM_Sampler_Clean": "TTM: Sampler (Clean)",
}
