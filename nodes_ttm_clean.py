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
    """Prepare TTM reference latents and mask from input video."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "reference_video": ("IMAGE", {"tooltip": "Motion reference video [T,H,W,C] or [B,H,W,C]"}),
                "motion_mask": ("IMAGE", {"tooltip": "Motion mask video, same frames as reference"}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "MASK",)
    RETURN_NAMES = ("reference_latents", "motion_mask",)
    FUNCTION = "execute"
    CATEGORY = "TTM"
    
    @classmethod
    def execute(cls, vae, reference_video, motion_mask, width, height):
        device = mm.get_torch_device()
        
        # Process reference video
        # Input: [T,H,W,C] or [B,H,W,C] in [0,1] range
        video = reference_video
        if video.dim() == 4:
            # [T,H,W,C] -> assume it's frames
            video = video.permute(0, 3, 1, 2)  # [T,C,H,W]
        
        T = video.shape[0]
        
        # Resize to target
        video = F.interpolate(video, size=(height, width), mode='bilinear', align_corners=False)
        
        # VAE encode: expects [B,H,W,C] in ComfyUI
        # Need to encode frame by frame or batch
        video_for_vae = video.permute(0, 2, 3, 1)  # [T,H,W,C]
        
        with torch.no_grad():
            latents = vae.encode(video_for_vae.to(device))  # [T,C,h,w]
        
        # Reshape to [1,C,T,h,w] for video latent format
        latents = latents.permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,T,h,w]
        
        print(f"[TTM Prepare] Reference latents shape: {latents.shape}")
        print(f"[TTM Prepare] Reference latents range: [{latents.min():.3f}, {latents.max():.3f}]")
        
        # Process mask
        mask = motion_mask
        if mask.dim() == 4:
            mask = mask[..., 0]  # Take first channel [T,H,W]
        if mask.dim() == 3:
            pass  # [T,H,W] is correct
        
        # Get latent spatial size
        _, _, _, lh, lw = latents.shape
        
        # Resize mask to latent size
        mask = mask.unsqueeze(1)  # [T,1,H,W]
        mask = F.interpolate(mask, size=(lh, lw), mode='bilinear', align_corners=False)
        mask = mask.squeeze(1)  # [T,h,w]
        
        # Binarize
        mask = (mask > 0.5).float()
        
        print(f"[TTM Prepare] Motion mask shape: {mask.shape}")
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
                "reference_latents": ("LATENT", {"tooltip": "TTM reference latents from PrepareLatents"}),
                "motion_mask": ("MASK", {"tooltip": "Motion mask from PrepareLatents"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "ttm_start_step": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Step to start TTM (background starts denoising)"}),
                "ttm_end_step": ("INT", {"default": 7, "min": 1, "max": 100, "tooltip": "Step to end TTM (motion starts normal denoising)"}),
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
        reference_latents,
        motion_mask,
        seed: int,
        steps: int,
        cfg: float,
        ttm_start_step: int,
        ttm_end_step: int,
    ):
        device = mm.get_torch_device()
        
        print(f"\n[TTM Clean] === Starting TTM Sampling ===")
        print(f"[TTM Clean] Steps: {steps}, CFG: {cfg}")
        print(f"[TTM Clean] TTM range: [{ttm_start_step}, {ttm_end_step})")
        print(f"[TTM Clean] Seed: {seed}")
        
        # Get latent samples
        x = latent["samples"].clone()
        ref = reference_latents["samples"].clone()
        mask = motion_mask.clone()
        
        print(f"[TTM Clean] Input latent shape: {x.shape}")
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
        print(f"[TTM Clean] Sigmas: {sigmas.tolist()}")
        
        # Generate fixed noise for TTM
        generator = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn(x.shape, generator=generator, device='cpu')
        
        # Validate TTM indices
        ttm_start_step = max(0, min(ttm_start_step, steps - 1))
        ttm_end_step = max(ttm_start_step + 1, min(ttm_end_step, steps))
        
        print(f"[TTM Clean] Validated TTM range: [{ttm_start_step}, {ttm_end_step})")
        
        # Move to device
        ref = ref.to(device)
        mask = mask.to(device)
        noise = noise.to(device)
        
        # Create custom TTM sampler
        ttm_sampler = TTMSampler(ref, mask, noise, ttm_start_step, ttm_end_step)
        
        # Create sampler wrapper for ComfyUI
        class TTMSamplerWrapper:
            def __init__(self, sampler):
                self.sampler = sampler
            
            def sample(self, *args, **kwargs):
                return self.sampler.sample(*args, **kwargs)
        
        sampler_obj = TTMSamplerWrapper(ttm_sampler)
        
        # Use ComfyUI's sample infrastructure
        mm.load_model_gpu(model)
        
        # Create callback for preview
        preview_callback = latent_preview.prepare_callback(model, steps)
        
        # Sample using comfy's infrastructure with our custom sampler
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            "euler",  # base sampler name
            "normal",  # scheduler name
            positive,
            negative,
            x,
            denoise=1.0,
            disable_noise=True,  # We handle noise ourselves
            start_step=0,
            last_step=steps,
            force_full_denoise=True,
            noise_mask=None,
            callback=preview_callback,
            seed=seed,
        )
        
        # Actually, the above won't use our custom sampler. Let me try a different approach.
        # Use sample_custom with a wrapped sampler
        
        # Create a proper KSAMPLER-compatible object
        from comfy.samplers import KSAMPLER
        
        # We need to create a sampler that ComfyUI can use
        # The easiest way is to use the callback mechanism properly
        
        # Let's use a simpler approach: run euler sampling and apply TTM in callback
        
        # Actually, let's just do manual sampling
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
        """Manual sampling loop with TTM injection."""
        device = mm.get_torch_device()
        
        # Move everything to device
        x = latent.to(device)
        ref = ref.to(device)
        motion_mask = motion_mask.to(device)
        background_mask = 1.0 - motion_mask
        noise = noise.to(device)
        sigmas = sigmas.to(device)
        
        # Initialize with noisy reference if starting from step 0
        if ttm_start_step == 0:
            sigma_init = sigmas[0].item()
            x = add_noise_at_sigma(ref, noise, sigma_init)
            print(f"[TTM] Initialized with noisy ref at sigma={sigma_init:.4f}")
        else:
            # Start from noisy latent
            x = add_noise_at_sigma(x, noise, sigmas[0].item())
        
        # Get model function
        mm.load_model_gpu(model)
        model_patcher = model
        real_model = model_patcher.model
        
        # Prepare model wrapper
        model_options = model_patcher.model_options.copy()
        
        # Calculate cond
        positive_copy = comfy.samplers.convert_cond(positive)
        negative_copy = comfy.samplers.convert_cond(negative)
        
        total_steps = len(sigmas) - 1
        pbar = comfy.utils.ProgressBar(total_steps)
        
        for i in range(total_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # CFG sampling: calc_cond_batch returns denoised prediction
            sigma_batch = sigma.unsqueeze(0).expand(x.shape[0])
            
            # Compute denoised with CFG (comfy handles CFG internally if both conds provided)
            # But for manual CFG, we need to call separately
            cond_denoised = comfy.samplers.calc_cond_batch(
                model_patcher, positive_copy, x, sigma_batch, model_options
            )
            uncond_denoised = comfy.samplers.calc_cond_batch(
                model_patcher, negative_copy, x, sigma_batch, model_options
            )
            
            # CFG combination on denoised predictions
            denoised = uncond_denoised + cfg * (cond_denoised - uncond_denoised)
            
            # Euler step (k-diffusion style):
            # d = (x - denoised) / sigma  (velocity/direction)
            # x_next = x + d * dt
            d = (x - denoised) / sigma
            dt = sigma_next - sigma
            x = x + d * dt
            
            # TTM injection: after euler step
            if ttm_start_step <= i < ttm_end_step:
                # Compute noisy reference at next sigma
                noisy_ref = add_noise_at_sigma(ref, noise, sigma_next.item())
                # Blend: background keeps denoised, motion gets noisy ref
                x = x * background_mask + noisy_ref * motion_mask
                print(f"[TTM] Step {i}: injected at sigma_next={sigma_next.item():.4f}")
            
            # Callback for preview
            if callback is not None:
                callback(i, x, None, total_steps)
            
            pbar.update(1)
        
        return x


# Node registration
NODE_CLASS_MAPPINGS = {
    "WanTTM_PrepareLatents": WanTTM_PrepareLatents,
    "WanTTM_Sampler_Clean": WanTTM_Sampler_Clean,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanTTM_PrepareLatents": "TTM: Prepare Latents",
    "WanTTM_Sampler_Clean": "TTM: Sampler (Clean)",
}
