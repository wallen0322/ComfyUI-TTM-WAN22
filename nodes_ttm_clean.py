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
                "model_high": ("MODEL", {"tooltip": "High noise model (MoE block 0)"}),
                "model_low": ("MODEL", {"tooltip": "Low noise model (MoE block 1)"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "switch_step": ("INT", {"default": 6, "min": 1, "max": 100, "tooltip": "Step to switch from high to low noise model"}),
                "ttm_start_step": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Step to start TTM"}),
                "ttm_end_step": ("INT", {"default": 6, "min": 1, "max": 100, "tooltip": "Step to end TTM (must be <= switch_step)"}),
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
        model_high,
        model_low,
        positive,
        negative,
        latent,
        seed: int,
        steps: int,
        cfg: float,
        switch_step: int,
        ttm_start_step: int,
        ttm_end_step: int,
        reference_latents=None,
        motion_mask=None,
    ):
        device = mm.get_torch_device()
        
        # Check if TTM is enabled
        ttm_enabled = reference_latents is not None and motion_mask is not None
        
        # Validate TTM range - must complete before MoE switch
        ttm_end_step = min(ttm_end_step, switch_step)
        
        print(f"\n[TTM Clean] === MoE TTM Sampling ===")
        print(f"[TTM Clean] Steps: {steps}, CFG: {cfg}")
        print(f"[TTM Clean] MoE switch at step: {switch_step}")
        print(f"[TTM Clean] Phase 1 (HIGH): steps [0, {switch_step})")
        print(f"[TTM Clean] Phase 2 (LOW): steps [{switch_step}, {steps})")
        print(f"[TTM Clean] TTM: {'ENABLED' if ttm_enabled else 'DISABLED'}")
        if ttm_enabled:
            print(f"[TTM Clean] TTM range: [{ttm_start_step}, {ttm_end_step})")
        print(f"[TTM Clean] Seed: {seed}")
        
        # Get latent info
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
        
        # Generate fixed noise (same for both phases)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn(x.shape, generator=generator, device='cpu')
        
        # Get sigmas from model
        sigmas = comfy.samplers.calculate_sigmas(
            model_high.get_model_object("model_sampling"),
            "normal",  # scheduler
            steps
        ).to(device)
        print(f"[TTM Clean] Sigmas: {sigmas[:8].tolist()}")
        
        # Prepare TTM components
        if ttm_enabled:
            # Ensure ref matches latent spatial dimensions
            if ref.shape[-2:] != x.shape[-2:]:
                print(f"[TTM] Resizing ref from {ref.shape[-2:]} to {x.shape[-2:]}")
                B, C, T, H_ref, W_ref = ref.shape
                H_lat, W_lat = x.shape[-2:]
                ref_flat = ref.view(-1, 1, H_ref, W_ref)
                ref_flat = F.interpolate(ref_flat, size=(H_lat, W_lat), mode='bilinear', align_corners=False)
                ref = ref_flat.view(B, C, T, H_lat, W_lat)
            
            # Ensure mask matches latent spatial dimensions
            if mask.shape[-2:] != x.shape[-2:]:
                print(f"[TTM] Resizing mask from {mask.shape[-2:]} to {x.shape[-2:]}")
                if mask.dim() == 5:
                    B, C, T, H, W = mask.shape
                    mask_flat = mask.view(B * C * T, 1, H, W)
                    mask_flat = F.interpolate(mask_flat, size=x.shape[-2:], mode='bilinear', align_corners=False)
                    mask = mask_flat.view(B, C, T, *x.shape[-2:])
            
            ref = ref.to(device)
            mask = mask.to(device)
            background_mask = 1.0 - mask
            fixed_noise = noise.to(device)
            
            # ===== TTM Initialization =====
            # CRITICAL: KJ's original uses simple add_noise_at_sigma (normalized space)
            # NOT noise_scaling - this matches TTMSampler implementation
            sigma_init = sigmas[0].item()
            
            # Motion region: noisy reference using simple add_noise_at_sigma
            noisy_ref_init = add_noise_at_sigma(ref, fixed_noise, sigma_init)
            # Background: pure noise using simple add_noise_at_sigma
            noisy_bg_init = add_noise_at_sigma(torch.zeros_like(ref), fixed_noise, sigma_init)
            
            # Blend: background gets pure noise, motion gets noisy reference
            x_init = noisy_bg_init * background_mask + noisy_ref_init * mask
            x = x_init.cpu()
            print(f"[TTM] Initialized latent: bg=noise, motion=noisy_ref at sigma={sigma_init:.4f} (normalized space)")
        
        has_phase_2 = switch_step < steps
        
        # ========== Manual Euler Sampling with TTM ==========
        result = cls._manual_euler_ttm(
            model_high=model_high,
            model_low=model_low,
            positive=positive,
            negative=negative,
            latent=x,
            noise=noise,
            sigmas=sigmas,
            cfg=cfg,
            switch_step=switch_step,
            ttm_enabled=ttm_enabled,
            ttm_start_step=ttm_start_step,
            ttm_end_step=ttm_end_step,
            ref=ref if ttm_enabled else None,
            mask=mask if ttm_enabled else None,
            fixed_noise=fixed_noise if ttm_enabled else None,
            device=device,
        )
        
        print(f"\n[TTM Clean] === Sampling Complete ===")
        print(f"[TTM Clean] Output latent range: [{result.min():.3f}, {result.max():.3f}]")
        
        return ({"samples": result.cpu()},)
    
    @classmethod
    def _manual_euler_ttm(
        cls,
        model_high,
        model_low,
        positive,
        negative,
        latent,
        noise,
        sigmas,
        cfg,
        switch_step,
        ttm_enabled,
        ttm_start_step,
        ttm_end_step,
        ref,
        mask,
        fixed_noise,
        device,
    ):
        """Manual step-by-step sampling with TTM injection."""
        steps = len(sigmas) - 1
        
        # Initialize x with TTM
        if ttm_enabled:
            background_mask = 1.0 - mask
            # CRITICAL: comfy.sample.sample with disable_noise=True expects x in scaled space
            # But it will apply inverse_noise_scaling at the end of each step
            # So we create x in scaled space, and it will be unscaled after first step
            sampling_high = model_high.get_model_object("model_sampling")
            sigma_init = sigmas[0].item()
            
            if hasattr(sampling_high, "noise_scaling"):
                sigma_tensor = torch.tensor(sigma_init, device=ref.device, dtype=ref.dtype)
                # Motion region: noisy reference using noise_scaling (in scaled space)
                noisy_ref_scaled = sampling_high.noise_scaling(
                    sigma_tensor,
                    fixed_noise,
                    ref,
                    True  # max_denoise
                )
                # Background: pure noise using noise_scaling (in scaled space)
                noisy_bg_scaled = sampling_high.noise_scaling(
                    sigma_tensor,
                    fixed_noise,
                    torch.zeros_like(ref),
                    True  # max_denoise
                )
                x = noisy_bg_scaled * background_mask + noisy_ref_scaled * mask
                print(f"[TTM] Init: created in SCALED space (will be unscaled after first step)")
            else:
                # Fallback: use add_noise_at_sigma (assumes unscaled space)
                x = fixed_noise * background_mask + add_noise_at_sigma(ref, fixed_noise, sigma_init) * mask
                print(f"[TTM] WARNING: model_sampling.noise_scaling not found, using add_noise_at_sigma")
            
            x = x.cpu()
            print(f"[TTM] Init: motion=noisy_ref, bg=noise at sigma={sigma_init:.4f}, range=[{x.min():.3f}, {x.max():.3f}]")
        else:
            x = latent
            background_mask = None
        
        print(f"[TTM] Starting step-by-step sampling ({steps} steps)")
        
        # CRITICAL: KJ's original implementation uses comfy.sample.sample() directly
        # WITHOUT noise_scaling/inverse_noise_scaling - all operations in normalized space
        # This matches TTMSampler which uses simple add_noise_at_sigma
        
        x = x.to(device)
        
        # Phase 1: High-noise model with TTM
        if switch_step > 0:
            # TTM callback for phase 1
            def ttm_callback_p1(d):
                step_idx = d.get("i", 0)
                x_cb = d["x"]  # x is in normalized space (no scaling)
                sigma_next_cb = d.get("sigma_next", sigmas[step_idx + 1] if step_idx + 1 < len(sigmas) else sigmas[-1])
                
                if ttm_enabled and ttm_start_step <= step_idx < ttm_end_step:
                    # Create noisy_ref in normalized space (like TTMSampler)
                    sigma_val = float(sigma_next_cb) if not torch.is_tensor(sigma_next_cb) else sigma_next_cb.item()
                    noisy_ref = add_noise_at_sigma(ref, fixed_noise, sigma_val)
                    
                    # Blend in normalized space
                    noisy_ref = noisy_ref.to(device=x_cb.device, dtype=x_cb.dtype)
                    bg_mask_cb = background_mask.to(device=x_cb.device, dtype=x_cb.dtype)
                    mot_mask_cb = mask.to(device=x_cb.device, dtype=x_cb.dtype)
                    x_cb.mul_(bg_mask_cb).add_(noisy_ref * mot_mask_cb)
                    print(f"[TTM] Step {step_idx} (HIGH): injected at sigma={sigma_val:.4f}")
            
            x = comfy.sample.sample(
                model_high,
                noise,
                steps,
                cfg,
                "euler",
                "normal",
                positive,
                negative,
                x,
                denoise=1.0,
                disable_noise=ttm_enabled,  # If TTM, x is already initialized
                start_step=0,
                last_step=switch_step,
                force_full_denoise=False,  # Don't apply inverse_noise_scaling yet
                noise_mask=None,
                callback=ttm_callback_p1 if ttm_enabled else None,
                disable_pbar=False,
                seed=0,
            )
        
        # Phase 2: Low-noise model (NO TTM)
        if switch_step < steps:
            x = comfy.sample.sample(
                model_low,
                noise,
                steps,
                cfg,
                "euler",
                "normal",
                positive,
                negative,
                x,
                denoise=1.0,
                disable_noise=True,  # Continue from phase 1
                start_step=switch_step,
                last_step=steps,
                force_full_denoise=True,  # Apply inverse_noise_scaling at the end
                noise_mask=None,
                callback=None,
                disable_pbar=False,
                seed=0,
            )
        
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
