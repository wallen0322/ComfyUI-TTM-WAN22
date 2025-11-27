"""
FlowMatch Euler utilities for Wan TTM nodes.

Key design:
- Pure sigma-based sampling (no timestep conversion)
- TTM injection uses sigma directly: noisy = (1-sigma)*clean + sigma*noise
- Callback receives full context for TTM dual-clock logic
"""

from __future__ import annotations

import torch
from typing import Optional, Callable, Dict, Any

import comfy.samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy import sample as comfy_sample

FLOW_SCHEDULER_NAME = "flow_euler_ttm"


def generate_wan_sigmas(model_sampling, steps: int) -> torch.Tensor:
    """Generate Wan FlowMatch sigma schedule.
    
    Uses model's percent_to_sigma if available, otherwise falls back to
    manual FlowMatch schedule with shift.
    """
    if hasattr(model_sampling, "percent_to_sigma"):
        eps = 1e-6
        percents = torch.linspace(1.0 - eps, eps, steps, dtype=torch.float32)
        sigmas = []
        for pct in percents:
            pct_value = float(min(max(pct, eps), 1.0 - eps))
            sigma = float(model_sampling.percent_to_sigma(pct_value))
            sigmas.append(sigma)
        sigmas.append(0.0)
        return torch.tensor(sigmas, dtype=torch.float32)

    # Fallback: manual Wan FlowMatch schedule
    sigma_max = float(getattr(model_sampling, "sigma_max", 1.0))
    sigma_min = float(getattr(model_sampling, "sigma_min", 0.003 / 1.002))
    shift = float(getattr(model_sampling, "flow_shift", 3.0))
    train_steps = int(getattr(model_sampling, "flow_train_steps", 1000))

    full_sigmas = torch.linspace(sigma_max, sigma_min, train_steps, dtype=torch.float32)
    stride = len(full_sigmas) / max(steps, 1)
    sampled = [full_sigmas[min(int(round(i * stride)), len(full_sigmas) - 1)] for i in range(steps)]
    sampled.append(torch.tensor(0.0, dtype=torch.float32))
    sigmas = torch.stack(sampled)
    # Apply shift (standard Wan FlowMatch)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas


def register_flow_scheduler():
    """Register the TTM flow scheduler with ComfyUI."""
    if FLOW_SCHEDULER_NAME in comfy.samplers.SCHEDULER_HANDLERS:
        return
    handler = lambda model_sampling, steps: generate_wan_sigmas(model_sampling, steps)
    comfy.samplers.SCHEDULER_HANDLERS[FLOW_SCHEDULER_NAME] = comfy.samplers.SchedulerHandler(handler)
    if FLOW_SCHEDULER_NAME not in comfy.samplers.SCHEDULER_NAMES:
        comfy.samplers.SCHEDULER_NAMES.append(FLOW_SCHEDULER_NAME)
    if FLOW_SCHEDULER_NAME not in comfy.samplers.KSampler.SCHEDULERS:
        comfy.samplers.KSampler.SCHEDULERS = comfy.samplers.SCHEDULER_NAMES


def add_noise_at_sigma(clean: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add noise to clean latent at given sigma level.
    
    FlowMatch formula: noisy = (1 - sigma) * clean + sigma * noise
    
    This is the core TTM operation - we use sigma directly, no timestep conversion.
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=clean.device, dtype=clean.dtype)
    # Expand sigma to match latent dimensions [B,C,T,H,W]
    while sigma.ndim < clean.ndim:
        sigma = sigma.unsqueeze(-1)
    sigma = sigma.to(device=clean.device, dtype=clean.dtype)
    return (1.0 - sigma) * clean + sigma * noise


class FlowEulerSamplerTTM(comfy.samplers.Sampler):
    """FlowMatch Euler sampler with TTM support.
    
    Key features:
    - Passes full sigmas array to callback for TTM sigma lookup
    - Callback can modify x in-place after each Euler step
    - Supports MoE two-phase sampling (disable_noise for phase continuation)
    """

    def __init__(self, disable_noise: bool = False, force_zero_end: bool = False):
        self.disable_noise = disable_noise
        self.force_zero_end = force_zero_end

    def sample(
        self,
        model_wrap,
        sigmas: torch.Tensor,
        extra_args: Dict[str, Any],
        callback: Optional[Callable],
        noise: torch.Tensor,
        latent_image: Optional[torch.Tensor] = None,
        denoise_mask: Optional[torch.Tensor] = None,
        disable_pbar: bool = False,
    ) -> torch.Tensor:
        extra_args = dict(extra_args)
        extra_args["denoise_mask"] = denoise_mask
        
        model_k = comfy.samplers.KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise

        # Initialize x
        if not self.disable_noise:
            # Phase 1 start: apply noise scaling
            x = model_wrap.inner_model.model_sampling.noise_scaling(
                sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas)
            )
        else:
            # Phase 2 or TTM continuation: use latent_image directly
            x = latent_image if latent_image is not None else noise

        total_steps = len(sigmas) - 1
        s_in = x.new_ones([x.shape[0]])

        for i in k_diffusion_sampling.trange(total_steps, disable=disable_pbar):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Model forward pass
            denoised = model_k(
                x, sigma * s_in,
                denoise_mask=extra_args.get("denoise_mask"),
                model_options=extra_args.get("model_options"),
                seed=extra_args.get("seed")
            )
            
            # Euler step: x_next = x + (denoised - x) / sigma * (sigma_next - sigma)
            # Simplified: x_next = x + noise_pred * dt
            sigma_t = sigma if torch.is_tensor(sigma) else torch.tensor(sigma, device=x.device, dtype=x.dtype)
            while sigma_t.ndim < x.ndim:
                sigma_t = sigma_t.unsqueeze(-1)
            
            noise_pred = (x - denoised) / sigma_t.clamp(min=1e-5)
            dt = sigma_next - sigma
            x = x + noise_pred * dt
            
            # Callback for TTM injection and preview
            # Pass full context so callback can compute noisy reference correctly
            if callback is not None:
                callback({
                    "i": i,                    # Step index within this phase
                    "x": x,                    # Latent AFTER Euler step (modifiable in-place)
                    "denoised": denoised,      # Model prediction (for preview)
                    "sigma": sigma,            # Current sigma
                    "sigma_next": sigma_next,  # Next sigma (use this for TTM noisy ref)
                    "sigmas": sigmas,          # Full sigma array for this phase
                    "total_steps": total_steps,
                })

        # Final inverse scaling
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], x)
        return samples


def sample_flowmatch_ttm(
    model,
    noise: torch.Tensor,
    cfg: float,
    sigmas: torch.Tensor,
    positive,
    negative,
    latent_image: torch.Tensor,
    disable_noise: bool = False,
    force_zero_end: bool = False,
    noise_mask: Optional[torch.Tensor] = None,
    callback: Optional[Callable] = None,
    disable_pbar: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample using FlowMatch Euler with TTM support.
    
    Args:
        model: ComfyUI model wrapper
        noise: Random noise tensor (same shape as latent)
        cfg: Classifier-free guidance scale
        sigmas: Sigma schedule array (length = steps + 1, ending with 0)
        positive: Positive conditioning
        negative: Negative conditioning
        latent_image: Initial latent (may be pre-noised for TTM)
        disable_noise: If True, use latent_image directly (for phase 2)
        force_zero_end: Ensure final sigma is 0
        noise_mask: Optional inpainting mask
        callback: TTM callback function
        disable_pbar: Disable progress bar
        seed: Random seed
    
    Returns:
        Denoised latent tensor
    """
    sampler = FlowEulerSamplerTTM(disable_noise=disable_noise, force_zero_end=force_zero_end)
    return comfy_sample.sample_custom(
        model=model,
        noise=noise,
        cfg=cfg,
        sampler=sampler,
        sigmas=sigmas,
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )


# Register on module load
register_flow_scheduler()
