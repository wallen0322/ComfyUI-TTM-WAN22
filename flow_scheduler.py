"""FlowMatch Euler utilities (scheduler + sampler) for Wan TTM nodes."""

from __future__ import annotations

import torch
from typing import Optional

import comfy.samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy import sample as comfy_sample

FLOW_SCHEDULER_NAME = "flow_euler_ttm"
FLOW_SAMPLER_NAME = "flow_euler_sampler_ttm"


def _generate_flow_sigmas(model_sampling, steps: int) -> torch.Tensor:
    """Use the flow percent_to_sigma mapping when available."""
    if hasattr(model_sampling, "percent_to_sigma"):
        eps = 1e-6
        percents = torch.linspace(1.0 - eps, eps, steps, dtype=torch.float32)
        sigmas = []
        for pct in percents:
            pct_value = float(pct)
            pct_value = min(max(pct_value, eps), 1.0 - eps)
            sigma = float(model_sampling.percent_to_sigma(pct_value))
            sigmas.append(sigma)
        sigmas.append(0.0)
        return torch.tensor(sigmas, dtype=torch.float32)

    # Fallback to default discrete scheduler (non-flow models)
    return comfy.samplers.simple_scheduler(model_sampling, steps)


def register_flow_scheduler():
    if FLOW_SCHEDULER_NAME in comfy.samplers.SCHEDULER_HANDLERS:
        return

    handler = lambda model_sampling, steps: _generate_flow_sigmas(model_sampling, steps)
    comfy.samplers.SCHEDULER_HANDLERS[FLOW_SCHEDULER_NAME] = comfy.samplers.SchedulerHandler(handler)
    if FLOW_SCHEDULER_NAME not in comfy.samplers.SCHEDULER_NAMES:
        comfy.samplers.SCHEDULER_NAMES.append(FLOW_SCHEDULER_NAME)
    if FLOW_SCHEDULER_NAME not in comfy.samplers.KSampler.SCHEDULERS:
        comfy.samplers.KSampler.SCHEDULERS = comfy.samplers.SCHEDULER_NAMES


class FlowEulerSampler(comfy.samplers.Sampler):
    """Minimal sampler that mirrors diffusers FlowMatchEulerDiscrete behavior."""

    def __init__(self, disable_noise: bool, force_zero_end: bool):
        self.disable_noise = disable_noise
        self.force_zero_end = force_zero_end

    def sample(
        self,
        model_wrap,
        sigmas: torch.Tensor,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        extra_args["denoise_mask"] = denoise_mask
        model_k = comfy.samplers.KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise

        scaling_latent = latent_image
        if not self.disable_noise:
            scaling_latent = model_wrap.inner_model.model_sampling.noise_scaling(
                sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas)
            )
        elif scaling_latent is None:
            scaling_latent = model_wrap.inner_model.model_sampling.noise_scaling(
                sigmas[0], torch.zeros_like(noise), latent_image, False
            )

        x = scaling_latent
        total_steps = len(sigmas) - 1
        s_in = x.new_ones([x.shape[0]])

        for i in k_diffusion_sampling.trange(total_steps, disable=disable_pbar):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            denoise_mask = extra_args.get("denoise_mask")
            model_options = extra_args.get("model_options")
            seed = extra_args.get("seed")
            denoised = model_k(x, sigma * s_in, denoise_mask=denoise_mask, model_options=model_options, seed=seed)
            sigma_value = sigma
            if not torch.is_tensor(sigma_value):
                sigma_value = torch.tensor(sigma_value, device=x.device, dtype=x.dtype)
            while sigma_value.ndim < x.ndim:
                sigma_value = sigma_value.view(*sigma_value.shape, *([1] * (x.ndim - sigma_value.ndim)))
            noise_pred = (x - denoised) / sigma_value.clamp(min=1e-5)
            dt = sigma_next - sigma
            x = x + noise_pred * dt
            if callback is not None:
                callback({"i": i, "denoised": noise_pred, "x": x, "sigma": sigma, "sigma_next": sigma_next})

        if self.force_zero_end and sigmas[-1] != 0:
            sigmas = sigmas.clone()
            sigmas[-1] = 0.0

        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], x)
        return samples


def sample_flowmatch(
    model,
    noise,
    cfg,
    sigmas: torch.Tensor,
    positive,
    negative,
    latent_image,
    disable_noise: bool,
    force_zero_end: bool,
    noise_mask=None,
    callback=None,
    disable_pbar=False,
    seed: Optional[int] = None,
):
    sampler = FlowEulerSampler(disable_noise=disable_noise, force_zero_end=force_zero_end)
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


register_flow_scheduler()
