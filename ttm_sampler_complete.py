"""Fixed TTM Sampler - Simplified and correct logic based on WanMoEKSampler."""

import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_sampling
from comfy_api.latest import io


def set_shift(model, sigma_shift):
    """Set sigma shift for flow matching models."""
    model_sampling = model.get_model_object("model_sampling")
    if not model_sampling:
        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST
        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=sigma_shift, multiplier=1000)
    model.add_object_patch("model_sampling", model_sampling)
    return model


class WanTTMSamplerComplete(io.ComfyNode):
    """Complete TTM sampler with dual-model MoE and dual-clock denoising."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanTTMSamplerComplete",
            category="sampling",
            description="Complete TTM sampler with MoE and dual-clock denoising",
            inputs=[
                # Models (MoE architecture)
                io.Model.Input("model_high", tooltip="High-noise model"),
                io.Model.Input("model_low", tooltip="Low-noise model"),
                
                # Conditioning
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                
                # Basic sampling
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("steps", default=50, min=10, max=150, step=1,
                           tooltip="Sampling steps (typically 30-80)"),
                io.Float.Input("cfg", default=5.0, min=0.0, max=15.0, step=0.1,
                             tooltip="CFG scale (typically 3.0-7.0)"),
                
                # Sampler config
                io.String.Input("sampler_name", default="euler"),
                io.String.Input("scheduler", default="normal"),
                
                # MoE parameters (slider-optimized)
                io.Float.Input("boundary", default=0.875, min=0.5, max=1.0, step=0.005,
                             tooltip="Model switch point (0.85-0.92 typical)"),
                io.Float.Input("sigma_shift", default=8.0, min=1.0, max=20.0, step=0.5,
                             tooltip="Flow matching shift (6.0-10.0 typical)"),
                
                # Advanced
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.05,
                             tooltip="Denoise strength (1.0 = full)"),
            ],
            outputs=[
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_high,
        model_low,
        positive,
        negative,
        latent_image,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        boundary: float,
        sigma_shift: float,
        denoise: float,
    ) -> io.NodeOutput:
        
        # Extract TTM parameters from conditioning
        ttm_params = cls._extract_ttm_params(positive)
        
        # Clone and configure models
        high_noise_model = set_shift(model_high.clone(), sigma_shift)
        low_noise_model = set_shift(model_low.clone(), sigma_shift)
        
        # Perform TTM sampling
        if ttm_params is not None:
            # TTM dual-clock sampling
            result = cls._ttm_dual_clock_sample(
                high_noise_model, low_noise_model,
                positive, negative, latent_image,
                seed, steps, cfg,
                sampler_name, scheduler,
                boundary, denoise,
                ttm_params
            )
        else:
            # Standard MoE sampling (fallback)
            result = cls._moe_sample(
                high_noise_model, low_noise_model,
                positive, negative, latent_image,
                seed, steps, cfg,
                sampler_name, scheduler,
                boundary, denoise
            )
        
        return io.NodeOutput(result)
    
    @classmethod
    def _extract_ttm_params(cls, conditioning):
        """Extract TTM parameters from conditioning."""
        if not conditioning or len(conditioning) == 0:
            return None
        
        cond = conditioning[0]
        if len(cond) < 2:
            return None
        
        cond_dict = cond[1]
        
        # Check for TTM params
        if "concat_latent_image" not in cond_dict:
            return None
        
        return {
            "motion_latent": cond_dict.get("concat_latent_image"),
            "motion_mask": cond_dict.get("motion_signal_mask"),
            "tweak_index": cond_dict.get("ttm_tweak_index", 3),
            "tstrong_index": cond_dict.get("ttm_tstrong_index", 7),
        }
    
    @classmethod
    def _ttm_dual_clock_sample(
        cls, model_high, model_low,
        positive, negative, latent_dict,
        seed, steps, cfg,
        sampler_name, scheduler,
        boundary, denoise,
        ttm_params
    ):
        """TTM dual-clock sampling - simplified logic based on WanMoEKSampler."""
        
        latent_image = latent_dict["samples"]
        noise = comfy.sample.prepare_noise(latent_image, seed, latent_dict.get("batch_index", None))
        noise_mask = latent_dict.get("noise_mask", None)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        # Get TTM parameters
        device = latent_image.device
        motion_latent = ttm_params["motion_latent"].to(device)
        motion_mask = ttm_params["motion_mask"].to(device)
        tweak_index = ttm_params["tweak_index"]
        tstrong_index = ttm_params["tstrong_index"]
        
        # Prepare masks for blending
        motion_mask_5d = motion_mask.unsqueeze(1).expand_as(latent_image)
        background_mask = 1.0 - motion_mask_5d
        
        # Generate fixed noise for motion injection
        generator = torch.Generator(device=device).manual_seed(seed)
        fixed_noise = torch.randn_like(motion_latent, generator=generator)
        
        # Calculate sigmas and MoE switching point
        model_sampling = model_high.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        timesteps = [model_sampling.timestep(sigma) for sigma in sigmas]
        
        # Find MoE switching step
        moe_switch = steps
        for i, t in enumerate(timesteps):
            if t / 1000.0 < boundary:
                moe_switch = i
                break
        
        print(f"[TTM-MoE] MoE Switch: {moe_switch}/{steps} (boundary={boundary:.3f})")
        print(f"[TTM-MoE] Dual-Clock: tweak={tweak_index}, tstrong={tstrong_index}")
        
        # Prepare all checkpoints
        checkpoints = sorted(set([0, tweak_index, tstrong_index, moe_switch, steps]))
        print(f"[TTM-MoE] Checkpoints: {checkpoints}")
        
        latent_image = comfy.sample.fix_empty_latent_channels(model_high, latent_image)
        
        # Process each segment
        for i in range(len(checkpoints) - 1):
            start = checkpoints[i]
            end = checkpoints[i + 1]
            
            if start >= end:
                continue
            
            # Select model based on MoE boundary
            if start < moe_switch:
                current_model = model_high
                model_name = "high"
            else:
                # Switch to low-noise model
                if start == moe_switch:
                    latent_image = comfy.sample.fix_empty_latent_channels(model_low, latent_image)
                current_model = model_low
                model_name = "low"
            
            # Determine if this is the final segment
            is_final = (end == steps)
            
            # Sample this segment
            print(f"[TTM-MoE] Segment {start}->{end}: {model_name} model, final={is_final}")
            latent_image = comfy.sample.sample(
                current_model, noise, steps, cfg,
                sampler_name, scheduler,
                positive, negative, latent_image,
                denoise=denoise, disable_noise=False,
                start_step=start, last_step=end,
                force_full_denoise=is_final,
                noise_mask=noise_mask,
                disable_pbar=disable_pbar, seed=seed
            )
            
            # Apply TTM blending if in dual-clock range
            if tweak_index <= end <= tstrong_index:
                print(f"[TTM-MoE] Applying TTM blend at step {end}")
                
                # Calculate noise level at current step
                current_sigma = sigmas[end]
                
                # Create noisy motion reference
                noisy_motion = motion_latent + fixed_noise * current_sigma
                
                # Blend: background (denoised) + motion (noisy reference)
                latent_image = latent_image * background_mask + noisy_motion * motion_mask_5d
        
        out = latent_dict.copy()
        out["samples"] = latent_image
        return out
    
    @classmethod
    def _moe_sample(
        cls, model_high, model_low,
        positive, negative, latent_dict,
        seed, steps, cfg,
        sampler_name, scheduler,
        boundary, denoise
    ):
        """Standard MoE sampling without TTM (fallback) - same as WanMoEKSampler."""
        
        latent_image = latent_dict["samples"]
        noise = comfy.sample.prepare_noise(latent_image, seed, latent_dict.get("batch_index", None))
        noise_mask = latent_dict.get("noise_mask", None)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        model_sampling = model_high.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        timesteps = [model_sampling.timestep(sigma) for sigma in sigmas]
        
        switching_step = steps
        for i, t in enumerate(timesteps):
            if t / 1000.0 < boundary:
                switching_step = i
                break
        
        print(f"[MoE] Switch@{switching_step}/{steps} High:0-{switching_step} Low:{switching_step}-{steps}")
        
        latent_image = comfy.sample.fix_empty_latent_channels(model_high, latent_image)
        
        # High-noise phase
        if switching_step > 0:
            latent_image = comfy.sample.sample(
                model_high, noise, steps, cfg,
                sampler_name, scheduler,
                positive, negative, latent_image,
                denoise=denoise, disable_noise=False,
                start_step=0, last_step=switching_step,
                force_full_denoise=True,
                noise_mask=noise_mask,
                disable_pbar=disable_pbar, seed=seed
            )
        
        latent_image = comfy.sample.fix_empty_latent_channels(model_low, latent_image)
        
        # Low-noise phase
        if switching_step < steps:
            latent_image = comfy.sample.sample(
                model_low, noise, steps, cfg,
                sampler_name, scheduler,
                positive, negative, latent_image,
                denoise=denoise, disable_noise=False,
                start_step=switching_step, last_step=steps,
                force_full_denoise=True,
                noise_mask=noise_mask,
                disable_pbar=disable_pbar, seed=seed
            )
        
        out = latent_dict.copy()
        out["samples"] = latent_image
        return out
