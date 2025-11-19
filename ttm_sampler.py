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
                # Models
                io.Model.Input("model_high", tooltip="High-noise model (early steps)"),
                io.Model.Input("model_low", tooltip="Low-noise model (later steps)"),
                
                # Conditioning
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                
                # Basic Sampling
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("steps", default=50, min=1, max=10000, step=1,
                           tooltip="Total steps (1-8 distilled / 30-80 normal)"),
                io.Float.Input("cfg_high", default=5.0, min=0.0, max=30.0, step=0.1,
                             tooltip="CFG scale for high-noise phase"),
                io.Float.Input("cfg_low", default=4.0, min=0.0, max=30.0, step=0.1,
                             tooltip="CFG scale for low-noise phase"),
                io.Combo.Input("sampler_name", list(comfy.samplers.KSampler.SAMPLERS), 
                             default="euler", tooltip="Sampling algorithm"),
                io.Combo.Input("scheduler", list(comfy.samplers.KSampler.SCHEDULERS),
                             default="normal", tooltip="Noise scheduler"),
                
                # MoE Settings
                io.Float.Input("boundary", default=0.875, min=0.0, max=1.0, step=0.005,
                             tooltip="Model switch timestep (0.875=t2v, 0.9=i2v)"),
                io.Float.Input("sigma_shift", default=8.0, min=0.0, max=100.0, step=0.5,
                             tooltip="Flow matching shift (6-10 typical)"),
                             
                # Advanced
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                             tooltip="Denoise strength (1.0 = full denoise)"),
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
        cfg_high: float,
        cfg_low: float,
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
                seed, steps, (cfg_high, cfg_low),
                sampler_name, scheduler,
                boundary, denoise,
                ttm_params
            )
        else:
            # Standard MoE sampling (fallback)
            result = cls._moe_sample(
                high_noise_model, low_noise_model,
                positive, negative, latent_image,
                seed, steps, (cfg_high, cfg_low),
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
    def _move_conds_to_device(cls, conds, device):
        """Recursively move conditioning tensors to device."""
        if conds is None:
            return None
        new_conds = []
        for cond in conds:
            new_cond = []
            for item in cond:
                if isinstance(item, dict):
                    new_dict = {}
                    for k, v in item.items():
                        if isinstance(v, torch.Tensor):
                            new_dict[k] = v.to(device)
                        else:
                            new_dict[k] = v
                    new_cond.append(new_dict)
                elif isinstance(item, torch.Tensor):
                    new_cond.append(item.to(device))
                else:
                    new_cond.append(item)
            new_conds.append(new_cond)
        return new_conds

    @classmethod
    def _ttm_dual_clock_sample(
        cls, model_high, model_low,
        positive, negative, latent_dict,
        seed, steps, cfgs,
        sampler_name, scheduler,
        boundary, denoise,
        ttm_params
    ):
        """TTM dual-clock sampling with manual Euler loop for Phase 2.
        
        Correctly implements TTM by manually stepping through [tweak, tstrong]
        using Euler method and applying blend after each step.
        """
        
        latent_image = latent_dict["samples"]
        batch_inds = latent_dict.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
        noise_mask = latent_dict.get("noise_mask", None)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        # Get TTM parameters
        device = latent_image.device
        motion_latent = ttm_params["motion_latent"].to(device)
        motion_mask = ttm_params["motion_mask"].to(device)
        tweak_index = ttm_params["tweak_index"]
        tstrong_index = ttm_params["tstrong_index"]
        
        # Ensure conditioning tensors live on the same device
        positive = cls._move_conds_to_device(positive, device)
        negative = cls._move_conds_to_device(negative, device)
        
        # Expand motion_mask to match latent shape
        motion_mask_5d = motion_mask.unsqueeze(1).expand_as(latent_image).to(device)
        background_mask = (1.0 - motion_mask_5d).to(device)
        
        # Generate fixed noise for motion
        generator = torch.Generator(device="cpu").manual_seed(seed)
        motion_noise = torch.randn(motion_latent.shape, generator=generator, 
                                  device="cpu", dtype=motion_latent.dtype).to(device)
        
        # Calculate sigmas
        sampling = model_high.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, steps).to(device)
        timesteps = [sampling.timestep(sigma) / 1000.0 for sigma in sigmas.tolist()]
        
        # Find MoE switching step
        switching_step = steps
        for i, t in enumerate(timesteps[1:]):
            if t < boundary:
                switching_step = i
                break
        
        print(f"[TTM] MoE switching at step {switching_step}/{steps}")
        print(f"[TTM] TTM range: [{tweak_index}, {tstrong_index})")
        
        # Phase 2: [tweak, tstrong) - Manual Euler with TTM blending
        if tweak_index < tstrong_index:
            print(f"[TTM] Phase 2: [{tweak_index}, {tstrong_index}) TTM dual-clock (manual Euler)")
            
            # Prepare for manual sampling directly from motion reference
            noise = noise.to(device)
            motion_latent = motion_latent.to(device)
            phase_start = max(0, min(tweak_index, len(sigmas) - 2))
            phase_end = max(phase_start + 1, min(tstrong_index, len(sigmas) - 1))
            phase2_sigmas = sigmas[phase_start:phase_end + 1]
            if phase2_sigmas.shape[0] < 2:
                raise ValueError("Invalid TTM sigma range; ensure tweak/tstrong are within total steps")
            
            # Determine starting model
            current_model = model_high if phase_start < switching_step else model_low
            current_cfg = cfgs[0] if phase_start < switching_step else cfgs[1]
            
            # Prepare model and conditioning
            from comfy.samplers import CFGGuider, process_conds
            from comfy.sampler_helpers import prepare_sampling, cleanup_models, convert_cond
            
            # Convert conditioning to proper format
            conds = {
                "positive": convert_cond(positive),
                "negative": convert_cond(negative)
            }
            
            # Initialize for sampling
            inner_model, conds, loaded_models = prepare_sampling(
                current_model, motion_latent.shape, 
                conds,
                current_model.model_options
            )
            
            # Process conds with original noise/latent
            conds_processed = process_conds(
                inner_model, noise, conds, device,
                latent_image=motion_latent, denoise_mask=noise_mask, seed=seed
            )
            
            # Set up patcher for model
            current_model.pre_run()
            
            # Update model sampling
            model_sampling = inner_model.model_sampling
            
            # Initialize latent directly from motion reference at tweak sigma
            x = model_sampling.noise_scaling(phase2_sigmas[0], noise, motion_latent)
            
            s_in = x.new_ones([x.shape[0]])
            
            for i in range(len(phase2_sigmas) - 1):
                actual_step = phase_start + i
                sigma = phase2_sigmas[i]
                sigma_next = phase2_sigmas[i + 1]
                
                # Check for MoE switch
                if actual_step == switching_step and actual_step < phase_end:
                    print(f"[TTM] MoE switch at step {actual_step}")
                    # Cleanup previous model
                    current_model.cleanup()
                    cleanup_models(conds_processed, loaded_models)
                    
                    current_model = model_low
                    current_cfg = cfgs[1]
                    
                    # Prepare new model
                    conds = {
                        "positive": convert_cond(positive),
                        "negative": convert_cond(negative)
                    }
                    
                    inner_model, conds, loaded_models = prepare_sampling(
                        current_model, motion_latent.shape,
                        conds,
                        current_model.model_options
                    )
                    
                    # Process conds with original noise/latent
                    conds_processed = process_conds(
                        inner_model, noise, conds, device,
                        latent_image=motion_latent, denoise_mask=noise_mask, seed=seed
                    )
                    
                    # Set up patcher for new model
                    current_model.pre_run()
                    
                    # Update model sampling
                    model_sampling = inner_model.model_sampling
                
                # Predict noise using ComfyUI's sampling function
                from comfy.samplers import sampling_function
                denoised = sampling_function(
                    inner_model, x, sigma * s_in,
                    conds_processed.get("negative", None),
                    conds_processed.get("positive", None),
                    current_cfg,
                    model_options=current_model.model_options,
                    seed=seed
                )
                
                # Euler step
                d = (x - denoised) / sigma
                dt = sigma_next - sigma
                x = x + d * dt
                
                # TTM blend: background (current denoised) + motion (noisy at next sigma)
                # Note: TTM blend happens in X space (noisy)
                noisy_motion = (motion_latent + motion_noise * sigma_next).to(device)
                x = x * background_mask + noisy_motion * motion_mask_5d
            
            # Apply inverse noise scaling (X -> Latent) at start of Phase 3
            latent_image = model_sampling.inverse_noise_scaling(phase2_sigmas[-1], x)
            
            # Cleanup Phase 2
            current_model.cleanup()
            cleanup_models(conds_processed, loaded_models)
        
        # Phase 3: [tstrong, steps) - Standard sampling with MoE
        if tstrong_index < steps:
            # Handle MoE switching in Phase 3
            if tstrong_index < switching_step:
                # Part 3a: High model portion
                high_end = min(switching_step, steps)
                print(f"[TTM] Phase 3a: [{tstrong_index}, {high_end}) high model")
                latent_image = comfy.sample.fix_empty_latent_channels(model_high, latent_image)
                latent_image = comfy.sample.sample(
                    model_high, noise, steps, cfgs[0],
                    sampler_name, scheduler,
                    positive, negative, latent_image,
                    denoise=denoise,
                    disable_noise=False,
                    start_step=tstrong_index,
                    last_step=high_end,
                    force_full_denoise=(high_end < steps), # Only full denoise if it's the very end
                    noise_mask=noise_mask,
                    callback=None,
                    disable_pbar=disable_pbar,
                    seed=seed
                )
            
            if switching_step < steps:
                # Part 3b: Low model portion
                low_start = max(tstrong_index, switching_step)
                print(f"[TTM] Phase 3b: [{low_start}, {steps}) low model")
                latent_image = comfy.sample.fix_empty_latent_channels(model_low, latent_image)
                latent_image = comfy.sample.sample(
                    model_low, noise, steps, cfgs[1],
                    sampler_name, scheduler,
                    positive, negative, latent_image,
                    denoise=denoise,
                    disable_noise=False,
                    start_step=low_start,
                    last_step=steps,
                    force_full_denoise=True,
                    noise_mask=noise_mask,
                    callback=None,
                    disable_pbar=disable_pbar,
                    seed=seed
                )
        
        out = latent_dict.copy()
        out["samples"] = latent_image
        return out
    
    @classmethod
    def _moe_sample(
        cls, model_high, model_low,
        positive, negative, latent_dict,
        seed, steps, cfgs,
        sampler_name, scheduler,
        boundary, denoise
    ):
        """Standard MoE sampling without TTM (fallback) - based on WanMoeKSampler."""
        
        latent_image = latent_dict["samples"]
        batch_inds = latent_dict.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
        noise_mask = latent_dict.get("noise_mask", None)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        # Calculate switching point (based on WanMoeKSampler)
        sampling = model_high.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, steps)
        timesteps = [sampling.timestep(sigma) / 1000.0 for sigma in sigmas.tolist()]
        
        switching_step = steps
        for i, t in enumerate(timesteps[1:]):
            if t < boundary:
                switching_step = i
                break
        
        print(f"[MoE] Switching at step {switching_step}/{steps}")
        
        start_step = 0
        last_step = steps
        start_with_high = start_step < switching_step
        end_with_low = last_step >= switching_step
        
        print(f"[MoE] start_with_high={start_with_high}, end_with_low={end_with_low}")
        
        # High-noise phase
        if start_with_high:
            print(f"[MoE] Running high-noise model...")
            latent_image = comfy.sample.fix_empty_latent_channels(model_high, latent_image)
            latent_image = comfy.sample.sample(
                model_high, noise, steps, cfgs[0],
                sampler_name, scheduler,
                positive, negative, latent_image,
                denoise=denoise,
                disable_noise=end_with_low,
                start_step=0,
                last_step=switching_step,
                force_full_denoise=end_with_low,
                noise_mask=noise_mask,
                disable_pbar=disable_pbar,
                seed=seed
            )
        
        # Low-noise phase
        if end_with_low:
            print(f"[MoE] Running low-noise model...")
            latent_image = comfy.sample.fix_empty_latent_channels(model_low, latent_image)
            latent_image = comfy.sample.sample(
                model_low, noise, steps, cfgs[1],
                sampler_name, scheduler,
                positive, negative, latent_image,
                denoise=denoise,
                disable_noise=False,
                start_step=switching_step,
                last_step=steps,
                force_full_denoise=True,
                noise_mask=noise_mask,
                disable_pbar=disable_pbar,
                seed=seed
            )
        
        out = latent_dict.copy()
        out["samples"] = latent_image
        return out
