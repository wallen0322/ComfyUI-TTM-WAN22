"""
Exact reproduction of Wan2.2 TTM sampling loop, using the same
FlowMatch Euler integrator as the official pipeline.

This node runs a two‑phase MoE sampler (HIGH then LOW model) and applies
TTM injection in the HIGH phase only, matching the dual‑clock logic.
"""

import torch
import comfy.model_management as mm
import comfy.sample

from .flow_scheduler import generate_wan_sigmas, sample_flowmatch_ttm, add_noise_at_sigma


class WanTTM_Sampler_Exact:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "switch_step": ("INT", {"default": 6, "min": 1, "max": 100}),
                "ttm_start_step": ("INT", {"default": 0, "min": 0, "max": 100}),
                "ttm_end_step": ("INT", {"default": 6, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "ttm_reference": ("LATENT",),
                "ttm_mask": ("IMAGE",),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "execute"
    CATEGORY = "TTM"

    def execute(
        self,
        model_high,
        model_low,
        positive,
        negative,
        latent,
        steps,
        cfg,
        switch_step,
        ttm_start_step,
        ttm_end_step,
        seed,
        ttm_reference=None,
        ttm_mask=None,
        debug=False,
    ):
        device = mm.get_torch_device()
        debug = bool(debug)

        # Base latent
        latent_image = latent["samples"].to(device)
        out = latent.copy()

        # TTM configuration
        tweak_index = int(ttm_start_step)
        tstrong_index = int(ttm_end_step)

        if tstrong_index > steps:
            if debug:
                print(f"[TTM Exact] WARNING: ttm_end_step {tstrong_index} > steps {steps}, clamping")
            tstrong_index = steps
        if tstrong_index > switch_step:
            if debug:
                print(f"[TTM Exact] WARNING: ttm_end_step {tstrong_index} > switch_step {switch_step}, clamping")
            tstrong_index = switch_step
        if tweak_index < 0:
            tweak_index = 0
        if tweak_index >= steps:
            if debug:
                print(f"[TTM Exact] WARNING: ttm_start_step {tweak_index} >= steps, disabling TTM")
            tweak_index = steps
        if tweak_index >= tstrong_index:
            if debug:
                print(f"[TTM Exact] WARNING: ttm_start_step >= ttm_end_step, disabling TTM")
            tweak_index = tstrong_index

        ttm_enabled = (
            ttm_reference is not None
            and ttm_mask is not None
            and tstrong_index > tweak_index
        )

        # Sigma schedule from FlowMatch sampler (same as Wan2.2)
        sampling_high = model_high.get_model_object("model_sampling")
        sigmas = generate_wan_sigmas(sampling_high, steps).to(
            device=device, dtype=latent_image.dtype
        )
        if debug:
            print(
                f"[TTM Exact] Sigma schedule ({len(sigmas)} values): "
                f"first={sigmas[0]:.4f}, last={sigmas[-1]:.4f}"
            )

        # Prepare TTM reference & mask
        ref = None
        motion_mask = None
        background_mask = None
        fixed_noise = None
        sampling_high_obj = None

        if ttm_enabled:
            if debug:
                print("[TTM Exact] TTM ENABLED")
                print(f"[TTM Exact]   start_step={tweak_index}, end_step={tstrong_index}")
                print(f"[TTM Exact]   switch_step={switch_step}")

            ref = ttm_reference["samples"].to(device)
            if ref.dim() == 4:
                ref = ref.unsqueeze(0)

            # Ensure reference channels match model latent format
            ref = comfy.sample.fix_empty_latent_channels(model_high, ref)
            latent_format = model_high.get_model_object("latent_format")
            if latent_format is not None:
                ref = latent_format.process_in(ref)

            # Mask: IMAGE -> [B,C,T,H,W] aligned with ref
            mask_img = ttm_mask.to(device)
            if mask_img.dim() == 5:
                # [B,T,H,W,C] -> [B,C,T,H,W]
                mask = mask_img.permute(0, 4, 1, 2, 3)
            elif mask_img.dim() == 4:
                # [T,H,W,C] -> [1,C,T,H,W]
                mask = mask_img.permute(3, 0, 1, 2).unsqueeze(0)
            elif mask_img.dim() == 3:
                # [T,H,W] -> [1,1,T,H,W]
                mask = mask_img.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError(f"[TTM Exact] Unsupported mask shape: {mask_img.shape}")

            # Resize temporal + spatial dims to match ref
            if mask.shape[2:] != ref.shape[2:]:
                import torch.nn.functional as F

                mask = F.interpolate(
                    mask,
                    size=ref.shape[2:],
                    mode="nearest",
                )

            # Match channel count
            if mask.shape[1] != ref.shape[1]:
                if mask.shape[1] > 1:
                    mask = mask.mean(dim=1, keepdim=True)
                mask = mask.expand(-1, ref.shape[1], -1, -1, -1)

            motion_mask = (mask > 0.5).to(ref.dtype)
            background_mask = 1.0 - motion_mask

            mask_cov = motion_mask.mean().item()
            if debug:
                print(
                    f"[TTM Exact] Reference latent: {ref.shape}, "
                    f"range=[{ref.min():.3f}, {ref.max():.3f}]"
                )
                print(
                    f"[TTM Exact] Motion mask: {motion_mask.shape}, "
                    f"coverage={mask_cov:.3f}"
                )

            # Fixed noise for all TTM noisy-ref injections
            torch.manual_seed(int(seed) & 0x7FFFFFFF)
            fixed_noise = torch.randn_like(ref)

            # Initialize latent with noisy reference at sigma[tweak_index]
            sigma_init = sigmas[tweak_index].item()
            sampling_high_obj = model_high.get_model_object("model_sampling")
            if sampling_high_obj is not None and hasattr(sampling_high_obj, "noise_scaling"):
                sigma_tensor = torch.tensor(
                    sigma_init, device=ref.device, dtype=ref.dtype
                )
                latent_image = sampling_high_obj.noise_scaling(
                    sigma_tensor,
                    fixed_noise,
                    ref,
                    True,
                )
            else:
                latent_image = add_noise_at_sigma(ref, fixed_noise, sigma_init)
                if debug:
                    print(
                        "[TTM Exact] WARNING: model_sampling.noise_scaling not found, "
                        "falling back to add_noise_at_sigma"
                    )

            if debug:
                print(
                    f"[TTM Exact] Init latent from ref at sigma={sigma_init:.4f}, "
                    f"range=[{latent_image.min():.3f}, {latent_image.max():.3f}]"
                )
        else:
            if debug:
                print("[TTM Exact] TTM DISABLED, standard MoE sampling")

        # Sampling noise (Comfy-style, deterministic w.r.t seed)
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds).to(device)
        noise_mask = latent.get("noise_mask")

        # MoE configuration
        switch_step_clamped = min(max(int(switch_step), 1), steps)
        has_phase_2 = switch_step_clamped < steps

        if ttm_enabled and tstrong_index > switch_step_clamped:
            if debug:
                print(
                    f"[TTM Exact] WARNING: ttm_end_step {tstrong_index} > switch_step_clamped "
                    f"{switch_step_clamped}, clamping"
                )
            tstrong_index = switch_step_clamped

        if debug:
            print(f"[TTM Exact] MoE config: steps={steps}, switch_step={switch_step_clamped}")
            if has_phase_2:
                print("[TTM Exact] Phase 1 = HIGH, Phase 2 = LOW")
            else:
                print("[TTM Exact] Single phase = HIGH only")

        # Global step counter across both phases
        global_step_counter = [0]

        def ttm_callback(data):
            """TTM injection callback (FlowMatch Euler domain)."""
            x = data["x"]
            sigma_next = data["sigma_next"]

            global_step = global_step_counter[0]

            # Apply TTM only in [tweak_index, tstrong_index)
            if (
                ttm_enabled
                and sampling_high_obj is not None
                and tweak_index <= global_step < tstrong_index
            ):
                sigma_val = (
                    float(sigma_next) if not torch.is_tensor(sigma_next) else sigma_next.item()
                )

                if sigma_val > 0 and hasattr(sampling_high_obj, "noise_scaling"):
                    sigma_tensor = torch.tensor(
                        sigma_val, device=ref.device, dtype=ref.dtype
                    )
                    noisy_ref = sampling_high_obj.noise_scaling(
                        sigma_tensor,
                        fixed_noise.to(device=ref.device, dtype=ref.dtype),
                        ref,
                        True,
                    )
                elif sigma_val > 0:
                    noisy_ref = add_noise_at_sigma(ref, fixed_noise, sigma_val)
                else:
                    noisy_ref = ref

                noisy_ref = noisy_ref.to(device=x.device, dtype=x.dtype)
                bg_mask = background_mask.to(device=x.device, dtype=x.dtype)
                mot_mask = motion_mask.to(device=x.device, dtype=x.dtype)

                if debug and global_step == tweak_index:
                    print(
                        f"[TTM Exact] TTM injection start at step {global_step}, "
                        f"sigma_next={sigma_val:.4f}"
                    )

                x.mul_(bg_mask).add_(noisy_ref * mot_mask)

                if debug and global_step == tstrong_index - 1:
                    print(f"[TTM Exact] TTM injection end at step {global_step}")

            global_step_counter[0] += 1

        def run_flow_phase(model, phase_sigmas, disable_noise, latent_in):
            return sample_flowmatch_ttm(
                model=model,
                noise=noise,
                cfg=cfg,
                sigmas=phase_sigmas,
                positive=positive,
                negative=negative,
                latent_image=latent_in,
                disable_noise=disable_noise,
                force_zero_end=True,
                noise_mask=noise_mask,
                callback=ttm_callback if ttm_enabled else None,
                disable_pbar=False,
                seed=seed,
            )

        # Phase 1: HIGH model
        if ttm_enabled and tweak_index > 0:
            sigmas_p1 = sigmas[tweak_index : switch_step_clamped + 1].clone()
            global_step_counter[0] = tweak_index
        else:
            sigmas_p1 = sigmas[: switch_step_clamped + 1].clone()
            global_step_counter[0] = 0

        if not has_phase_2:
            sigmas_p1 = sigmas[tweak_index:].clone() if ttm_enabled and tweak_index > 0 else sigmas.clone()

        if debug:
            print(
                f"[TTM Exact] Phase 1 (HIGH): {len(sigmas_p1) - 1} steps, "
                f"sigmas [{sigmas_p1[0]:.4f} -> {sigmas_p1[-1]:.4f}]"
            )

        # If TTM or we have a second phase, we treat latent_image as already at sigma[0]
        disable_noise_p1 = ttm_enabled or has_phase_2
        latent_image = run_flow_phase(
            model_high,
            sigmas_p1,
            disable_noise=disable_noise_p1,
            latent_in=latent_image,
        )

        # Phase 2: LOW model, no TTM
        if has_phase_2:
            sigmas_p2 = sigmas[switch_step_clamped:].clone()
            if debug:
                print(
                    f"[TTM Exact] Phase 2 (LOW): {len(sigmas_p2) - 1} steps, "
                    f"sigmas [{sigmas_p2[0]:.4f} -> {sigmas_p2[-1]:.4f}]"
                )

            latent_image = run_flow_phase(
                model_low,
                sigmas_p2,
                disable_noise=True,
                latent_in=latent_image,
            )

        out["samples"] = latent_image
        if debug:
            print(
                f"[TTM Exact] Done. Output shape={latent_image.shape}, "
                f"range=[{latent_image.min():.3f}, {latent_image.max():.3f}]"
            )

        return (out,)

# Register
NODE_CLASS_MAPPINGS = {
    "WanTTM_Sampler_Exact": WanTTM_Sampler_Exact,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanTTM_Sampler_Exact": "Wan2.2 TTM Sampler (Exact)",
}

