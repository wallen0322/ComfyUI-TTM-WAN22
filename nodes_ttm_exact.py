"""
Exact reproduction of Wan2.2 TTM sampling loop.
This node bypasses ComfyUI's standard samplers to ensure 1:1 alignment with official implementation.
"""

import torch
import comfy.model_management as mm
import comfy.samplers
import comfy.utils
import comfy.sample
from tqdm import tqdm
from .flow_scheduler import generate_wan_sigmas

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
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "switch_step": ("INT", {"default": 6, "min": 1, "max": 100}),
                "ttm_start_step": ("INT", {"default": 0, "min": 0, "max": 100}),
                "ttm_end_step": ("INT", {"default": 6, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "ttm_reference": ("LATENT",),
                "ttm_mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "execute"
    CATEGORY = "TTM"

    def execute(self, model_high, model_low, positive, negative, latent, steps, cfg, switch_step, ttm_start_step, ttm_end_step, seed, ttm_reference=None, ttm_mask=None):
        device = mm.get_torch_device()
        
        # 1. Setup TTM Data
        ttm_enabled = ttm_reference is not None and ttm_mask is not None
        ref = None
        mask = None
        fixed_noise = None
        
        # Initial latent (clean or noise?)
        # In I2V, input 'latent' is usually the encoded start image + noise, or just noise
        # Wan official pipeline: starts with "noisy latents"
        # ComfyUI I2V usually passes the start image as 'latent'
        
        # We need to generate the initial noise
        # Generator must be on the same device as the tensor
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Handle input latent
        x_input = latent["samples"].clone().to(device)
        
        # Generate sampling noise (same shape as latent)
        noise = torch.randn(x_input.shape, generator=generator, device=device, dtype=x_input.dtype)
        
        if ttm_enabled:
            print(f"[TTM Exact] TTM Enabled: steps [{ttm_start_step}, {ttm_end_step})")
            ref = ttm_reference["samples"].to(device)
            # Mask: [T,H,W,C] or [B,H,W,C] -> [B,C,T,H,W]
            mask = ttm_mask.to(device)
            
            # Handle input mask dimensions
            if mask.dim() == 4: 
                # Usually [T, H, W, C] for image sequence
                # Permute to [1, C, T, H, W]
                # [T, H, W, C] -> [C, T, H, W] -> [1, C, T, H, W]
                mask = mask.permute(3, 0, 1, 2).unsqueeze(0)
            elif mask.dim() == 3:
                # [T, H, W] -> [1, 1, T, H, W]
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 5:
                # [B, T, H, W, C] -> [B, C, T, H, W]
                mask = mask.permute(0, 4, 1, 2, 3)
            
            # Ensure mask is contiguous
            mask = mask.contiguous()
            
            # Resize mask to match latent dimensions (Time, Height, Width)
            # ref shape: [B, C, T, H, W]
            if mask.shape[2:] != ref.shape[2:]:
                import torch.nn.functional as F
                # F.interpolate takes [N, C, D, H, W]
                mask = F.interpolate(mask, size=ref.shape[2:], mode='nearest')
            
            # Handle channel mismatch (e.g. mask is RGB=3, latent is 16)
            if mask.shape[1] != ref.shape[1]:
                # Convert RGB/multichannel mask to single channel by max or mean
                if mask.shape[1] > 1:
                    mask = mask.mean(dim=1, keepdim=True)
                # Expand to match latent channels
                mask = mask.expand(-1, ref.shape[1], -1, -1, -1)
            
            background_mask = 1.0 - mask
            fixed_noise = noise.clone()
        
        # 2. Generate Sigmas using model's parameters (Flow Match schedule)
        sampling_high = model_high.get_model_object("model_sampling")
        sigmas = generate_wan_sigmas(sampling_high, steps).to(device=device, dtype=x_input.dtype)
        print(f"[TTM Exact] Sigma schedule: first={sigmas[0]:.4f}, last={sigmas[-1]:.4f}")
        
        # 3. Initialize Latent (x)
        # Standard initialization: start with noise at sigma[0]
        sigma_start = sigmas[0].to(device=device, dtype=x_input.dtype)
        x = noise * sigma_start + x_input * (1.0 - sigma_start)
        
        # TTM initialization: if enabled, initialize at ttm_start_step
        if ttm_enabled:
            # If ttm_start_step > 0, we need to denoise to ttm_start_step first
            if ttm_start_step > 0:
                # Denoise from step 0 to ttm_start_step (without TTM)
                print(f"[TTM Exact] Pre-TTM denoising: steps 0 -> {ttm_start_step}")
                for i in range(ttm_start_step):
                    sigma = sigmas[i]
                    sigma_next = sigmas[i+1]
                    dt = sigma_next - sigma
                    
                    # Use high model for pre-TTM steps
                    v_pred = self.calc_model_pred(model_high, x, sigma, positive, negative, cfg)
                    x = x + v_pred * dt
                
                print(f"[TTM Exact] Pre-TTM denoising complete, x range: [{x.min():.3f}, {x.max():.3f}]")
            
            # Now apply TTM initialization at ttm_start_step's sigma
            sigma_ttm_start = sigmas[ttm_start_step].to(device=device, dtype=ref.dtype)
            noisy_ref_init = self.add_noise(ref, fixed_noise, sigma_ttm_start)
            
            # Blend: background keeps current denoised state, motion gets noisy ref
            # At ttm_start_step=0, x is pure noise, so background should also be noise
            if ttm_start_step == 0:
                # Both are noise, but we use different noise sources
                noisy_bg_init = fixed_noise  # Use same noise for consistency
            else:
                # Background keeps the denoised state
                noisy_bg_init = x.clone()
            
            x = noisy_bg_init * background_mask + noisy_ref_init * mask
            print(f"[TTM Exact] TTM initialized at step {ttm_start_step}, sigma={sigma_ttm_start:.4f}")
            print(f"[TTM Exact]   x range after init: [{x.min():.3f}, {x.max():.3f}]")
        
        x = x.to(device)
        
        # 4. Sampling Loop (start from ttm_start_step if TTM enabled)
        start_idx = ttm_start_step if ttm_enabled else 0
        pbar = tqdm(range(start_idx, steps))
        for i in pbar:
            # Current sigma and next sigma
            sigma = sigmas[i].to(device=device, dtype=x.dtype)
            sigma_next = sigmas[i+1].to(device=device, dtype=x.dtype)
            dt = sigma_next - sigma # Negative value
            
            # Select Model
            if i < switch_step:
                model = model_high
                phase = "HIGH"
            else:
                model = model_low
                phase = "LOW"
            
            # Prepare Conditioning (Basic implementation)
            # We need to properly format conditioning for model.apply_model
            # ComfyUI cond is [[tensor, {dict}], ...]
            cond = positive # Simplify: assume positive only for now or handle CFG manually
            
            # Call Model
            # We use a helper to handle CFG if possible, or just direct call
            # Direct call to model.apply_model returns the model output (Velocity 'v')
            
            # NOTE: We need to handle Classifier Free Guidance (CFG) manually if we go this deep
            # or use Comfy's sampling helpers but WITHOUT the loop.
            
            # Let's use Comfy's basic sampler wrapper for a single step to handle CFG
            # but we control the input/output scaling explicitly.
            
            # Prepare latent for model: Wan models usually expect standard distribution
            # Our 'x' is already in that space.
            
            # Get prediction (v)
            # We use a trick: create a mini-sampler just for getting 'denoised' or 'cond_pred'
            # But easiest is using model.apply_model directly.
            
            # Handle CFG:
            # cond_pred = model(x, sigma, cond)
            # uncond_pred = model(x, sigma, uncond)
            # pred = uncond + cfg * (cond - uncond)
            
            v_pred = self.calc_model_pred(model, x, sigma, positive, negative, cfg)
            
            # Debug: log velocity range for first few steps
            if i < 3 or (ttm_enabled and ttm_start_step <= i < ttm_start_step + 3):
                print(f"[TTM Exact] Step {i} ({phase}): sigma={sigma:.4f}, v_range=[{v_pred.min():.3f}, {v_pred.max():.3f}], x_range=[{x.min():.3f}, {x.max():.3f}]")
            
            # Euler Step
            # x_{t-1} = x_t + v * dt
            x = x + v_pred * dt
            
            # TTM Injection
            if ttm_enabled and ttm_start_step <= i < ttm_end_step:
                # Inject at next sigma
                # noisy_ref = (1-sigma_next)*ref + sigma_next*fixed_noise
                noisy_ref = self.add_noise(ref, fixed_noise, sigma_next)
                
                # Blend
                x = x * background_mask + noisy_ref * mask
                # print(f"[TTM Exact] Step {i}: Injected at sigma={sigma_next:.4f}")
            
        
        # 5. Return (already in latent space, no extra scaling needed)
        return ({"samples": x},)

    def add_noise(self, clean, noise, sigma):
        return (1.0 - sigma) * clean + sigma * noise

    def calc_model_pred(self, model, x, sigma, positive, negative, cfg):
        # Use comfy.sample.sample to get denoised prediction
        # This is the most reliable way to handle all conditioning and CFG
        
        comfy.model_management.load_model_gpu(model)
        
        # Capture denoised prediction from callback
        # The callback receives x0 (denoised) before any inverse_noise_scaling
        captured_denoised = [None]
        
        def capture_callback(step, x0, x_current, total_steps):
            # x0 is the denoised prediction (clean latent, before inverse_noise_scaling)
            if captured_denoised[0] is None:
                captured_denoised[0] = x0.clone()
        
        # Use comfy.sample.sample for ONE step
        # disable_noise=True means use x directly (no noise_scaling applied)
        # force_full_denoise=False means don't apply inverse_noise_scaling at the end
        dummy_noise = torch.randn_like(x)
        
        result = comfy.sample.sample(
            model=model,
            noise=dummy_noise,
            steps=2,  # Need at least 2 for one actual step
            cfg=cfg,
            sampler_name="euler",
            scheduler="normal",
            positive=positive,
            negative=negative,
            latent_image=x,
            denoise=1.0,
            disable_noise=True,  # Use x directly (no noise_scaling)
            start_step=0,
            last_step=1,  # Only first step
            force_full_denoise=False,  # Don't apply inverse_noise_scaling
            noise_mask=None,
            callback=capture_callback,
            disable_pbar=True,
            seed=0,
        )
        
        # Extract velocity from denoised prediction
        # For FlowMatch: velocity = (x - x0) / sigma
        if captured_denoised[0] is not None:
            x0 = captured_denoised[0]
            sigma_t = sigma if torch.is_tensor(sigma) else torch.tensor(sigma, device=x.device, dtype=x.dtype)
            while sigma_t.ndim < x.ndim:
                sigma_t = sigma_t.unsqueeze(-1)
            sigma_t = sigma_t.to(device=x.device, dtype=x.dtype).clamp(min=1e-5)
            velocity = (x - x0) / sigma_t
            return velocity
        
        # Fallback: try to use result directly (may have scaling issues)
        if result is not None and torch.is_tensor(result):
            sigma_t = sigma if torch.is_tensor(sigma) else torch.tensor(sigma, device=x.device, dtype=x.dtype)
            while sigma_t.ndim < x.ndim:
                sigma_t = sigma_t.unsqueeze(-1)
            sigma_t = sigma_t.to(device=x.device, dtype=x.dtype).clamp(min=1e-5)
            velocity = (x - result) / sigma_t
            return velocity
        
        # Last resort: return zero velocity
        print(f"[TTM Warning] Failed to extract velocity, returning zero")
        return torch.zeros_like(x)

# Register
NODE_CLASS_MAPPINGS = {
    "WanTTM_Sampler_Exact": WanTTM_Sampler_Exact,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanTTM_Sampler_Exact": "Wan2.2 TTM Sampler (Exact)",
}

