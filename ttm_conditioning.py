"""TTM Conditioning Node - Prepares motion latents and masks for dual-clock denoising."""

import torch
import torch.nn.functional as F
from comfy_api.latest import io
import comfy.utils


class WanTTMConditioning(io.ComfyNode):
    """Prepares TTM conditioning with motion latents and masks."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanTTMConditioning",
            category="conditioning",
            description="Prepare TTM motion conditioning for dual-clock denoising",
            inputs=[
                # Core inputs
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                
                # Media inputs
                io.Image.Input("start_image", tooltip="Starting frame"),
                io.Image.Input("motion_signal_video", tooltip="Motion reference video (IMAGE batch)"),
                io.Image.Input("motion_signal_mask", tooltip="Motion mask (IMAGE batch)"),
                
                # Video dimensions
                io.Int.Input("width", default=832, min=64, max=2048, step=8),
                io.Int.Input("height", default=480, min=64, max=2048, step=8),
                io.Int.Input("num_frames", default=81, min=5, max=257, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=16, step=1,
                           tooltip="Batch size (usually 1)"),
                
                # TTM dual-clock control (slider-friendly ranges)
                io.Int.Input("tweak_index", default=3, min=0, max=15, step=1,
                           tooltip="Background denoising start (0-15, typically 2-5)"),
                io.Int.Input("tstrong_index", default=7, min=0, max=20, step=1,
                           tooltip="Motion control start (must be >= tweak_index)"),
            ],
            outputs=[
                io.Conditioning.Output("positive_out", display_name="positive"),
                io.Conditioning.Output("negative_out", display_name="negative"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        start_image,
        motion_signal_video,
        motion_signal_mask,
        width: int,
        height: int,
        num_frames: int,
        batch_size: int,
        tweak_index: int,
        tstrong_index: int,
    ) -> io.NodeOutput:
        
        if tweak_index > tstrong_index:
            raise ValueError("tweak_index must be <= tstrong_index")
        
        device = comfy.model_management.get_torch_device()
        
        # Calculate latent dimensions
        latent_width = width // 8
        latent_height = height // 8
        latent_frames = (num_frames - 1) // 4 + 1  # Temporal compression factor
        
        # Initialize empty latent
        latent = torch.zeros(
            [batch_size, 16, latent_frames, latent_height, latent_width],
            device=device
        )
        
        # Encode start image
        start_image_encoded = vae.encode(start_image[:, :, :, :3])
        
        # Handle VAE output (dict or tensor)
        if isinstance(start_image_encoded, dict):
            start_image_latent = start_image_encoded["samples"]
        else:
            start_image_latent = start_image_encoded
        
        # Ensure 5D: [B, C, T, H, W]
        if start_image_latent.ndim == 4:
            start_image_latent = start_image_latent.unsqueeze(2)  # Add temporal dim
        elif start_image_latent.ndim != 5:
            raise ValueError(f"Expected 4D or 5D start_image_latent, got {start_image_latent.ndim}D")
        
        if start_image_latent.shape[2] != 1:
            raise ValueError(f"Start image should have temporal dim=1, got {start_image_latent.shape[2]}")
        
        # Expand to batch size and place in first frame
        start_image_latent = start_image_latent.expand(batch_size, -1, -1, -1, -1)
        latent[:, :, :start_image_latent.shape[2]] = start_image_latent.to(latent.device)
        
        # Encode motion signal video
        motion_signal_encoded = vae.encode(motion_signal_video[:, :, :, :3])
        
        if isinstance(motion_signal_encoded, dict):
            motion_signal_latent = motion_signal_encoded["samples"]
        else:
            motion_signal_latent = motion_signal_encoded
        
        # Convert to 5D: [B, C, T, H, W]
        if motion_signal_latent.ndim == 4:
            # Assume batch of images, reshape to video
            # motion_signal_latent: [T, C, H, W]
            motion_signal_latent = motion_signal_latent.unsqueeze(0).permute(0, 2, 1, 3, 4)
            # Now: [1, C, T, H, W]
        elif motion_signal_latent.ndim == 5:
            # Already in correct format
            pass
        else:
            raise ValueError(f"Expected 4D or 5D motion_signal_latent, got {motion_signal_latent.ndim}D")
        
        # Expand to batch size
        motion_signal_latent = motion_signal_latent.expand(batch_size, -1, -1, -1, -1)
        
        # Match latent length
        motion_latent_length = motion_signal_latent.shape[2]
        if motion_latent_length > latent_frames:
            motion_signal_latent = motion_signal_latent[:, :, :latent_frames]
        elif motion_latent_length < latent_frames:
            # Pad or loop
            padding_needed = latent_frames - motion_latent_length
            motion_signal_latent = torch.cat([
                motion_signal_latent,
                motion_signal_latent[:, :, :padding_needed]
            ], dim=2)
        
        # Process mask
        # motion_signal_mask: [T, H, W, C]
        mask_latent = motion_signal_mask.view(
            motion_signal_mask.shape[0], motion_signal_mask.shape[1], motion_signal_mask.shape[2], -1
        )
        # Permute to [T, C, H, W]
        mask_latent = mask_latent.permute(0, 3, 1, 2)
        # Take mean over channels
        mask_latent = mask_latent.mean(dim=1)  # [T, H, W]
        # Add batch and channel dims
        mask_latent = mask_latent.unsqueeze(0).unsqueeze(1)  # [1, 1, T, H, W]
        
        # Downsample spatially to match latent size
        mask_latent = F.interpolate(
            mask_latent.view(-1, 1, mask_latent.shape[-2], mask_latent.shape[-1]),  # [T, 1, H, W]
            size=(latent_height, latent_width),
            mode='bilinear',
            align_corners=False
        ).view(1, 1, mask_latent.shape[2], latent_height, latent_width)
        
        # Temporal downsample to match latent frames
        if mask_latent.shape[2] > latent_frames:
            # Simple subsampling (every 4th frame)
            indices = torch.linspace(0, mask_latent.shape[2] - 1, latent_frames).long()
            mask_latent = mask_latent[:, :, indices]
        elif mask_latent.shape[2] < latent_frames:
            # Pad
            padding_needed = latent_frames - mask_latent.shape[2]
            mask_latent = torch.cat([
                mask_latent,
                mask_latent[:, :, :padding_needed]
            ], dim=2)
        
        # Remove channel dim: [1, T, H, W]
        mask_latent = mask_latent.squeeze(1)
        # Expand to batch size: [B, T, H, W]
        mask_latent = mask_latent.expand(batch_size, -1, -1, -1)
        
        # Prepare TTM conditioning dict
        ttm_conditioning = {
            "concat_latent_image": motion_signal_latent,
            "motion_signal_mask": mask_latent,
            "ttm_tweak_index": tweak_index,
            "ttm_tstrong_index": tstrong_index,
        }
        
        # Append to conditioning
        positive = comfy.utils.conditioning_set_values(positive, ttm_conditioning)
        negative = comfy.utils.conditioning_set_values(negative, ttm_conditioning)
        
        return io.NodeOutput(
            positive,
            negative,
            {"samples": latent}
        )
