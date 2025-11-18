"""
TTM (Time-to-Move) Node for Wan2.2 in ComfyUI
Dual-clock denoising for motion-controlled video generation
https://github.com/time-to-move/TTM
"""

import torch
import nodes
import node_helpers
import comfy.model_management
import comfy.utils
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class WanTTMImageToVideo(io.ComfyNode):
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanTTMImageToVideo",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image"),
                io.Image.Input("motion_signal_video"),
                io.Mask.Input("motion_signal_mask"),
                io.Int.Input("tweak_index", default=3, min=0, max=50),
                io.Int.Input("tstrong_index", default=7, min=0, max=50),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, 
                start_image, motion_signal_video, motion_signal_mask,
                tweak_index, tstrong_index, clip_vision_output=None) -> io.NodeOutput:
        
        if tweak_index > tstrong_index:
            raise ValueError(f"tweak_index ({tweak_index}) must be <= tstrong_index ({tstrong_index})")
        
        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
            device=comfy.model_management.intermediate_device()
        )
        
        start_image = comfy.utils.common_upscale(
            start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        start_image_latent = vae.encode(start_image[:, :, :, :3])
        
        motion_signal_video = comfy.utils.common_upscale(
            motion_signal_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        
        if motion_signal_video.shape[0] < length:
            padding = torch.ones(
                (length - motion_signal_video.shape[0], height, width, motion_signal_video.shape[-1]), 
                device=motion_signal_video.device, dtype=motion_signal_video.dtype
            ) * 0.5
            motion_signal_video = torch.cat([motion_signal_video, padding], dim=0)
        
        motion_signal_latent = vae.encode(motion_signal_video[:, :, :, :3])
        
        if motion_signal_mask.ndim == 2:
            motion_signal_mask = motion_signal_mask.unsqueeze(0)
        if motion_signal_mask.ndim == 3:
            motion_signal_mask = motion_signal_mask.unsqueeze(-1)
        
        motion_signal_mask = comfy.utils.common_upscale(
            motion_signal_mask[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        
        if motion_signal_mask.shape[0] < length:
            padding_mask = torch.ones(
                (length - motion_signal_mask.shape[0], height, width, 1),
                device=motion_signal_mask.device, dtype=motion_signal_mask.dtype
            )
            motion_signal_mask = torch.cat([motion_signal_mask, padding_mask], dim=0)
        
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        latent_length = ((length - 1) // 4) + 1
        
        mask_latent = motion_signal_mask.view(
            length, height_mask, vae_stride, width_mask, vae_stride, 1
        )
        mask_latent = mask_latent.permute(5, 0, 2, 4, 1, 3)
        mask_latent = mask_latent.reshape(1, length, vae_stride * vae_stride, height_mask, width_mask)
        mask_latent = mask_latent.mean(dim=2)
        
        mask_latent = torch.nn.functional.interpolate(
            mask_latent, size=(latent_length, height_mask, width_mask), mode='nearest'
        )
        
        concat_latent_image = torch.cat([start_image_latent, motion_signal_latent], dim=1)
        
        ttm_conditioning = {
            "concat_latent_image": concat_latent_image,
            "motion_signal_mask": mask_latent,
            "ttm_tweak_index": tweak_index,
            "ttm_tstrong_index": tstrong_index,
        }
        
        positive = node_helpers.conditioning_set_values(positive, ttm_conditioning)
        negative = node_helpers.conditioning_set_values(negative, ttm_conditioning)
        
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(
                positive, {"clip_vision_output": clip_vision_output}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"clip_vision_output": clip_vision_output}
            )
        
        out_latent = {"samples": latent}
        return io.NodeOutput(positive, negative, out_latent)


class TTMExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [WanTTMImageToVideo]


async def comfy_entrypoint() -> TTMExtension:
    return TTMExtension()
