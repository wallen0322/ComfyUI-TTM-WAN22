"""
TTM (Time-to-Move) Nodes for Wan2.2 in ComfyUI
Dual-clock denoising for motion-controlled video generation
https://github.com/time-to-move/TTM
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .ttm_conditioning import WanTTMConditioning
from .ttm_sampler import WanTTMSampler
from .ttm_sampler_complete import WanTTMSamplerComplete


class TTMExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WanTTMConditioning,
            WanTTMSamplerComplete,  # Complete implementation with MoE + TTM
            WanTTMSampler,  # Legacy/simple version
        ]


async def comfy_entrypoint() -> TTMExtension:
    return TTMExtension()
