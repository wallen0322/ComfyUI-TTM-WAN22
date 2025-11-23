"""Extension entrypoint for the new Wan2.2 TTM node set.

This file is kept minimal on purpose. It only exposes the new nodes implemented
in ``nodes_ttm_new.py`` via the async ``comfy_entrypoint`` expected by
``__init__.py``.
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .nodes_ttm_new import (
    WanTTM_ModelFromUNet,
    WanTTM_Conditioning,
    WanTTM_Sampler,
    WanTTM_Decode,
)


class TTMExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WanTTM_ModelFromUNet,
            WanTTM_Conditioning,
            WanTTM_Sampler,
            WanTTM_Decode,
        ]


async def comfy_entrypoint() -> TTMExtension:
    return TTMExtension()
