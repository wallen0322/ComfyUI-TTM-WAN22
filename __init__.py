"""ComfyUI-TTM: Time-to-Move nodes for Wan2.2.

This package registers the new Wan2.2 TTM node set using the classic
``NODE_CLASS_MAPPINGS`` / ``NODE_DISPLAY_NAME_MAPPINGS`` mechanism so it
works on standard ComfyUI without relying on the comfy_api extension
entrypoint.
"""

from .nodes_ttm_new import (
    WanTTM_ModelFromUNet,
    WanTTM_Conditioning,
    WanTTM_Sampler,
    WanTTM_Decode,
)


NODE_CLASS_MAPPINGS = {
    "WanTTM_ModelFromUNet": WanTTM_ModelFromUNet,
    "WanTTM_Conditioning_New": WanTTM_Conditioning,
    "WanTTM_Sampler_New": WanTTM_Sampler,
    "WanTTM_Decode_New": WanTTM_Decode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanTTM_ModelFromUNet": "Wan2.2 TTM: Model From UNet",
    "WanTTM_Conditioning_New": "Wan2.2 TTM: Conditioning",
    "WanTTM_Sampler_New": "Wan2.2 TTM: Sampler",
    "WanTTM_Decode_New": "Wan2.2 TTM: Decode",
}
