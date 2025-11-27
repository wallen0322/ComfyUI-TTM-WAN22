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

from .nodes_ttm_clean import (
    WanTTM_PrepareLatents,
    WanTTM_Sampler_Clean,
)

from .nodes_ttm_exact import (
    WanTTM_Sampler_Exact,
)


NODE_CLASS_MAPPINGS = {
    # Original nodes
    "WanTTM_ModelFromUNet": WanTTM_ModelFromUNet,
    "WanTTM_Conditioning_New": WanTTM_Conditioning,
    "WanTTM_Sampler_New": WanTTM_Sampler,
    "WanTTM_Decode_New": WanTTM_Decode,
    # Clean TTM nodes (extracted from KJ)
    "WanTTM_PrepareLatents": WanTTM_PrepareLatents,
    "WanTTM_Sampler_Clean": WanTTM_Sampler_Clean,
    # Exact TTM node
    "WanTTM_Sampler_Exact": WanTTM_Sampler_Exact,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanTTM_ModelFromUNet": "Wan2.2 TTM: Model From UNet",
    "WanTTM_Conditioning_New": "Wan2.2 TTM: Conditioning",
    "WanTTM_Sampler_New": "Wan2.2 TTM: Sampler",
    "WanTTM_Decode_New": "Wan2.2 TTM: Decode",
    # Clean TTM nodes
    "WanTTM_PrepareLatents": "TTM: Prepare Latents (Clean)",
    "WanTTM_Sampler_Clean": "TTM: Sampler (Clean)",
    # Exact TTM node
    "WanTTM_Sampler_Exact": "TTM: Sampler (Exact Reproduction)",
}
