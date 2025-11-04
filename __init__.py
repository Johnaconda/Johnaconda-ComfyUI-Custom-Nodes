# __init__.py

"""Auto-load all custom node classes in this module for ComfyUI."""

# Import individual node definitions
from .comfyui_model_merger_pro import ComfyUIModelMergerPro
from .prompt_combiner_node import PromptCombinerNode
from .sdxl_dual_merger import SDXLDualMerger
from .ultra_cfg import UltraCFG
from .ultra_duper_sampler_fluxaware import UltraDuperSamplerFluxaware
from .ultra_hires_script import UltraHiResScript
from .ultra_noise_sweeps import UltraNoiseSweeps

# Map internal names to classes for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ComfyUIModelMergerPro": ComfyUIModelMergerPro,
    "PromptCombinerNode": PromptCombinerNode,
    "SDXLDualMerger": SDXLDualMerger,
    "UltraCFG": UltraCFG,
    "UltraDuperSamplerFluxaware": UltraDuperSamplerFluxaware,
    "UltraHiResScript": UltraHiResScript,
    "UltraNoiseSweeps": UltraNoiseSweeps,
}

# Optionally provide human-readable names
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUIModelMergerPro": "Model Merger Pro",
    "PromptCombinerNode": "Prompt Combiner",
    "SDXLDualMerger": "SDXL Dual Merger",
    "UltraCFG": "Ultra CFG Injector",
    "UltraDuperSamplerFluxaware": "Ultra Duper Sampler (Flux-Aware)",
    "UltraHiResScript": "Ultra HiRes Script Node",
    "UltraNoiseSweeps": "Ultra Noise Sweeps",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
