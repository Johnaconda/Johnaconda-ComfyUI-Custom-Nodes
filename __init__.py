# Auto-generated init for custom nodes

from .comfyui_model_merger_pro import ModelMergerPro
from .comfyui_model_merger_pro import FromStateDictPro
from .comfyui_model_merger_pro import SaveMergedPro
from .comfyui_model_merger_pro import ApplyLoRAPro
from .prompt_combiner_node import PromptCombiner
from .ultra_cfg import UltraCFG
from .ultra_duper_sampler_fluxaware import UltraDuperSampler, UltraSamplerMatrix, UltraBenchmarkTester, UltraBenchmarkSave
from .ultra_hires_script import UltraHiResScript
from .ultra_noise_sweeps import UltraNoiseSweeps


NODE_CLASS_MAPPINGS = {
    'ModelMergerPro': ModelMergerPro,
    "FromStateDictPro": FromStateDictPro,
    "SaveMergedPro": SaveMergedPro,
    "ApplyLoRAPro": ApplyLoRAPro,
    'PromptCombiner': PromptCombiner,
    'UltraCFG': UltraCFG,
    'UltraDuperSampler': UltraDuperSampler,
    'UltraSamplerMatrix': UltraSamplerMatrix,
    'UltraBenchmarkTester': UltraBenchmarkTester,
    "UltraBenchmarkSave": UltraBenchmarkSave,
    'UltraHiResScript': UltraHiResScript,
    'UltraNoiseSweeps': UltraNoiseSweeps,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ModelMergerPro': 'Model Merger Pro',
    'FromStateDictPro': "Instantiate from STATE_DICT (Pro)",
    'SaveMergedPro': "Save Merged (Pro, metadata)",
    'ApplyLoRAPro': "Apply LoRA to STATE_DICT",
    'PromptCombiner': 'Prompt Combiner',
    'UltraCFG': 'Ultra C F G',
    'UltraDuperSampler': 'Ultra Duper Sampler',
    'UltraSamplerMatrix': "Ultra Sampler Matrix",
    'UltraBenchmarkTester': "Ultra Benchmark Tester",
    "UltraBenchmarkSave": "Ultra Benchmark Save",
    'UltraHiResScript': 'Ultra Hi Res Script',
    'UltraNoiseSweeps': 'Ultra Noise Sweeps',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
