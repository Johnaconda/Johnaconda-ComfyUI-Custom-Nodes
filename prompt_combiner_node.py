class PromptCombiner:
    """
    A ComfyUI node that takes two string inputs (base_prompt and subject_prompt),
    combines them into a single string, and outputs it as a STRING.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_prompt":    ("STRING", {"multiline": True, "default": "masterpiece, 3d, soft shadows, best quality"}),
            "subject_prompt": ("STRING", {"multiline": True, "default": "a sunset at the beach with a couple sitting under a palm tree"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "combine"
    CATEGORY = "Prompt Tools"

    def combine(self, base_prompt, subject_prompt):
        # Merge the two text fields into one prompt string
        combined = f"{base_prompt}, {subject_prompt}"
        return (combined,)

# Node mappings
NODE_CLASS_MAPPINGS = {"PromptCombiner": PromptCombiner}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptCombiner": "Prompt Combiner"}
