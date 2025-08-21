# UltraCFG v2.1 — CFG-Focus (in-loop) for a single-run sampler
# Purpose: keep a constant base CFG but allow a per-step "focus" nudge toward denoised x0 without reruns.
# This approximates "in-run CFG shaping" and can enhance detail or stability depending on schedule.

class UltraCFG:
    MODES = ["none", "linear", "cosine", "exp", "plateau", "late_bloom", "early_boost"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (cls.MODES, {"tooltip": "Focus schedule shape"}),
                "base_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.05}),
                "focus_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "0 disables; 0.05–0.25 subtle"}),
                "focus_start": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "focus_end": ("INT", {"default": 10000, "min": 1, "max": 10000, "step": 1, "tooltip": "Exclusive end step; clamped to total steps"}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("special_cfg",)
    FUNCTION = "build"
    CATEGORY = "sampling/ultra/tweaks"
    DESCRIPTION = "Base CFG + in-loop CFG-Focus schedule (no re-run)."

    def build(self, mode, base_cfg, focus_strength, focus_start, focus_end):
        return ({
            "type": "cfg_focus",
            "mode": mode,
            "base_cfg": float(base_cfg),
            "focus_strength": float(focus_strength),
            "focus_start": int(focus_start),
            "focus_end": int(focus_end),
        },)

NODE_CLASS_MAPPINGS = {"UltraCFG": UltraCFG}
NODE_DISPLAY_NAME_MAPPINGS = {"UltraCFG": "Ultra CFG (CFG-Focus)"}