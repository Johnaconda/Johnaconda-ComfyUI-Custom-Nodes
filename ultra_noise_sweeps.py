# Ultra Noise Sweeps v2.2 — clearer controls, presets, auto decrease, manual grain list
# Acts as an "enchanter": adds noise during/post without replacing base start noise.
#
# DURING mode: choose the first step, the number of sweeps, gap between, and the FIRST strength.
# - If "auto_decrease" is ON, later sweeps are computed as: s_k = first_strength - k*(first_strength/num_sweeps).
# - You can also choose a grain profile or specify manual grain per-sweep.
# PRESETS (when enabled) overwrite the fields with proven combos.
#
# POST mode (optional): the engine will compose base+injected and do a short tail refine.

PRESETS = {
    "Early Punch":      {"mode":"during","noise_pattern":"hybrid","start_step":0,"num_sweeps":2,"gap_steps":2,"first_strength":0.25,"auto_decrease":True,"grain_profile":"coarse_to_fine"},
    "Late Spice":       {"mode":"during","noise_pattern":"gaussian","start_step":6,"num_sweeps":3,"gap_steps":2,"first_strength":0.18,"auto_decrease":True,"grain_profile":"fine_to_coarse"},
    "Edge Pop":         {"mode":"during","noise_pattern":"checker","start_step":2,"num_sweeps":2,"gap_steps":3,"first_strength":0.14,"auto_decrease":False,"grain_profile":"textured"},
    "Breath":           {"mode":"during","noise_pattern":"perlin","start_step":1,"num_sweeps":3,"gap_steps":1,"first_strength":0.10,"auto_decrease":True,"grain_profile":"balanced"},
    "Staccato":         {"mode":"during","noise_pattern":"saltpepper","start_step":0,"num_sweeps":4,"gap_steps":1,"first_strength":0.12,"auto_decrease":True,"grain_profile":"wild"},
    "Fine Mist":        {"mode":"during","noise_pattern":"uniform","start_step":0,"num_sweeps":2,"gap_steps":2,"first_strength":0.08,"auto_decrease":True,"grain_profile":"balanced"},
    "Grain Storm":      {"mode":"during","noise_pattern":"hybrid","start_step":3,"num_sweeps":5,"gap_steps":1,"first_strength":0.20,"auto_decrease":True,"grain_profile":"wild"},
    "Two Jab":          {"mode":"during","noise_pattern":"gaussian","start_step":1,"num_sweeps":2,"gap_steps":4,"first_strength":0.16,"auto_decrease":True,"grain_profile":"balanced"},
    "Triple Hook":      {"mode":"during","noise_pattern":"hybrid","start_step":2,"num_sweeps":3,"gap_steps":2,"first_strength":0.22,"auto_decrease":True,"grain_profile":"textured"},
    "Chaos Then Calm":  {"mode":"during","noise_pattern":"saltpepper","start_step":0,"num_sweeps":3,"gap_steps":1,"first_strength":0.25,"auto_decrease":True,"grain_profile":"coarse_to_fine"},
}

class UltraNoiseSweepsV2:
    MODES = ["during", "post"]
    PATTERNS = ["gaussian", "uniform", "perlin", "checker", "wave", "saltpepper", "hybrid"]
    GRAINS = ["fine", "mid", "coarse", "structured", "chaotic"]
    GRAIN_PROFILES = ["balanced","fine_to_coarse","coarse_to_fine","textured","wild"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # core
                "mode": (cls.MODES, {}),
                "noise_pattern": (cls.PATTERNS, {}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "num_sweeps": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),  # capped to 10 to match manual grains
                "gap_steps": ("INT", {"default": 2, "min": 1, "max": 256, "step": 1}),

                # strengths (simple and explicit)
                "first_strength": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.01}),
                "min_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "auto_decrease": ("BOOLEAN", {"default": True, "tooltip": "Compute later sweeps from first_strength / num_sweeps"}),

                # grain options
                "grain_profile": (cls.GRAIN_PROFILES, {"tooltip": "Preset grain pattern across sweeps"}),
                "manual_grain_on": ("BOOLEAN", {"default": False}),
                "grain_1": (cls.GRAINS, {}),
                "grain_2": (cls.GRAINS, {}),
                "grain_3": (cls.GRAINS, {}),
                "grain_4": (cls.GRAINS, {}),
                "grain_5": (cls.GRAINS, {}),
                "grain_6": (cls.GRAINS, {}),
                "grain_7": (cls.GRAINS, {}),
                "grain_8": (cls.GRAINS, {}),
                "grain_9": (cls.GRAINS, {}),
                "grain_10": (cls.GRAINS, {}),

                # presets
                "use_preset": ("BOOLEAN", {"default": False}),
                "preset": (list(PRESETS.keys()), {}),

                # misc
                "experimental": ("BOOLEAN", {"default": False, "tooltip": "Randomize base pattern per sweep"}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("special_noise",)
    FUNCTION = "build"
    CATEGORY = "sampling/ultra/tweaks"
    DESCRIPTION = "Add-on noise schedule (during/post). DURING: first step + first strength with optional auto decrease; plus grain profile or manual per-sweep grain."

    def build(self, mode, noise_pattern, start_step, num_sweeps, gap_steps,
              first_strength, min_strength, max_strength, auto_decrease,
              grain_profile, manual_grain_on,
              grain_1, grain_2, grain_3, grain_4, grain_5, grain_6, grain_7, grain_8, grain_9, grain_10,
              use_preset, preset, experimental):

        # Apply preset if asked
        if use_preset and preset in PRESETS:
            pr = PRESETS[preset]
            mode = pr["mode"]
            noise_pattern = pr["noise_pattern"]
            start_step = pr["start_step"]
            num_sweeps = pr["num_sweeps"]
            gap_steps = pr["gap_steps"]
            first_strength = pr["first_strength"]
            auto_decrease = pr["auto_decrease"]
            grain_profile = pr["grain_profile"]

        # Strengths
        n = int(max(1, num_sweeps))
        first = float(max(0.0, first_strength))
        lo = float(min_strength); hi = float(max_strength)
        if auto_decrease:
            unit = first / n
            strengths = [max(lo, min(hi, first - k * unit)) for k in range(n)]
        else:
            strengths = [max(lo, min(hi, first)) for _ in range(n)]

        # Grains
        grains = [grain_1, grain_2, grain_3, grain_4, grain_5, grain_6, grain_7, grain_8, grain_9, grain_10]
        plan = {
            "type": "noise_sweeps_v2",
            "mode": mode,
            "noise_pattern": noise_pattern,
            "start_step": int(start_step),
            "num_sweeps": n,
            "gap_steps": int(gap_steps),
            "strengths": strengths,
            "experimental": bool(experimental),
        }
        if manual_grain_on:
            plan["manual_grain_on"] = True
            for i in range(n):
                plan[f"grain_{i+1}"] = grains[i]
        else:
            plan["manual_grain_on"] = False
            plan["grain_profile"] = grain_profile

        return (plan,)


NODE_CLASS_MAPPINGS = {"UltraNoiseSweepsV2": UltraNoiseSweepsV2}
NODE_DISPLAY_NAME_MAPPINGS = {"UltraNoiseSweepsV2": "Ultra Noise Sweeps v2.2 (presets)"}