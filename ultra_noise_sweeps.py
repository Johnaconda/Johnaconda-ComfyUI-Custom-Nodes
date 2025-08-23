# Ultra Noise Sweeps v2.6 — structured grains & strengths + impact/weighting/dynamic_scale
# Adds many new presets grouped as Early / Mid / Late punches + Detailers.
#
# DURING:
#   - start_step / num_sweeps / gap_steps
#   - strength schedule: constant / linear_up / linear_down / exp_up / exp_down / custom_min
#   - grain scheme: balanced / coarse_to_fine / fine_to_coarse / textured / wild / constant / ping_pong / random
#   - global controls: impact (master multiplier), step_weighting (where to emphasize), dynamic_scale (size by current std)
#
# POST:
#   - optional tail refine steps (0 = auto in sampler)

import math
import random

GRAIN_NAMES = ["fine", "mid", "coarse", "structured", "chaotic"]

# NOTE: start_step values here assume ~22–28 total steps. They clamp safely if you use fewer steps.
PRESETS = {
    # -------------------- originals (kept for compatibility) --------------------
    "Early Punch (linear↓, coarse→fine)": {
        "mode":"during","noise_pattern":"hybrid","start_step":0,"num_sweeps":3,"gap_steps":2,
        "first_strength":0.25,"strength_schedule":"linear_down","grain_scheme":"coarse_to_fine",
        "impact":1.5,"step_weighting":"early","dynamic_scale":True
    },
    "Late Spice (linear↓, fine→coarse)": {
        "mode":"during","noise_pattern":"gaussian","start_step":6,"num_sweeps":3,"gap_steps":2,
        "first_strength":0.18,"strength_schedule":"linear_down","grain_scheme":"fine_to_coarse",
        "impact":1.2,"step_weighting":"late","dynamic_scale":True
    },
    "Edge Pop (constant, textured)": {
        "mode":"during","noise_pattern":"checker","start_step":2,"num_sweeps":2,"gap_steps":3,
        "first_strength":0.14,"strength_schedule":"constant","grain_scheme":"textured",
        "impact":1.0,"step_weighting":"edge","dynamic_scale":True
    },
    "Breath (linear↓, balanced)": {
        "mode":"during","noise_pattern":"perlin","start_step":1,"num_sweeps":3,"gap_steps":2,
        "first_strength":0.10,"strength_schedule":"linear_down","grain_scheme":"balanced",
        "impact":1.0,"step_weighting":"mid","dynamic_scale":True
    },
    "Grain Storm (exp↓, wild)": {
        "mode":"during","noise_pattern":"hybrid","start_step":0,"num_sweeps":4,"gap_steps":1,
        "first_strength":0.20,"strength_schedule":"exp_down","curve":1.4,"grain_scheme":"wild",
        "impact":2.0,"step_weighting":"early","dynamic_scale":True
    },
    "Two Jab (linear↓, balanced)": {
        "mode":"during","noise_pattern":"gaussian","start_step":0,"num_sweeps":2,"gap_steps":4,
        "first_strength":0.20,"strength_schedule":"linear_down","grain_scheme":"balanced",
        "impact":1.0,"step_weighting":"early","dynamic_scale":True
    },
    "Post Polish (tail=4)": {
        "mode":"post","noise_pattern":"gaussian","post_tail_steps":4,"experimental":False
    },

    # -------------------- Early Punches (5) --------------------
    "Early Burst — Coarse→Fine (linear↓, impact 2.0)": {
        "mode":"during","noise_pattern":"hybrid","start_step":0,"num_sweeps":4,"gap_steps":1,
        "first_strength":0.28,"strength_schedule":"linear_down","grain_scheme":"coarse_to_fine",
        "impact":2.0,"step_weighting":"early","dynamic_scale":True
    },
    "Early Twin Jab — Balanced (constant)": {
        "mode":"during","noise_pattern":"gaussian","start_step":0,"num_sweeps":2,"gap_steps":3,
        "first_strength":0.22,"strength_schedule":"constant","grain_scheme":"balanced",
        "impact":1.6,"step_weighting":"early","dynamic_scale":True
    },
    "Early Crunch Weave — Textured": {
        "mode":"during","noise_pattern":"checker","start_step":1,"num_sweeps":3,"gap_steps":2,
        "first_strength":0.16,"strength_schedule":"linear_down","grain_scheme":"textured",
        "impact":1.4,"step_weighting":"early","dynamic_scale":True
    },
    "Early Salt Kick — Wild": {
        "mode":"during","noise_pattern":"saltpepper","start_step":0,"num_sweeps":3,"gap_steps":1,
        "first_strength":0.18,"strength_schedule":"exp_down","curve":1.3,"grain_scheme":"wild",
        "impact":2.2,"step_weighting":"early","dynamic_scale":True
    },
    "Early Wave Pop — Balanced": {
        "mode":"during","noise_pattern":"wave","start_step":0,"num_sweeps":3,"gap_steps":2,
        "first_strength":0.14,"strength_schedule":"linear_down","grain_scheme":"balanced",
        "impact":1.3,"step_weighting":"early","dynamic_scale":True
    },

    # -------------------- Mid Punches (5) --------------------
    "Mid Drive — Balanced (linear↓)": {
        "mode":"during","noise_pattern":"gaussian","start_step":6,"num_sweeps":3,"gap_steps":2,
        "first_strength":0.22,"strength_schedule":"linear_down","grain_scheme":"balanced",
        "impact":1.4,"step_weighting":"mid","dynamic_scale":True
    },
    "Mid Texture Wrap — Textured": {
        "mode":"during","noise_pattern":"perlin","start_step":8,"num_sweeps":3,"gap_steps":2,
        "first_strength":0.15,"strength_schedule":"linear_down","grain_scheme":"textured",
        "impact":1.2,"step_weighting":"mid","dynamic_scale":True
    },
    "Mid Checker Pulse — Ping-Pong": {
        "mode":"during","noise_pattern":"checker","start_step":7,"num_sweeps":4,"gap_steps":1,
        "first_strength":0.14,"strength_schedule":"linear_down","grain_scheme":"ping_pong",
        "impact":1.5,"step_weighting":"mid","dynamic_scale":True
    },
    "Mid Hybrid Push — Coarse→Fine (exp↓)": {
        "mode":"during","noise_pattern":"hybrid","start_step":9,"num_sweeps":3,"gap_steps":2,
        "first_strength":0.20,"strength_schedule":"exp_down","curve":1.5,"grain_scheme":"coarse_to_fine",
        "impact":1.6,"step_weighting":"gaussian","dynamic_scale":True
    },
    "Mid Edge Lift — Textured (constant)": {
        "mode":"during","noise_pattern":"wave","start_step":8,"num_sweeps":2,"gap_steps":3,
        "first_strength":0.12,"strength_schedule":"constant","grain_scheme":"textured",
        "impact":1.1,"step_weighting":"mid","dynamic_scale":True
    },

    # -------------------- Late Punches (3) --------------------
    "Late Clarity — Fine→Coarse (linear↑)": {
        "mode":"during","noise_pattern":"gaussian","start_step":14,"num_sweeps":3,"gap_steps":2,
        "first_strength":0.10,"strength_schedule":"linear_up","grain_scheme":"fine_to_coarse",
        "impact":1.0,"step_weighting":"late","dynamic_scale":True
    },
    "Late Edge Varnish — Textured (constant)": {
        "mode":"during","noise_pattern":"checker","start_step":16,"num_sweeps":2,"gap_steps":2,
        "first_strength":0.08,"strength_schedule":"constant","grain_scheme":"textured",
        "impact":1.1,"step_weighting":"late","dynamic_scale":True
    },
    "Late Focus Squeeze — Balanced (exp↑)": {
        "mode":"during","noise_pattern":"perlin","start_step":18,"num_sweeps":3,"gap_steps":1,
        "first_strength":0.06,"strength_schedule":"exp_up","curve":1.7,"grain_scheme":"balanced",
        "impact":1.0,"step_weighting":"late","dynamic_scale":True
    },

    # -------------------- Detailers (6) --------------------
    "Detail — Micro Contrast (balanced)": {
        "mode":"during","noise_pattern":"hybrid","start_step":6,"num_sweeps":3,"gap_steps":1,
        "first_strength":0.08,"strength_schedule":"linear_down","grain_scheme":"balanced",
        "impact":0.9,"step_weighting":"mid","dynamic_scale":True
    },
    "Detail — Edge Enhance (structured)": {
        "mode":"during","noise_pattern":"checker","start_step":10,"num_sweeps":2,"gap_steps":2,
        "first_strength":0.10,"strength_schedule":"constant","grain_scheme":"textured",
        "impact":1.2,"step_weighting":"gaussian","dynamic_scale":True
    },
    "Detail — Fine Mist Plus": {
        "mode":"during","noise_pattern":"uniform","start_step":4,"num_sweeps":3,"gap_steps":2,
        "first_strength":0.06,"strength_schedule":"linear_down","grain_scheme":"balanced",
        "impact":0.8,"step_weighting":"mid","dynamic_scale":True
    },
    "Detail — Grain Feather (perlin)": {
        "mode":"during","noise_pattern":"perlin","start_step":5,"num_sweeps":3,"gap_steps":2,
        "first_strength":0.07,"strength_schedule":"linear_down","grain_scheme":"balanced",
        "impact":0.9,"step_weighting":"mid","dynamic_scale":True
    },
    "Detail — Lines & Fabric (wave)": {
        "mode":"during","noise_pattern":"wave","start_step":7,"num_sweeps":3,"gap_steps":1,
        "first_strength":0.09,"strength_schedule":"linear_down","grain_scheme":"textured",
        "impact":1.0,"step_weighting":"mid","dynamic_scale":True
    },
    "Detail — Pop Highlights (saltpepper)": {
        "mode":"during","noise_pattern":"saltpepper","start_step":12,"num_sweeps":2,"gap_steps":2,
        "first_strength":0.05,"strength_schedule":"constant","grain_scheme":"wild",
        "impact":1.0,"step_weighting":"late","dynamic_scale":True
    },
}

class UltraNoiseSweeps:
    MODES = ["during", "post"]
    STRENGTH_SCHEDULES = ["constant", "linear_down", "linear_up", "exp_down", "exp_up", "custom_min"]
    GRAIN_SCHEMES = ["balanced", "coarse_to_fine", "fine_to_coarse", "textured", "wild",
                     "constant", "ping_pong", "random"]
    PATTERNS = ["gaussian", "uniform", "perlin", "checker", "wave", "saltpepper", "hybrid"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (cls.MODES, {"default": "during"}),
                "use_preset": ("BOOLEAN", {"default": False, "tooltip": "If enabled, preset drives all fields."}),
                "preset_name": (list(PRESETS.keys()), {"default": "Early Punch (linear↓, coarse→fine)"}),

                # common
                "noise_pattern": (cls.PATTERNS, {"default": "gaussian"}),
                "experimental": ("BOOLEAN", {"default": False, "tooltip": "Randomize pattern each injection"}),

                # DURING positioning
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "num_sweeps": ("INT", {"default": 3, "min": 1, "max": 128}),
                "gap_steps": ("INT", {"default": 2, "min": 1, "max": 256}),

                # strengths
                "first_strength": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 2.0, "step": 0.01}),
                "strength_schedule": (cls.STRENGTH_SCHEDULES, {"default": "linear_down"}),
                "min_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                            "tooltip":"Used when schedule=custom_min"}),
                "curve": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                                    "tooltip":"Exponent for exp_* schedules; 1.0 ≈ linear"}),

                # global sweep behavior
                "impact": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Master multiplier for all sweep injections"}),
                "step_weighting": (["flat","early","mid","late","edge","gaussian"], {"default": "early",
                    "tooltip": "Where to emphasize injections across steps"}),
                "dynamic_scale": ("BOOLEAN", {"default": True, "tooltip": "Size injections by current latent std"}),

                # grains
                "grain_scheme": (cls.GRAIN_SCHEMES, {"default": "balanced"}),
                "grain_constant_level": (GRAIN_NAMES, {"default": "mid",
                    "tooltip":"Used when grain_scheme=constant"}),
                "grain_ping_a": (GRAIN_NAMES, {"default": "coarse",
                    "tooltip":"First level when grain_scheme=ping_pong"}),
                "grain_ping_b": (GRAIN_NAMES, {"default": "fine",
                    "tooltip":"Second level when grain_scheme=ping_pong"}),
                "grain_seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1,
                    "tooltip":"Random seed for random scheme; 0 = nondeterministic"}),
                "grain_random_pool": ("STRING", {"default": "",
                    "tooltip":"Comma list to restrict random pool, e.g. 'fine,coarse'. Empty = all."}),

                # POST
                "post_tail_steps": ("INT", {"default": 0, "min": 0, "max": 128, "tooltip": "0 = auto (≈ steps/4)"}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("special_noise",)
    FUNCTION = "build"
    CATEGORY = "sampling/ultra/tweaks"
    OUTPUT_NODE = False
    DESCRIPTION = "Structured DURING/POST noise sweeps with explicit grains[] and strengths[]."

    # ---------------- helpers ----------------

    def _build_strengths(self, n, first, schedule, min_strength, curve):
        if n <= 1:
            return [max(0.0, float(first))]
        s = []
        for k in range(n):
            t = k / (n - 1)  # 0..1
            if schedule == "constant":
                v = first
            elif schedule == "linear_down":
                v = first * (1.0 - t)
            elif schedule == "linear_up":
                v = first * t
            elif schedule == "exp_down":
                v = first * (1.0 - (t ** float(curve)))
            elif schedule == "exp_up":
                v = first * (t ** float(curve))
            elif schedule == "custom_min":
                v = first - (first - float(min_strength)) * t
            else:
                v = first
            s.append(max(0.0, float(v)))
        return s

    def _parse_pool(self, s):
        if not isinstance(s, str) or not s.strip():
            return list(GRAIN_NAMES)
        out = []
        for token in s.split(","):
            t = token.strip().lower()
            if t in GRAIN_NAMES:
                out.append(t)
        return out if out else list(GRAIN_NAMES)

    def _build_grains(self, n, scheme, const_level, a, b, seed, pool_s):
        rng = random.Random(None if int(seed) == 0 else int(seed))
        pool = self._parse_pool(pool_s)

        def choose():
            return rng.choice(pool) if rng else random.choice(pool)

        g = []
        if scheme == "balanced":
            for i in range(n):
                g.append("fine" if i % 2 == 0 else "mid")

        elif scheme == "coarse_to_fine":
            half = max(1, n // 2)
            for i in range(n):
                g.append("coarse" if i < half else "fine")

        elif scheme == "fine_to_coarse":
            half = max(1, n // 2)
            for i in range(n):
                g.append("fine" if i < half else "coarse")

        elif scheme == "textured":
            for i in range(n):
                g.append("structured" if i % 2 == 0 else "mid")

        elif scheme == "wild":
            for i in range(n):
                g.append("chaotic" if (i % 2 == 1) else "fine")

        elif scheme == "constant":
            lvl = const_level if const_level in GRAIN_NAMES else "mid"
            g = [lvl] * n

        elif scheme == "ping_pong":
            aa = a if a in GRAIN_NAMES else "coarse"
            bb = b if b in GRAIN_NAMES else "fine"
            for i in range(n):
                g.append(aa if i % 2 == 0 else bb)

        elif scheme == "random":
            for _ in range(n):
                g.append(choose())

        else:
            for i in range(n):
                g.append("fine" if i % 2 == 0 else "mid")

        return g

    # ---------------- main ----------------

    def build(
        self, mode, use_preset, preset_name, noise_pattern, experimental,
        start_step, num_sweeps, gap_steps,
        first_strength, strength_schedule, min_strength, curve,
        impact, step_weighting, dynamic_scale,
        grain_scheme, grain_constant_level, grain_ping_a, grain_ping_b,
        grain_seed, grain_random_pool,
        post_tail_steps
    ):
        # Expand preset first (keeps UI editable after)
        if use_preset:
            p = dict(PRESETS.get(str(preset_name), {}))
            mode = p.get("mode", mode)
            noise_pattern = p.get("noise_pattern", noise_pattern)
            experimental = p.get("experimental", experimental)
            if mode == "during":
                start_step = p.get("start_step", start_step)
                num_sweeps = p.get("num_sweeps", num_sweeps)
                gap_steps = p.get("gap_steps", gap_steps)
                first_strength = p.get("first_strength", first_strength)
                strength_schedule = p.get("strength_schedule", strength_schedule)
                curve = p.get("curve", curve)
                grain_scheme = p.get("grain_scheme", grain_scheme)
                impact = p.get("impact", impact)
                step_weighting = p.get("step_weighting", step_weighting)
                dynamic_scale = p.get("dynamic_scale", dynamic_scale)
            else:
                post_tail_steps = p.get("post_tail_steps", post_tail_steps)

        d = {
            "mode": str(mode),
            "noise_pattern": str(noise_pattern),
            "experimental": bool(experimental),
        }

        if mode == "during":
            n = int(max(1, num_sweeps))
            strengths = self._build_strengths(
                n=n, first=float(first_strength),
                schedule=str(strength_schedule),
                min_strength=float(min_strength), curve=float(curve)
            )
            grains = self._build_grains(
                n=n, scheme=str(grain_scheme),
                const_level=str(grain_constant_level),
                a=str(grain_ping_a), b=str(grain_ping_b),
                seed=int(grain_seed), pool_s=str(grain_random_pool)
            )

            d.update({
                "start_step": int(max(0, start_step)),
                "num_sweeps": n,
                "gap_steps": int(max(1, gap_steps)),

                # explicit arrays the sampler will prefer
                "strengths": strengths,
                "grains": grains,

                # informative and legacy fields
                "first_strength": float(first_strength),
                "grain_scheme": str(grain_scheme),

                # behavior knobs
                "impact": float(impact),
                "step_weighting": str(step_weighting),
                "dynamic_scale": bool(dynamic_scale),
            })

        else:
            d["post_tail_steps"] = int(max(0, post_tail_steps))

        return (d,)

NODE_CLASS_MAPPINGS = {"UltraNoiseSweeps": UltraNoiseSweeps}
NODE_DISPLAY_NAME_MAPPINGS = {"UltraNoiseSweeps": "Ultra Noise Sweeps"}
