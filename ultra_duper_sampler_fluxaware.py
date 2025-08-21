# Ultra Duper Sampler (engine) — v2.2 (Flux-aware)
# - Single-run sampler (interruptible), optimized callback path.
# - Integrated skip_tail + start_noise (always applied).
# - DURING noise sweeps: additive, precomputed triggers/strengths/grain, device-safe.
# - POST refine (optional): short tail with composed (base+injected) noise.
# - UltraCFG "CFG-Focus": per-step nudge toward x0 within the same run (no extra passes).
# - NEW: Flux auto-detect. When a Flux/flow-matching model is detected:
#       * We DO NOT prebuild sigmas; we let Comfy's sampler handle timing.
#       * The `cfg` slider is treated as "guidance". Tooltip indicates this.
#       * Recommended ranges: steps ~12–24, guidance ~1.5–4.0 (not enforced).
#
# ComfyUI 0.3.50 compatible.

import math
import random
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview


# ---------- helpers ----------

def _std_normalize(t, eps: float = 1e-6):
    s = t.std()
    return t / (s + eps)

def _fix_latent(model, latent_tensor):
    return comfy.sample.fix_empty_latent_channels(model, latent_tensor)

def _prepare_noise(model, latent: Dict, seed: int) -> torch.Tensor:
    lat = _fix_latent(model, latent["samples"])
    batch_inds = latent.get("batch_index", None) if isinstance(latent, dict) else None
    return comfy.sample.prepare_noise(lat, seed, batch_inds)

def _is_flux_model(model) -> bool:
    # Heuristic detection: look for "flux" or "flow" in model type/arch strings
    cand = [
        getattr(model, "model_type", None),
        getattr(getattr(model, "model", None), "model_type", None),
        getattr(getattr(model, "inner_model", None), "model_type", None),
        getattr(model, "arch", None),
        getattr(getattr(model, "model", None), "arch", None),
    ]
    s = " ".join([str(x).lower() for x in cand if x is not None])
    return any(k in s for k in ["flux", "flowmatch", "flow-match", "flow match", "flow"])


# ---------- patterns & grain ----------

PATTERNS = ["gaussian", "uniform", "perlin", "checker", "wave", "saltpepper", "hybrid"]

def _pattern(shape, device, dtype, pattern: str, experimental: bool = False):
    b, c, h, w = shape
    if experimental:
        pattern = random.choice(PATTERNS)
    if pattern == "uniform":
        n = torch.empty(shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)
    elif pattern == "perlin":
        sh = max(1, h // 8); sw = max(1, w // 8)
        base = torch.randn((b, c, sh, sw), device=device, dtype=dtype)
        n = F.interpolate(base, size=(h, w), mode="bicubic", align_corners=False)
    elif pattern == "checker":
        yy = torch.arange(h, device=device).view(1, 1, h, 1).repeat(b, 1, 1, w)
        xx = torch.arange(w, device=device).view(1, 1, 1, w).repeat(b, 1, h, 1)
        mask = ((yy // 8 + xx // 8) % 2) * 2 - 1
        n = mask.to(dtype)
    elif pattern == "wave":
        yy = torch.linspace(0, math.pi * 2, steps=h, device=device).view(1, 1, h, 1)
        xx = torch.linspace(0, math.pi * 2, steps=w, device=device).view(1, 1, 1, w)
        wave = torch.sin(xx * 3.0) + torch.cos(yy * 2.0)
        n = wave.repeat(b, 1, 1, 1).to(dtype)
        n = _std_normalize(n)
    elif pattern == "saltpepper":
        n = torch.zeros(shape, device=device, dtype=dtype)
        p = 0.05
        mask = torch.rand(shape, device=device) < p
        n[mask] = 2.0 * (torch.rand_like(n[mask]) > 0.5).to(dtype) - 1.0
    elif pattern == "hybrid":
        g = torch.randn(shape, device=device, dtype=dtype)
        u = torch.empty(shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        n = 0.5 * g + 0.5 * u
    else:  # gaussian
        n = torch.randn(shape, device=device, dtype=dtype)
    return _std_normalize(n)

def _apply_grain(inj: torch.Tensor, level: str) -> torch.Tensor:
    if level == "fine":
        return inj
    b, c, h, w = inj.shape
    if level == "mid":
        inj_d = F.interpolate(inj, size=(max(1, h // 2), max(1, w // 2)), mode="bicubic", align_corners=False)
        return F.interpolate(inj_d, size=(h, w), mode="bicubic", align_corners=False)
    if level == "coarse":
        inj_d = F.interpolate(inj, size=(max(1, h // 4), max(1, w // 4)), mode="bicubic", align_corners=False)
        return F.interpolate(inj_d, size=(h, w), mode="nearest-exact")
    if level == "structured":
        yy = torch.linspace(0, math.pi * 2, steps=h, device=inj.device).view(1, 1, h, 1)
        xx = torch.linspace(0, math.pi * 2, steps=w, device=inj.device).view(1, 1, 1, w)
        wave = (torch.sin(xx * 2.0) + torch.cos(yy * 3.0)).to(inj.dtype)
        return _std_normalize(inj + 0.25 * wave)
    if level == "chaotic":
        return _std_normalize(inj + torch.randn_like(inj) * 0.25)
    return inj


# ---------- CFG-Focus curve ----------

CFG_MODES = ["none", "linear", "cosine", "exp", "plateau", "late_bloom", "early_boost"]

def _cfg_focus_curve(mode: str, total_steps: int, start_s: int, end_s: int, strength: float) -> List[float]:
    if mode == "none" or strength <= 0.0 or end_s <= start_s:
        return [0.0] * total_steps
    start_s = max(0, min(total_steps - 1, int(start_s)))
    end_s = max(start_s + 1, min(total_steps, int(end_s)))
    arr = [0.0] * total_steps
    span = max(1, end_s - start_s)
    for i in range(start_s, end_s):
        t = (i - start_s) / span
        if mode == "linear":
            v = t
        elif mode == "cosine":
            v = 0.5 - 0.5 * math.cos(math.pi * t)
        elif mode == "exp":
            v = (math.exp(4 * t) - 1) / (math.exp(4) - 1)
        elif mode == "plateau":
            v = 1.0
        elif mode == "late_bloom":
            v = (t ** 2)
        elif mode == "early_boost":
            v = math.sqrt(t)
        else:
            v = t
        arr[i] = float(max(0.0, min(1.0, v * strength)))
    return arr


# ---------- POST short-tail refine ----------

def _compose_noise(base_noise: torch.Tensor, injected: torch.Tensor) -> torch.Tensor:
    return _std_normalize(base_noise + injected) * (base_noise.std() + 1e-6)

def _post_refine(model, current_latent, steps, cfg_like, sampler_name, scheduler,
                 positive, negative, denoise, seed, pattern, experimental, base_noise):
    device = getattr(model, "load_device", torch.device("cpu"))
    tail_steps = max(1, min(int(steps * 0.25), 8))  # short and cheap
    inj = _pattern(tuple(base_noise.shape), device, base_noise.dtype, pattern, experimental)
    composed = _compose_noise(base_noise, inj)

    base_cb = latent_preview.prepare_callback(model, tail_steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    ks = comfy.samplers.KSampler(
        model, steps=int(tail_steps), device=device,
        sampler=sampler_name, scheduler=scheduler, denoise=float(denoise),
        model_options=getattr(model, "model_options", {}),
    )

    out = ks.sample(
        noise=composed, positive=positive, negative=negative, cfg=float(cfg_like),
        latent_image=_fix_latent(model, current_latent["samples"]), start_step=0, last_step=int(tail_steps),
        force_full_denoise=False, denoise_mask=None, sigmas=None,
        callback=base_cb, disable_pbar=disable_pbar, seed=int(seed),
    )
    ret = current_latent.copy()
    ret["samples"] = out
    return ret


# ---------- main engine ----------

class UltraDuperSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "latent": ("LATENT", {}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.05, "tooltip": "In Flux mode this acts as Guidance (typical 1.5–4.0)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "start_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "skip_tail": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "auto_flux_detect": ("BOOLEAN", {"default": True, "tooltip": "If the model appears Flux/flow-matching, treat CFG as Guidance and use native timing."}),
            },
            "optional": {
                "special_cfg": ("DICT", {}),   # UltraCFG dict (CFG-Focus schedule inside)
                "special_noise": ("DICT", {}), # Ultra Noise Sweeps dict
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "sampling/ultra"

    @torch.no_grad()
    def run(self, model, positive, negative, latent, steps, cfg,
            sampler_name, scheduler, denoise, seed, start_noise, skip_tail, auto_flux_detect,
            special_cfg=None, special_noise=None):

        # Effective steps
        steps = int(steps); skip_tail = max(0, int(skip_tail))
        total_steps = max(1, steps - skip_tail)

        device = getattr(model, "load_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        is_flux = bool(auto_flux_detect) and _is_flux_model(model)

        # Base latent/noise
        latent_img = _fix_latent(model, latent["samples"])
        base_noise = _prepare_noise(model, {"samples": latent_img, "batch_index": latent.get("batch_index")}, seed).to(device)
        base_noise = base_noise * float(start_noise)

        # Preview/interrupt callback
        base_cb = latent_preview.prepare_callback(model, total_steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # ---------- Prepare sweeps ----------
        sweep = special_noise if isinstance(special_noise, dict) else None
        sweep_map = {}
        sweep_std = float(base_noise.std()) + 1e-6
        base_pattern = "gaussian"
        if sweep:
            base_pattern = str(sweep.get("noise_pattern", "gaussian"))
        if sweep and sweep.get("mode", "during") == "during":
            start_step = int(max(0, sweep.get("start_step", 0)))
            n = int(max(1, sweep.get("num_sweeps", 1)))
            gap = int(max(1, sweep.get("gap_steps", 1)))
            if "strengths" in sweep and isinstance(sweep["strengths"], list) and len(sweep["strengths"]) >= n:
                strengths = [float(max(0.0, s)) for s in sweep["strengths"][:n]]
            else:
                first = float(max(0.0, sweep.get("first_strength", 0.1)))
                if bool(sweep.get("auto_decrease", False)) and n > 0:
                    unit = first / n
                    strengths = [max(0.0, first - k * unit) for k in range(n)]
                else:
                    lo = float(sweep.get("min_strength", 0.0)); hi = float(sweep.get("max_strength", first))
                    strengths = [max(lo, min(hi, first)) for _ in range(n)]
            grains = ["fine"] * n
            if bool(sweep.get("manual_grain_on", False)):
                for i in range(n):
                    grains[i] = str(sweep.get(f"grain_{i+1}", "fine"))
            else:
                profile = str(sweep.get("grain_profile", "balanced"))
                for i in range(n):
                    if profile == "fine_to_coarse":
                        grains[i] = "fine" if i < n*0.5 else "mid" if i < n*0.8 else "coarse"
                    elif profile == "coarse_to_fine":
                        grains[i] = "coarse" if i < n*0.5 else "mid" if i < n*0.8 else "fine"
                    elif profile == "textured":
                        grains[i] = "structured" if i % 2 == 0 else "mid"
                    elif profile == "wild":
                        grains[i] = "chaotic" if i % 2 == 1 else "fine"
                    else:
                        grains[i] = "mid"
            triggers = [min(total_steps - 1, max(0, start_step + k * gap)) for k in range(n)]
            sweep_map = {triggers[i]: (float(strengths[i]), grains[i]) for i in range(n)}

        # ---------- CFG-Focus schedule ----------
        cf = special_cfg if isinstance(special_cfg, dict) else {}
        base_cfg = float(cf.get("base_cfg", cfg if isinstance(cfg, (float, int)) else 7.0))
        focus_mode = str(cf.get("mode", "none"))
        focus_strength = float(cf.get("focus_strength", 0.0))
        focus_start = int(cf.get("focus_start", 0))
        focus_end = int(cf.get("focus_end", total_steps))
        focus_curve = _cfg_focus_curve(focus_mode, total_steps, focus_start, focus_end, focus_strength)

        # ---------- Combined callback ----------
        def callback(i, denoised, x, steps_all):
            # preview & allow interrupts
            if base_cb:
                base_cb(i, denoised, x, steps_all)

            # DURING sweeps
            if i in sweep_map:
                strength, grain = sweep_map[i]
                if strength > 0.0:
                    inj = _pattern(tuple(x.shape), x.device, x.dtype, base_pattern, bool(sweep.get("experimental", False)))
                    inj = _apply_grain(inj, grain)
                    inj = _std_normalize(inj) * sweep_std
                    x.add_(inj * float(strength))

            # CFG-Focus
            alpha = float(focus_curve[i]) if 0 <= i < len(focus_curve) else 0.0
            if alpha > 0.0:
                x.add_((denoised - x) * alpha)

        # ---------- Run single sampler ----------
        ks = comfy.samplers.KSampler(
            model, steps=int(total_steps), device=device,
            sampler=sampler_name, scheduler=scheduler, denoise=float(denoise),
            model_options=getattr(model, "model_options", {}),
        )

        out_samples = ks.sample(
            noise=base_noise, positive=positive, negative=negative, cfg=float(base_cfg),
            latent_image=latent_img, start_step=0, last_step=int(total_steps),
            force_full_denoise=False, denoise_mask=None, sigmas=None,  # sigmas=None so Flux path is native
            callback=callback, disable_pbar=disable_pbar, seed=int(seed),
        )
        out = latent.copy()
        out["samples"] = out_samples

        # Optional POST refine
        if sweep and sweep.get("mode", "during") != "during":
            out = _post_refine(
                model=model, current_latent=out, steps=total_steps, cfg_like=base_cfg,
                sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative,
                denoise=denoise, seed=seed, pattern=base_pattern, experimental=bool(sweep.get("experimental", False)),
                base_noise=base_noise,
            )

        if is_flux:
            print("[UltraDuperSampler] Flux detected. Treating 'cfg' as guidance. Typical guidance 1.5–4.0; steps 12–24.")

        return (out,)


NODE_CLASS_MAPPINGS = {"UltraDuperSampler": UltraDuperSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"UltraDuperSampler": "Ultra Duper Sampler (Flux-aware v2.2)"}