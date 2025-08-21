# Ultra Duper Sampler (engine) — v2.0
# - Integrated skip-tail INT input (removed special_skiptail)
# - Removed special_plan / multi-segment path (single schedule always)
# - Start noise is ALWAYS applied; special noise is additive (during) or composed (post)
# - DURING: generate injected noise on x.device/x.dtype
# - POST: compose injected with base (normalize(base+inj) * base.std())
#
# Compatible with ComfyUI 0.3.50; keeps FUNCTION='run' and 'latent' arg name.

import math
import random
import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview

# ---------- helpers: UI / sampler plumbing ----------

def _preview_callback(model, steps):
    return latent_preview.prepare_callback(model, steps), (not comfy.utils.PROGRESS_BAR_ENABLED)

def _fix_latent(model, latent_tensor):
    return comfy.sample.fix_empty_latent_channels(model, latent_tensor)

def _prepare_noise(model, latent, seed):
    lat = _fix_latent(model, latent["samples"])
    batch_inds = latent.get("batch_index", None) if isinstance(latent, dict) else None
    return comfy.sample.prepare_noise(lat, seed, batch_inds)

def _build_sigmas_for(model, scheduler: str, steps: int, device):
    ks = comfy.samplers.KSampler(
        model, steps=int(steps), device=device,
        sampler="euler", scheduler=scheduler, denoise=1.0,
        model_options=getattr(model, "model_options", {}),
    )
    return ks.sigmas.to(device)

# ---------- helpers: noise patterns ----------

def _std_normalize(t, eps=1e-6):
    s = t.std()
    return t / (s + eps)

def _make_pattern(shape, device, dtype, pattern: str, experimental: bool = False):
    b, c, h, w = shape
    if experimental:
        pattern = random.choice(["gaussian", "uniform", "perlin", "checker", "wave", "saltpepper", "hybrid"])
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
    else:
        n = torch.randn(shape, device=device, dtype=dtype)
    return _std_normalize(n)

def _compose_noise(base_noise: torch.Tensor, injected: torch.Tensor) -> torch.Tensor:
    return _std_normalize(base_noise + injected) * (base_noise.std() + 1e-6)

# ---------- sweep strength (supports auto_harsh) ----------

def _strength_for(k, plan):
    """
    Compute strength for sweep k using either auto_harsh or legacy modes.
    """
    n = max(1, int(plan.get("num_sweeps", 1)))
    init = float(plan.get("initial_strength", 0.10))
    lo = float(plan.get("min_strength", 0.0))
    hi = float(plan.get("max_strength", 1.0))

    if bool(plan.get("auto_harsh", False)):
        harsh = max(0.0, min(1.0, float(plan.get("harshness", 0.5))))
        p = 1.0 + 3.0 * harsh          # exponent 1..4
        t = (k + 1) / n                # 1/N .. 1
        val = lo + (hi - lo) * (t ** p)
        # pin first sweep near init for predictability
        if k == 0:
            val = max(lo, min(hi, init))
        return float(val)

    # legacy
    if bool(plan.get("reducer", False)):
        val = init * (1.0 - (k / n))
    else:
        mode = plan.get("decay_mode", "none")
        rate = float(plan.get("decay_rate", 0.0))
        if mode == "linear":
            val = init + (hi - init) * (k / max(1, n - 1))
        elif mode == "exp":
            val = init * math.exp(rate * (k / max(1, n - 1)))
        else:
            val = init
    return float(max(lo, min(hi, val)))

# ---------- core run-with-sigmas (supports DURING via callback) ----------

def _run_with_sigmas(
    model, noise, sigmas, sampler_name, cfg,
    positive, negative, latent, denoise, seed,
    during_plan=None, base_noise_for_patterns=None
):
    steps = sigmas.numel() - 1
    lat_img = _fix_latent(model, latent["samples"])

    # base preview callback
    base_cb, disable_pbar = _preview_callback(model, steps)

    # DURING-mode injections
    if isinstance(during_plan, dict) and during_plan.get("mode", "during") == "during":
        start = int(max(0, during_plan.get("start_step", 0)))
        gap   = int(max(1, during_plan.get("gap_steps", 2)))
        n     = int(max(0, during_plan.get("num_sweeps", 0)))
        triggers = [min(steps - 1, start + i * gap) for i in range(n)] if n > 0 else []
        state = {"k": 0}

        def wrapped_cb(*args, **kwargs):
            # Accept (i, denoised, x, total_steps) style
            if len(args) >= 4 and isinstance(args[0], (int, torch.Tensor)):
                i = int(args[0]); x = args[2]
                if i in triggers and state["k"] < len(triggers):
                    s = float(_strength_for(state["k"], during_plan))
                    if s > 0.0 and x is not None:
                        inj = _make_pattern(tuple(x.shape), x.device, x.dtype,
                                            str(during_plan.get("noise_pattern", "gaussian")),
                                            bool(during_plan.get("experimental", False)))
                        inj = _std_normalize(inj) * (base_noise_for_patterns.std() + 1e-6)
                        # Additive on top
                        x.add_(inj * s)
                    state["k"] += 1
                if base_cb: base_cb(*args, **kwargs)
                return
            # Dict style
            if len(args) == 1 and isinstance(args[0], dict):
                data = args[0]; i = int(data.get("i", -1)); x = data.get("x", None)
                if i in triggers and state["k"] < len(triggers):
                    s = float(_strength_for(state["k"], during_plan))
                    if s > 0.0 and x is not None:
                        inj = _make_pattern(tuple(x.shape), x.device, x.dtype,
                                            str(during_plan.get("noise_pattern", "gaussian")),
                                            bool(during_plan.get("experimental", False)))
                        inj = _std_normalize(inj) * (base_noise_for_patterns.std() + 1e-6)
                        data["x"] = x.add(inj * s)
                    state["k"] += 1
                if base_cb: base_cb(data)
                return
            # Fallback
            if base_cb: base_cb(*args, **kwargs)

        cb_to_use = wrapped_cb
    else:
        cb_to_use = base_cb

    ks = comfy.samplers.KSampler(
        model,
        steps=int(steps),
        device=getattr(model, "load_device", torch.device("cpu")),
        sampler=sampler_name,
        scheduler="karras",  # ignored when sigmas provided
        denoise=float(denoise),
        model_options=getattr(model, "model_options", {}),
    )

    out = ks.sample(
        noise=noise, positive=positive, negative=negative, cfg=float(cfg),
        latent_image=lat_img, start_step=0, last_step=int(steps),
        force_full_denoise=True, denoise_mask=None, sigmas=sigmas,
        callback=cb_to_use, disable_pbar=disable_pbar, seed=int(seed) if seed is not None else None,
    )
    ret = latent.copy()
    ret["samples"] = out
    return ret

# ---------- POST mode refiner (compose noise) ----------

def _apply_post(model, current, total_steps, scheduler, sampler_name, cfg,
                positive, negative, denoise, seed, device, base_noise, plan):

    full = _build_sigmas_for(model, scheduler, int(total_steps), device)
    tail = int(max(1, min(int(plan.get("refine_window", 6)), full.numel() - 1)))
    tail_sig = full[-(tail + 1):]

    n = int(max(0, plan.get("num_sweeps", 0)))
    for k in range(n):
        s = _strength_for(k, plan)
        pattern = plan.get("noise_pattern", "gaussian")
        exp = bool(plan.get("experimental", False))
        inj = _make_pattern(tuple(base_noise.shape), base_noise.device, base_noise.dtype, pattern, exp)
        composed = _compose_noise(base_noise, inj) * float(max(0.0, s))
        current = _run_with_sigmas(
            model=model, noise=composed, sigmas=tail_sig,
            sampler_name=sampler_name, cfg=float(cfg),
            positive=positive, negative=negative,
            latent=current, denoise=denoise, seed=seed,
            during_plan=None, base_noise_for_patterns=base_noise
        )
    return current

# ---------- main engine ----------

class UltraDuperSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning."}),
                "latent": ("LATENT", {"tooltip": "Latent to denoise."}),
                "steps": ("INT", {"default": 32, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Noise schedule."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "start_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "tooltip": "Scale the initial base noise (global)."}),
                "skip_tail": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "Trim late steps from the schedule."}),
            },
            "optional": {
                "special_cfg": ("DICT", {}),        # UltraCFG
                "special_noise": ("DICT", {}),      # Ultra Noise Sweeps v2
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "sampling/ultra"

    @torch.no_grad()
    def run(self, model, positive, negative, latent, steps, cfg,
            sampler_name, scheduler, denoise, seed, start_noise, skip_tail,
            special_cfg=None, special_noise=None):

        # sanitize tweak dicts (do not enforce type names strictly to ease migration)
        if not isinstance(special_cfg, dict):
            special_cfg = None
        if not isinstance(special_noise, dict):
            special_noise = None

        total_steps = int(max(1, steps - max(0, int(skip_tail))))

        device = getattr(model, "load_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        base_noise = _prepare_noise(model, latent, seed).to(device)
        # global start noise scaler ALWAYS applied
        base_noise = base_noise * float(max(0.0, start_noise))

        # Single segment; optional UltraCFG taper (kept)
        base_cfg = float(cfg)
        taper_tail = 0
        if special_cfg:
            base_cfg = float(special_cfg.get("base_cfg", base_cfg))
            taper_tail = int(max(0, min(special_cfg.get("taper_tail", 0), total_steps)))
            rescale = float(special_cfg.get("rescale", 0.0))
        else:
            rescale = 0.0

        # Build one schedule
        if taper_tail > 0:
            early = max(1, total_steps - taper_tail)
            late = total_steps - early
            late_cfg = max(1.0, base_cfg * (1.0 - 0.5 * rescale))

            # A) Early
            sigA = _build_sigmas_for(model, scheduler, early, device)
            current = _run_with_sigmas(
                model, base_noise, sigA, sampler_name, float(base_cfg),
                positive, negative, latent, denoise, seed,
                during_plan=special_noise if (special_noise and special_noise.get("mode", "during") == "during") else None,
                base_noise_for_patterns=base_noise
            )
            # B) Late
            if late > 0:
                sigB = _build_sigmas_for(model, scheduler, late, device)
                # seam rejoin: start sigma equals end of A
                sigB[0] = sigA[-2].clone()
                current = _run_with_sigmas(
                    model, base_noise, sigB, sampler_name, float(late_cfg),
                    positive, negative, current, denoise, seed,
                    during_plan=special_noise if (special_noise and special_noise.get("mode", "during") == "during") else None,
                    base_noise_for_patterns=base_noise
                )
        else:
            sig = _build_sigmas_for(model, scheduler, total_steps, device)
            current = _run_with_sigmas(
                model, base_noise, sig, sampler_name, float(base_cfg),
                positive, negative, latent, denoise, seed,
                during_plan=special_noise if (special_noise and special_noise.get("mode", "during") == "during") else None,
                base_noise_for_patterns=base_noise
            )

        # POST refine
        if special_noise and special_noise.get("mode", "during") == "post":
            current = _apply_post(
                model, current, total_steps,
                scheduler, sampler_name, base_cfg,
                positive, negative, denoise, seed,
                device, base_noise, special_noise
            )

        return (current,)


NODE_CLASS_MAPPINGS = {"UltraDuperSampler": UltraDuperSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"UltraDuperSampler": "Ultra Duper Sampler (engine) v2"}