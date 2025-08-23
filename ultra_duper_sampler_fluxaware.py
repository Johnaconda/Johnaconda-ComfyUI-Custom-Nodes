# Ultra Duper Sampler — v2.8.1
# Flux-aware + Efficiency XY + HiRes-Fix + IMAGE & VAE outputs
#
# - v2.3: purge_vram + TF32 toggle + tiny grid cache
# - v2.5: dynamic sweep scaling, step weighting, visible start_noise kick, arrays-aware sweeps
# - v2.6: accepts Efficiency XY-Plot "script" and runs X×Y sweeps
# - v2.7: accepts Efficiency HiRes-Fix script (dict/tuple) and runs 2nd-pass upscale+refine
# - v2.8: optional IMAGE decode + VAE passthrough outputs (latent, image, vae)
# - v2.8.1: **SCRIPT** input for Efficiency compatibility + **xy_script** fallback
#
# ComfyUI ≥ 0.3.5x compatible.

import math
import random
import gc
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview


# -------------------------- utils & housekeeping --------------------------

def _std_normalize(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    s = t.std()
    return t / (s + eps)

def _fix_latent(model, latent_tensor: torch.Tensor) -> torch.Tensor:
    return comfy.sample.fix_empty_latent_channels(model, latent_tensor)

def _prepare_noise(model, latent: Dict, seed: int) -> torch.Tensor:
    lat = _fix_latent(model, latent["samples"])
    batch_inds = latent.get("batch_index", None) if isinstance(latent, dict) else None
    return comfy.sample.prepare_noise(lat, seed, batch_inds)

def _is_flux_model(model) -> bool:
    cand = [
        getattr(model, "model_type", None),
        getattr(getattr(model, "model", None), "model_type", None),
        getattr(getattr(model, "inner_model", None), "model_type", None),
        getattr(model, "arch", None),
        getattr(getattr(model, "model", None), "arch", None),
    ]
    s = " ".join([str(x).lower() for x in cand if x is not None])
    return any(k in s for k in ["flux", "flowmatch", "flow-match", "flow match", "flow"])

def _purge_vram(sync: bool = True):
    try:
        gc.collect()
        if torch.cuda.is_available():
            if sync:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception as e:
        print(f"[UltraDuperSampler] purge_vram failed: {e}")

# -------------------------- tiny grid cache (wave/checker) --------------------------

_GRID_CACHE = OrderedDict()  # key: (kind, h, w, device_str)
_GRID_CACHE_MAX = 16

def _grid_get(kind: str, h: int, w: int, device: torch.device):
    key = (kind, int(h), int(w), str(device))
    if key in _GRID_CACHE:
        _GRID_CACHE.move_to_end(key)
        return _GRID_CACHE[key]
    if kind == "linspace":
        yy = torch.linspace(0, math.pi * 2, steps=h, device=device).view(1, 1, h, 1)
        xx = torch.linspace(0, math.pi * 2, steps=w, device=device).view(1, 1, 1, w)
        val = (yy, xx)
    elif kind == "arange":
        yy = torch.arange(h, device=device).view(1, 1, h, 1)
        xx = torch.arange(w, device=device).view(1, 1, 1, w)
        val = (yy, xx)
    else:
        raise ValueError("unknown grid kind")
    _GRID_CACHE[key] = val
    if len(_GRID_CACHE) > _GRID_CACHE_MAX:
        _GRID_CACHE.popitem(last=False)
    return val

def _grid_cache_clear():
    _GRID_CACHE.clear()

# -------------------------- patterns & grain --------------------------

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
        yy, xx = _grid_get("arange", h, w, device)
        mask = ((yy // 8 + xx // 8) % 2) * 2 - 1
        n = mask.to(dtype).repeat(b, 1, 1, 1)
    elif pattern == "wave":
        yy, xx = _grid_get("linspace", h, w, device)
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
        yy, xx = _grid_get("linspace", h, w, inj.device)
        wave = (torch.sin(xx * 2.0) + torch.cos(yy * 3.0)).to(inj.dtype)
        return _std_normalize(inj + 0.25 * wave)
    if level == "chaotic":
        return _std_normalize(inj + torch.randn_like(inj) * 0.25)
    return inj

# -------------------------- CFG-Focus curve --------------------------

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

# -------------------------- POST short-tail refine --------------------------

def _compose_noise(base_noise: torch.Tensor, injected: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    if alpha <= 0.0:
        return base_noise
    bn = _std_normalize(base_noise)
    inj = _std_normalize(injected)
    composed = _std_normalize(bn + inj * float(alpha))
    return composed

def _post_refine(
    model, current_latent: Dict, steps: int, cfg_like: float,
    sampler_name: str, scheduler: str, positive, negative,
    denoise: float, seed: int, pattern: str, experimental: bool,
    base_noise: torch.Tensor, tail_steps: Optional[int] = None
) -> Dict:
    tail_steps = int(tail_steps if tail_steps is not None else max(1, steps // 4))
    device = getattr(model, "load_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    latent_img = _fix_latent(model, current_latent["samples"])
    base_cb = latent_preview.prepare_callback(model, tail_steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    inj = _pattern(tuple(base_noise.shape), device, base_noise.dtype, pattern, experimental)
    inj = _apply_grain(inj, "mid")
    composed = _compose_noise(base_noise, inj, alpha=1.0)

    ks = comfy.samplers.KSampler(
        model, steps=tail_steps, device=device,
        sampler=sampler_name, scheduler=scheduler, denoise=float(denoise),
        model_options=getattr(model, "model_options", {}),
    )
    out = ks.sample(
        noise=composed, positive=positive, negative=negative, cfg=float(cfg_like),
        latent_image=latent_img, start_step=0, last_step=int(tail_steps),
        force_full_denoise=False, denoise_mask=None, sigmas=None,
        callback=base_cb, disable_pbar=disable_pbar, seed=int(seed),
    )
    ret = current_latent.copy()
    ret["samples"] = out
    del inj, composed, latent_img, ks, out
    return ret

# -------------------------- XY helpers (Efficiency compatibility) --------------------------

_XY_SUPPORTED = {"CFG Scale", "Denoise", "Steps", "Scheduler", "Sampler", "Seed"}

def _xy_expand_axis(axis_type: str, axis_values):
    if axis_values is None:
        return []
    if axis_type in ("CFG Scale", "Denoise"):
        return [float(v) for v in axis_values]
    if axis_type in ("Steps", "Seed"):
        return [int(v) for v in axis_values]
    if axis_type == "Scheduler":
        return [str(v) for v in axis_values]
    if axis_type == "Sampler":
        return [(str(v[0]), str(v[1])) for v in axis_values]
    return []

def _xy_iter_pairs(X_type, X_vals, Y_type, Y_vals, flip_xy: bool):
    X = list(enumerate(_xy_expand_axis(X_type, X_vals)))
    Y = list(enumerate(_xy_expand_axis(Y_type, Y_vals)))
    if not X: X = [(0, None)]
    if not Y: Y = [(0, None)]
    if flip_xy:
        for yi, yv in Y:
            for xi, xv in X:
                yield (xi, xv, yi, yv)
    else:
        for xi, xv in X:
            for yi, yv in Y:
                yield (xi, xv, yi, yv)

def _xy_apply_to_params(params: dict, axis_type: str, axis_value):
    if axis_type not in _XY_SUPPORTED or axis_value is None:
        return
    if axis_type == "CFG Scale":
        params["cfg"] = float(axis_value)
    elif axis_type == "Denoise":
        params["denoise"] = float(axis_value)
    elif axis_type == "Steps":
        params["steps"] = int(axis_value)
    elif axis_type == "Seed":
        params["seed"] = int(axis_value)
    elif axis_type == "Scheduler":
        params["scheduler"] = str(axis_value)
    elif axis_type == "Sampler":
        sampler, scheduler = axis_value
        params["sampler_name"] = str(sampler)
        params["scheduler"] = str(scheduler)

# -------------------------- HiRes-Fix helpers (Efficiency compatibility) --------------------------

def _looks_like_hires_dict(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    keys = set(k.lower() for k in x.keys())
    return ("upscale_type" in keys) or ("upscale_by" in keys) or ("hires_steps" in keys)

def _parse_hires_from_tuple(t: tuple) -> Optional[dict]:
    if not isinstance(t, tuple) or len(t) < 2:
        return None
    tag = str(t[0]).lower()
    if "hires" in tag and isinstance(t[1], dict):
        return t[1]
    return None

def _parse_scripts(script: Any):
    """
    Accepts:
      - a single script object
      - a chain (list/tuple) of script objects
      - wrapped scripts: ("XY Plot", <xy_tuple>) or ("HighRes-Fix Script", <dict>)
      - raw XY tuples (len >= 10; see Efficiency XY)
      - raw HiRes dicts

    Returns: (xy_tuple_or_None, hires_dict_or_None)
    """
    XY_NAMES = {"xy plot", "xyplot", "xy-plot"}
    HIRES_NAMES = {"hires-fix", "highres-fix", "hires fix", "highres fix"}

    def is_raw_xy_tuple(x):
        if isinstance(x, tuple) and len(x) >= 10:
            # Heuristic: X_type and Y_type should be strings
            return isinstance(x[0], str) and isinstance(x[2], str)
        return False

    def unwrap_wrapped(x):
        """
        ("XY Plot", <xy_tuple>) -> ("xy", xy_tuple)
        ("HighRes-Fix Script", <dict>) -> ("hires", dict)
        Otherwise returns (None, None)
        """
        if isinstance(x, tuple) and len(x) >= 2 and isinstance(x[0], str):
            tag = x[0].strip().lower()
            payload = x[1]
            if any(k in tag for k in XY_NAMES):
                return ("xy", payload)
            if any(k in tag for k in HIRES_NAMES):
                return ("hires", payload)
        return (None, None)

    def is_hires_dict(x):
        if not isinstance(x, dict):
            return False
        keys = {k.lower() for k in x.keys()}
        # Be generous with aliases Efficiency nodes have used across versions
        aliases = {"upscale_type", "upscale_by", "scale", "factor", "multiplier",
                   "hires_steps", "steps", "denoise", "use_same_seed", "seed",
                   "iterations", "pixel_upscaler", "latent_upscaler"}
        return bool(keys & aliases)

    xy = None
    hires = None

    def visit(obj):
        nonlocal xy, hires
        if obj is None:
            return
        # wrapped case
        tag, payload = unwrap_wrapped(obj)
        if tag == "xy":
            if is_raw_xy_tuple(payload):
                xy = payload
            elif isinstance(payload, (list, tuple)) and is_raw_xy_tuple(payload[0]):
                xy = payload[0]
        elif tag == "hires":
            if isinstance(payload, dict) and is_hires_dict(payload):
                hires = payload
            elif isinstance(payload, (list, tuple)):
                for p in payload:
                    if isinstance(p, dict) and is_hires_dict(p):
                        hires = p
                        break
            return  # done with wrapped case

        # raw xy
        if xy is None and is_raw_xy_tuple(obj):
            xy = obj
            return

        # raw hires dict
        if hires is None and is_hires_dict(obj):
            hires = obj
            return

        # chain: list/tuple of possibly mixed items
        if isinstance(obj, (list, tuple)):
            for it in obj:
                visit(it)

    visit(script)
    return (xy, hires)

def _round_to_multiple(v: int, m: int) -> int:
    return int((v + m - 1) // m) * m

def _latent_size_from_scale(latent: torch.Tensor, scale: float) -> Tuple[int, int]:
    _, _, h, w = latent.shape
    nh = max(1, int(round(h * float(scale))))
    nw = max(1, int(round(w * float(scale))))
    return nh, nw

def _pixel_size_from_scale(latent: torch.Tensor, vae, scale: float) -> Tuple[int, int]:
    _, _, h, w = latent.shape
    ph = int(round(h * 8 * float(scale)))
    pw = int(round(w * 8 * float(scale)))
    ph = _round_to_multiple(ph, 8)
    pw = _round_to_multiple(pw, 8)
    return ph, pw

def _latent_upscale(latent: torch.Tensor, nh: int, nw: int) -> torch.Tensor:
    return F.interpolate(latent, size=(nh, nw), mode="nearest-exact")

def _pixel_upscale(img: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    img_nchw = img.movedim(-1, 1)
    up = F.interpolate(img_nchw, size=(ph, pw), mode="bicubic", align_corners=False)
    return up.movedim(1, -1).clamp(0, 1)

def _apply_hires(model, vae, positive, negative, cfg_like: float,
                 sampler_name: str, scheduler: str, base_denoise: float,
                 latent_dict: Dict, seed: int, hires_conf: dict) -> Dict:
    """
    Runs hires upscaling (latent/pixel/both) then a short refine pass.
    Accepts multiple key aliases like scale/factor/multiplier, steps/hires_steps, etc.
    """
    if not isinstance(hires_conf, dict):
        return latent_dict

    # --- normalize incoming config (support multiple key names) ---
    def pick(d, *names, default=None, cast=lambda x: x):
        for n in names:
            # exact key
            if n in d:
                try:
                    return cast(d[n])
                except Exception:
                    return default
            # lowercase alias
            ln = n.lower()
            if ln in d:
                try:
                    return cast(d[ln])
                except Exception:
                    return default
        return default

    upscale_type = str(pick(hires_conf, "upscale_type", default="latent")).lower()
    if upscale_type not in ("latent", "pixel", "both"):
        upscale_type = "latent"

    # scale: allow aliases; enforce sane value
    scale = pick(hires_conf, "upscale_by", "scale", "factor", "multiplier",
                 default=1.5, cast=float)
    try:
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.5
    except Exception:
        scale = 1.5

    hires_steps = max(1, int(pick(hires_conf, "hires_steps", "steps", default=12, cast=int)))
    denoise = float(pick(hires_conf, "denoise", default=base_denoise, cast=float))

    # booleans can come as "true"/"1"/1 etc.
    def to_bool(v):
        if isinstance(v, bool): return v
        if isinstance(v, (int, float)): return v != 0
        if isinstance(v, str): return v.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(v)

    use_same_seed = to_bool(pick(hires_conf, "use_same_seed", default=True, cast=to_bool))
    seed2 = int(pick(hires_conf, "seed", default=(seed if use_same_seed else (seed + 1)), cast=int))
    iterations = max(1, int(pick(hires_conf, "iterations", default=1, cast=int)))

    # FYI: external upscaler names are ignored (safe fallback)
    if any(k in hires_conf for k in ("pixel_upscaler", "latent_upscaler")):
        print("[UltraDuperSampler][HiRes] External upscaler requested; falling back to built-in interpolate/VAE path.")

    print(f"[UltraDuperSampler][HiRes] type={upscale_type} scale={scale} steps={hires_steps} denoise={denoise} its={iterations} seed={seed2}")

    device = getattr(model, "load_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    lat = _fix_latent(model, latent_dict["samples"])

    for it in range(iterations):
        # 1) Pixel upscaling (decode -> resize -> encode) if requested
        if "pixel" in upscale_type:
            if vae is None:
                print("[UltraDuperSampler][HiRes] pixel mode requested but no VAE connected; falling back to latent-only.")
            else:
                ph, pw = _pixel_size_from_scale(lat, vae, scale)
                try:
                    dec = vae.decode(lat)
                    img = dec["images"] if isinstance(dec, dict) else dec
                except Exception as e:
                    print(f"[UltraDuperSampler][HiRes] VAE decode failed: {e}")
                    img = None
                if isinstance(img, torch.Tensor):
                    img = img.clamp(0, 1)
                    up = _pixel_upscale(img, ph, pw)
                    try:
                        enc = vae.encode(up)
                        lat = enc["samples"] if isinstance(enc, dict) else enc
                    except Exception as e:
                        print(f"[UltraDuperSampler][HiRes] VAE encode failed: {e}")
                else:
                    print("[UltraDuperSampler][HiRes] VAE decode did not return tensor; skipping pixel stage.")

        # 2) Latent upscaling if requested (or as the only mode)
        if "latent" in upscale_type:
            nh, nw = _latent_size_from_scale(lat, scale)
            lat = _latent_upscale(lat, nh, nw)

        # 3) Short refine pass at high-res
        base_cb = latent_preview.prepare_callback(model, hires_steps)
        ks = comfy.samplers.KSampler(
            model, steps=int(hires_steps), device=device,
            sampler=sampler_name, scheduler=scheduler, denoise=float(denoise),
            model_options=getattr(model, "model_options", {}),
        )
        lat = ks.sample(
            noise=None, positive=positive, negative=negative, cfg=float(cfg_like),
            latent_image=lat, start_step=0, last_step=int(hires_steps),
            force_full_denoise=False, denoise_mask=None, sigmas=None,
            callback=base_cb, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=int(seed2 + it),
        )

    out = dict(latent_dict)
    out["samples"] = lat
    return out


# -------------------------- main engine --------------------------

class UltraDuperSampler:
    """
    Ultra Duper Sampler (Flux-aware v2.8.1)
    - purge_vram + interrupt-safe purge
    - allow_tf32 toggle (Ampere+)
    - dynamic sweep scaling + step weighting + start_noise kick
    - arrays-aware sweeps (strengths/grains)
    - XY-Plot compatible (Efficiency SCRIPT / XY fallback)
    - HiRes-Fix compatible (dict/tuple), including stacked with XY
    - NEW: emits (LATENT, IMAGE, VAE) — image decode is opt-in (emit_image)
    """
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
                "purge_vram": ("BOOLEAN", {"default": False, "tooltip": "Free VRAM after the run. On interrupt, purge is always performed."}),
                "allow_tf32": ("BOOLEAN", {"default": False, "tooltip": "Enable TF32 on matmul/cudnn (Ampere+). Can speed up FP32 models."}),
            },
            "optional": {
                "vae": ("VAE", {}),              # required for emit_image & pixel upscaling
                "special_cfg": ("DICT", {}),     # UltraCFG dict
                "special_noise": ("DICT", {}),   # Ultra Noise Sweeps dict
                # Efficiency compatibility:
                "script": ("SCRIPT", {}),        # primary (matches Efficiency output type)
                "xy_script": ("XY", {}),         # fallback for old graphs (raw XY tuple)
                "xy_preview_grid": ("BOOLEAN", {"default": False, "tooltip": "If VAE is connected, show a simple XY grid preview in the UI."}),
                "emit_image": ("BOOLEAN", {"default": True, "tooltip": "Decode and return IMAGE as second output (requires VAE)."}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "VAE")
    RETURN_NAMES = ("latent", "image", "vae")
    FUNCTION = "run"
    CATEGORY = "sampling/ultra"
    OUTPUT_NODE = False

    @torch.no_grad()
    def run(
        self, model, positive, negative, latent, steps, cfg,
        sampler_name, scheduler, denoise, seed, start_noise, skip_tail, auto_flux_detect,
        purge_vram, allow_tf32,
        special_cfg=None, special_noise=None,
        script=None, xy_script=None, vae=None, xy_preview_grid=False, emit_image=True
    ):
        steps = int(steps)
        skip_tail = max(0, min(int(skip_tail), max(0, steps - 1)))
        total_steps = max(1, steps - skip_tail)

        device = getattr(model, "load_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        is_flux = bool(auto_flux_detect) and _is_flux_model(model)

        latent_img0 = _fix_latent(model, latent["samples"])
        base_noise0 = _prepare_noise(model, {"samples": latent_img0, "batch_index": latent.get("batch_index")}, seed).to(device)
        base_noise0 = base_noise0 * float(start_noise)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # ---------- Prepare sweeps ----------
        sweep = special_noise if isinstance(special_noise, dict) else None
        base_pattern = "gaussian"
        post_tail_steps = None
        impact = 1.0
        step_weighting = "flat"
        dynamic_scale = True
        sweep_map_template = {}

        if sweep:
            base_pattern = str(sweep.get("noise_pattern", "gaussian"))
            post_tail_steps = sweep.get("post_tail_steps", None)
            strengths_arr = sweep.get("strengths")
            grains_arr = sweep.get("grains")
            start_step = int(max(0, sweep.get("start_step", 0)))
            n = int(max(1, sweep.get("num_sweeps", len(strengths_arr) if isinstance(strengths_arr, list) else 1)))
            gap = int(max(1, sweep.get("gap_steps", 1)))

            if not isinstance(strengths_arr, list):
                first = float(sweep.get("first_strength", 0.0))
                auto_dec = bool(sweep.get("auto_decrease", True))
                strengths_arr = [max(0.0, first - k * (first / n)) for k in range(n)] if auto_dec and n > 1 else [max(0.0, first)] * n

            if not isinstance(grains_arr, list):
                profile = str(sweep.get("grain_profile", sweep.get("grain_scheme", "balanced")))
                tmp = []
                for i in range(n):
                    if profile == "balanced":
                        tmp.append("fine" if i % 2 == 0 else "mid")
                    elif profile in ("coarse_to_fine", "coarse→fine"):
                        tmp.append("coarse" if i < n // 2 else "fine")
                    elif profile in ("fine_to_coarse", "fine→coarse"):
                        tmp.append("fine" if i < n // 2 else "coarse")
                    elif profile == "textured":
                        tmp.append("structured" if i % 2 == 0 else "mid")
                    elif profile == "wild":
                        tmp.append("chaotic" if i % 2 == 1 else "fine")
                    else:
                        tmp.append("mid")
                grains_arr = tmp

            m = min(len(strengths_arr), len(grains_arr), n)
            triggers = [min(total_steps - 1, max(0, start_step + k * gap)) for k in range(m)]
            sweep_map_template = {triggers[i]: (float(strengths_arr[i]), str(grains_arr[i])) for i in range(m)}
            impact = float(sweep.get("impact", 1.0))
            step_weighting = str(sweep.get("step_weighting", "flat"))
            dynamic_scale = bool(sweep.get("dynamic_scale", True))

        # ---------- CFG-Focus schedule ----------
        cf = special_cfg if isinstance(special_cfg, dict) else {}
        base_cfg = float(cf.get("base_cfg", cfg if isinstance(cfg, (float, int)) else 7.0))
        focus_mode = str(cf.get("mode", "none"))
        focus_strength = float(cf.get("focus_strength", 0.0))
        focus_start = int(cf.get("focus_start", 0))
        focus_end = int(cf.get("focus_end", total_steps))
        focus_curve_template = _cfg_focus_curve(focus_mode, total_steps, focus_start, focus_end, focus_strength)

        # ---------- parse scripts (XY + HiRes) ----------
        script_obj = script if script is not None else xy_script
        xy_tuple, hires_conf = _parse_scripts(script_obj)

        # ---------- helpers ----------
        def _step_weight(i: int, total: int, mode: str) -> float:
            total = max(1, int(total))
            if total <= 1: return 1.0
            t = i / (total - 1)
            m = (mode or "flat").lower()
            if m == "flat":     return 1.0
            if m == "early":    return 1.0 - t
            if m == "late":     return t
            if m == "mid":      return 1.0 - abs(2.0 * t - 1.0)
            if m == "edge":     return abs(2.0 * t - 1.0)
            if m == "gaussian":
                mu, sigma = 0.5, 0.20
                return math.exp(-0.5 * ((t - mu) / sigma) ** 2)
            return 1.0

        def _run_single(params: dict):
            _steps = int(params.get("steps", total_steps))
            _cfg_v = float(params.get("cfg", base_cfg))
            _sampler = str(params.get("sampler_name", sampler_name))
            _sched = str(params.get("scheduler", scheduler))
            _denoise = float(params.get("denoise", denoise))
            _seed = int(params.get("seed", seed))

            latent_img = _fix_latent(model, latent["samples"])
            base_noise = _prepare_noise(model, {"samples": latent_img, "batch_index": latent.get("batch_index")}, _seed).to(device)
            base_noise = base_noise * float(start_noise)
            sweep_std = float(base_noise.std()) + 1e-6
            focus_curve = list(focus_curve_template)
            base_cb = latent_preview.prepare_callback(model, _steps)

            def callback(i, denoised, x, steps_all):
                if base_cb: base_cb(i, denoised, x, steps_all)
                if i == 0 and abs(float(start_noise) - 1.0) > 1e-6:
                    sn_gain = float(start_noise) - 1.0
                    cur_std = x.std() + 1e-6
                    extra = _std_normalize(base_noise) * cur_std * (sn_gain * 0.5)
                    x.add_(extra); del extra
                if sweep_map_template and i in sweep_map_template:
                    strength, grain = sweep_map_template[i]
                    cur_std = (x.std() + 1e-6) if dynamic_scale else sweep_std
                    if strength > 0.0 and cur_std > 0.0:
                        inj = _pattern(tuple(x.shape), x.device, x.dtype, base_pattern, bool(sweep.get("experimental", False)) if sweep else False)
                        inj = _apply_grain(inj, grain)
                        inj = _std_normalize(inj) * cur_std
                        gain = float(strength) * float(impact) * _step_weight(i, _steps, step_weighting)
                        if gain != 0.0: x.add_(inj * gain)
                        del inj
                alpha = float(focus_curve[i]) if 0 <= i < len(focus_curve) else 0.0
                if alpha > 0.0:
                    x.add_((denoised - x) * alpha)

            ks = comfy.samplers.KSampler(
                model, steps=int(_steps), device=device,
                sampler=_sampler, scheduler=_sched, denoise=float(_denoise),
                model_options=getattr(model, "model_options", {}),
            )
            out_samples = ks.sample(
                noise=base_noise, positive=positive, negative=negative, cfg=float(_cfg_v),
                latent_image=latent_img, start_step=0, last_step=int(_steps),
                force_full_denoise=False, denoise_mask=None, sigmas=None,
                callback=callback, disable_pbar=disable_pbar, seed=int(_seed),
            )
            out_dict = dict(latent)
            out_dict["samples"] = out_samples

            if sweep and sweep.get("mode", "during") != "during":
                out_dict = _post_refine(
                    model=model, current_latent=out_dict, steps=_steps, cfg_like=_cfg_v,
                    sampler_name=_sampler, scheduler=_sched, positive=positive, negative=negative,
                    denoise=_denoise, seed=_seed, pattern=base_pattern, experimental=bool(sweep.get("experimental", False)),
                    base_noise=base_noise, tail_steps=post_tail_steps,
                )

            if isinstance(hires_conf, dict) or (isinstance(hires_conf, tuple) and _parse_hires_from_tuple(hires_conf)):
                print("[UltraDuperSampler] HiRes script detected; applying second pass.")
                hc = hires_conf if isinstance(hires_conf, dict) else _parse_hires_from_tuple(hires_conf)
                out_dict = _apply_hires(
                    model=model, vae=vae, positive=positive, negative=negative, cfg_like=_cfg_v,
                    sampler_name=_sampler, scheduler=_sched, base_denoise=_denoise,
                    latent_dict=out_dict, seed=_seed, hires_conf=hc
    )

            return out_dict["samples"]

        interrupted = False
        prev_tf32_matmul = None
        prev_tf32_cudnn = None
        ui_images = []

        try:
            if allow_tf32 and torch.cuda.is_available():
                prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
                prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            def _decode_images(latent_tensor):
                if not bool(emit_image):
                    return None
                if vae is None:
                    print("[UltraDuperSampler] emit_image=True but no VAE connected; skipping decode.")
                    return None
                try:
                    dec = vae.decode(latent_tensor)
                    images = dec["images"] if isinstance(dec, dict) else dec
                    if isinstance(images, torch.Tensor):
                        return images.clamp(0, 1)
                except Exception as e:
                    print(f"[UltraDuperSampler] VAE decode failed: {e}")
                return None

            if not (isinstance(xy_tuple, tuple) and len(xy_tuple) >= 10):
                params = {
                    "steps": total_steps, "cfg": base_cfg, "sampler_name": sampler_name,
                    "scheduler": scheduler, "denoise": denoise, "seed": seed
                }
                out_samples = _run_single(params)
                out = dict(latent); out["samples"] = out_samples
                images = _decode_images(out_samples)

                if is_flux:
                    print("[UltraDuperSampler] Flux detected. Treating 'cfg' as guidance. Typical guidance 1.5–4.0; steps 12–24.")
                return (out, images, vae)

            # XY enabled
            X_type, X_val, Y_type, Y_val, grid_spacing, _y_orient, _cache_models, _xy_as_img, flip_xy, _deps = xy_tuple
            xy_grid_spacing = int(grid_spacing) if grid_spacing is not None else 0
            print("[UltraDuperSampler][XY] Running grid. Supported axis types:", _XY_SUPPORTED)

            batch_samples = []
            for (xi, xv, yi, yv) in _xy_iter_pairs(X_type, X_val, Y_type, Y_val, bool(flip_xy)):
                params = {
                    "steps": total_steps, "cfg": base_cfg, "sampler_name": sampler_name,
                    "scheduler": scheduler, "denoise": denoise, "seed": seed
                }
                _xy_apply_to_params(params, X_type, xv)
                _xy_apply_to_params(params, Y_type, yv)

                out_samples = _run_single(params)
                batch_samples.append(out_samples)

                # Optional UI preview grid (decode each cell)
                if vae is not None and bool(xy_preview_grid):
                    try:
                        img = vae.decode(out_samples)["images"] if isinstance(vae.decode(out_samples), dict) else vae.decode(out_samples)
                        if isinstance(img, torch.Tensor):
                            img = img.clamp(0, 1)
                            ui_images.append(img[0:1])
                    except Exception as e:
                        print(f"[UltraDuperSampler][XY] VAE preview failed: {e}")

            cat = torch.cat(batch_samples, dim=0) if len(batch_samples) > 1 else batch_samples[0]
            out = dict(latent); out["samples"] = cat
            images = _decode_images(cat)

            ui = {}
            if vae is not None and bool(xy_preview_grid) and ui_images:
                try:
                    imgs = torch.cat(ui_images, dim=0)  # (B,H,W,C)
                    B, H, W, C = imgs.shape
                    cols = len(_xy_expand_axis(X_type, X_val)) or 1
                    rows = len(_xy_expand_axis(Y_type, Y_val)) or 1
                    if bool(flip_xy): cols, rows = rows, cols
                    grid_h = rows * H + max(0, rows - 1) * int(xy_grid_spacing)
                    grid_w = cols * W + max(0, cols - 1) * int(xy_grid_spacing)
                    grid = torch.zeros((grid_h, grid_w, C), dtype=imgs.dtype, device=imgs.device)
                    k = 0
                    for r in range(rows):
                        for c in range(cols):
                            if k >= B: break
                            y0 = r * (H + xy_grid_spacing)
                            x0 = c * (W + xy_grid_spacing)
                            grid[y0:y0+H, x0:x0+W, :] = imgs[k]
                            k += 1
                    ui = {"images": [grid.unsqueeze(0)]}
                except Exception as e:
                    print(f"[UltraDuperSampler][XY] Failed to assemble UI grid: {e}")

            if is_flux:
                print("[UltraDuperSampler] Flux detected. Treating 'cfg' as guidance. Typical guidance 1.5–4.0; steps 12–24.")

            if ui:
                return {"ui": ui, "result": (out, images, vae)}
            else:
                return (out, images, vae)

        except BaseException:
            interrupted = True
            raise
        finally:
            if prev_tf32_matmul is not None:
                torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
            if prev_tf32_cudnn is not None:
                torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn

            for _name in ("base_noise0", "latent_img0"):
                if _name in locals():
                    try: del locals()[_name]
                    except Exception: pass

            if interrupted or bool(purge_vram):
                _grid_cache_clear()
                _purge_vram(sync=True)


NODE_CLASS_MAPPINGS = {"UltraDuperSampler": UltraDuperSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"UltraDuperSampler": "Ultra Duper Sampler (Flux-aware v2.8.1)"}
