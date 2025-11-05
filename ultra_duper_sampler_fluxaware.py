# Ultra Duper Sampler — v3.2 (Ultra-only + Supersample→Downscale)
# Fix: no align_corners for nearest/nearest-exact in interpolate (PyTorch error)
# Order: latent → pixel → refine, do_pixel/do_latent flags
# Pixel blur device/dtype-safe; refine pass always gets real noise
# Final Output controls: keep_original / hires_size / custom (+ optional latent downscale)

import math
import random
import gc
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
import folder_paths as paths


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


def _safe_cleanup(lat=None):
    del lat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


# -------------------------- tiny grid cache (used by noise patterns) --------------------------

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


# -------------------------- patterns & grain (Ultra Noise Sweeps) --------------------------

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


# -------------------------- Post refine (tail) --------------------------

def _compose_noise(bn: torch.Tensor, inj: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    if alpha <= 0.0:
        return bn
    bn = _std_normalize(bn)
    inj = _std_normalize(inj)
    return _std_normalize(bn + inj * float(alpha))

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


# -------------------------- Ultra-only script parsing --------------------------

def _parse_ultra_hires(script_obj):
    if script_obj is None:
        return None
    if isinstance(script_obj, tuple) and len(script_obj) >= 2 and isinstance(script_obj[0], str):
        tag = script_obj[0].strip().lower()
        if "ultra hires" in tag and isinstance(script_obj[1], dict):
            return script_obj[1]
    if isinstance(script_obj, dict):
        return script_obj
    return None


# -------------------------- Ultra HiRes core --------------------------

def _apply_hires(model, vae, positive, negative, cfg_like: float,
                 sampler_name: str, scheduler: str, base_denoise: float,
                 latent_dict: Dict, seed: int, hires_conf: dict) -> Dict:
    if not isinstance(hires_conf, dict):
        return latent_dict

    # ---------- helpers ----------
    def pick(d, *names, default=None, cast=lambda x: x):
        for n in names:
            if n in d:
                try: return cast(d[n])
                except Exception: return default
            ln = n.lower()
            if ln in d:
                try: return cast(d[ln])
                except Exception: return default
        return default

    def _round8(v: int) -> int:
        return max(8, int((v + 7) // 8) * 8)

    def _latent_up(lat, nh, nw, mode: str):
        mode = str(mode or "nearest-exact")
        if mode in ("nearest", "nearest-exact"):
            return F.interpolate(lat, size=(int(nh), int(nw)), mode="nearest-exact")
        elif mode in ("bilinear", "bicubic"):
            return F.interpolate(lat, size=(int(nh), int(nw)), mode=mode, align_corners=False)
        else:
            return F.interpolate(lat, size=(int(nh), int(nw)), mode="nearest-exact")

    def _pixel_up(img_bhwc, ph, pw, mode: str):
        mode = str(mode or "bicubic")
        nchw = img_bhwc.movedim(-1, 1)
        if mode == "nearest":
            up = F.interpolate(nchw, size=(int(ph), int(pw)), mode="nearest")
        elif mode in ("bilinear", "bicubic"):
            up = F.interpolate(nchw, size=(int(ph), int(pw)), mode=mode, align_corners=False)
        else:
            up = F.interpolate(nchw, size=(int(ph), int(pw)), mode="bicubic", align_corners=False)
        return up.movedim(1, -1).clamp(0, 1)

    # Device/dtype-safe blur & beautify
    def _gauss_kernel1d(sigma: float, device, dtype):
        sigma = max(0.2, float(sigma))
        k = int(max(3, round(sigma * 6)) // 2 * 2 + 1)
        c = (k - 1) / 2
        x = torch.arange(k, device=device, dtype=dtype) - c
        w = torch.exp(-(x**2) / (2 * sigma * sigma))
        w = w / w.sum()
        return w.view(1, 1, -1)

    def _gaussian_blur(img_bhwc, sigma: float):
        if sigma <= 0.0:
            return img_bhwc
        b, h, w, c = img_bhwc.shape
        dev = img_bhwc.device
        dt  = img_bhwc.dtype
        kx = _gauss_kernel1d(sigma, device=dev, dtype=dt)
        ky = _gauss_kernel1d(sigma, device=dev, dtype=dt)
        x = img_bhwc.movedim(-1, 1)
        x = F.pad(x, (kx.shape[-1]//2, kx.shape[-1]//2, 0, 0), mode="reflect")
        x = F.conv2d(x, kx.expand(c,1,1,kx.shape[-1]), groups=c)
        x = F.pad(x, (0, 0, ky.shape[-1]//2, ky.shape[-1]//2), mode="reflect")
        x = F.conv2d(x, ky.transpose(-1,-2).expand(c,1,ky.shape[-1],1), groups=c)
        return x.movedim(1, -1)

    def _unsharp(img_bhwc, amount: float, sigma: float):
        if amount <= 0.0:
            return img_bhwc
        blurred = _gaussian_blur(img_bhwc, sigma)
        return (img_bhwc + amount * (img_bhwc - blurred)).clamp(0, 1)

    def _sat_contrast(img_bhwc, sat: float, micro_c: float):
        if sat <= 0.0 and micro_c <= 0.0:
            return img_bhwc
        rgb = img_bhwc
        y = (rgb[...,0]*0.299 + rgb[...,1]*0.587 + rgb[...,2]*0.114).unsqueeze(-1)
        if sat > 0.0:
            rgb = (y + (rgb - y) * (1.0 + sat)).clamp(0, 1)
        if micro_c > 0.0:
            rgb = ((rgb - 0.5) * (1.0 + micro_c) + 0.5).clamp(0, 1)
        return rgb

    # ---------- read + normalize config ----------
    upscale_type = str(pick(hires_conf, "upscale_type", default="latent")).lower()
    if upscale_type not in ("latent", "pixel", "both"):
        upscale_type = "latent"
    do_pixel  = (upscale_type == "pixel"  or upscale_type == "both")
    do_latent = (upscale_type == "latent" or upscale_type == "both")

    latent_resampler = str(pick(hires_conf, "latent_resampler", default="nearest-exact"))
    pixel_resampler  = str(pick(hires_conf, "pixel_resampler", default="bicubic"))

    scale = pick(hires_conf, "upscale_by", "scale", "factor", "multiplier", default=1.5, cast=float)
    try:
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.5
    except Exception:
        scale = 1.5

    hires_steps = max(1, int(pick(hires_conf, "hires_steps", "steps", default=12, cast=int)))
    denoise     = float(pick(hires_conf, "denoise", default=base_denoise, cast=float))

    def _to_bool(v):
        if isinstance(v, bool): return v
        if isinstance(v, (int, float)): return v != 0
        if isinstance(v, str): return v.strip().lower() in ("1","true","yes","y","on")
        return bool(v)

    use_same    = _to_bool(pick(hires_conf, "use_same_seed", default=True, cast=_to_bool))
    seed2       = int(pick(hires_conf, "seed", default=(seed if use_same else (seed + 1)), cast=int))
    iterations  = max(1, int(pick(hires_conf, "iterations", default=1, cast=int)))

    # smart targets (pixels)
    tgt_long  = int(pick(hires_conf, "target_long_edge", default=0, cast=int))
    tgt_short = int(pick(hires_conf, "target_short_edge", default=0, cast=int))
    tgt_w     = int(pick(hires_conf, "target_width", default=0, cast=int))
    tgt_h     = int(pick(hires_conf, "target_height", default=0, cast=int))
    max_mp    = float(pick(hires_conf, "max_megapixels", default=0.0, cast=float))

    beauty    = hires_conf.get("beautify", {}) if isinstance(hires_conf.get("beautify", {}), dict) else {}
    b_enable  = bool(beauty.get("enable", True))
    b_sharp   = float(beauty.get("sharpen_amount", 0.15))
    b_sigma   = float(beauty.get("sharpen_sigma", 0.8))
    b_mic     = float(beauty.get("micro_contrast", 0.08))
    b_sat     = float(beauty.get("saturation", 0.06))

    device = getattr(model, "load_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    lat = _fix_latent(model, latent_dict["samples"])  # (B,C,H,W)

    # ---------- compute target sizes ----------
    _, _, lh, lw = lat.shape
    base_ph, base_pw = lh * 8, lw * 8

    ph, pw = None, None
    if tgt_w > 0 and tgt_h > 0:
        pw, ph = tgt_w, tgt_h
    elif tgt_w > 0:
        pw = tgt_w; ph = int(round(pw * (base_ph / base_pw)))
    elif tgt_h > 0:
        ph = tgt_h; pw = int(round(ph * (base_pw / base_ph)))
    elif tgt_long > 0 or tgt_short > 0:
        long_is_w = base_pw >= base_ph
        if tgt_long > 0:
            if long_is_w:
                pw = tgt_long; ph = int(round(pw * (base_ph / base_pw)))
            else:
                ph = tgt_long; pw = int(round(ph * (base_pw / base_ph)))
        else:
            if long_is_w:
                ph = tgt_short; pw = int(round(ph * (base_pw / base_ph)))
            else:
                pw = tgt_short; ph = int(round(pw * (base_ph / base_pw)))

    if ph is None or pw is None:
        ph = int(round(base_ph * scale))
        pw = int(round(base_pw * scale))

    ph = max(8, (ph + 7) // 8 * 8)
    pw = max(8, (pw + 7) // 8 * 8)

    if max_mp > 0.0:
        pix = ph * pw
        cap = int(max_mp * 1_000_000)
        if pix > cap > 0:
            r = math.sqrt(cap / pix)
            ph = max(8, int(round(ph * r)) // 8 * 8)
            pw = max(8, int(round(pw * r)) // 8 * 8)

    nh, nw = int(ph // 8), int(pw // 8)

    print(f"[UltraDuperSampler][HiRes] type={upscale_type} -> {pw}x{ph}px ({nw}x{nh} latent), steps={hires_steps}, denoise={denoise}, its={iterations}, seed={seed2}")

    # ---------- run iterations ----------
    for it in range(iterations):

        # 1) Latent up first
        if do_latent:
            lat = _latent_up(lat, nh, nw, latent_resampler)

        # 2) Pixel path: decode → maybe resize → beautify → re-encode
        if do_pixel:
            if vae is None:
                print("[UltraDuperSampler][HiRes] pixel mode requested but no VAE connected; falling back to latent-only.")
            else:
                try:
                    dec = vae.decode(lat)
                    img = dec["images"] if isinstance(dec, dict) else dec
                except Exception as e:
                    print(f"[UltraDuperSampler][HiRes] VAE decode failed: {e}")
                    img = None

                if isinstance(img, torch.Tensor):
                    img = img.clamp(0, 1)
                    _, h, w, _ = img.shape
                    if h != ph or w != pw:
                        img = _pixel_up(img, ph, pw, pixel_resampler)

                    if b_enable and (b_sharp > 0 or b_mic > 0 or b_sat > 0):
                        img = _unsharp(img, b_sharp, b_sigma)
                        img = _sat_contrast(img, b_sat, b_mic)

                    try:
                        enc = vae.encode(img)
                        lat = enc["samples"] if isinstance(enc, dict) else enc
                    except Exception as e:
                        print(f"[UltraDuperSampler][HiRes] VAE encode failed: {e}")
                else:
                    print("[UltraDuperSampler][HiRes] VAE decode returned non-tensor; skipping pixel stage.")

        # 3) Short refine at high-res (must pass real noise on 0.3.50)
        base_cb = latent_preview.prepare_callback(model, hires_steps)
        ks = comfy.samplers.KSampler(
            model, steps=int(hires_steps), device=device,
            sampler=sampler_name, scheduler=scheduler, denoise=float(denoise),
            model_options=getattr(model, "model_options", {}),
        )
        try:
            ref_noise = _prepare_noise(model, {"samples": lat, "batch_index": None}, int(seed2 + it))
        except Exception as e:
            print(f"[UltraDuperSampler][HiRes] prepare_noise failed: {e}; using randn_like.")
            ref_noise = None
        if ref_noise is None:
            ref_noise = torch.randn_like(lat, device=lat.device)

        lat = ks.sample(
            noise=ref_noise,
            positive=positive, negative=negative, cfg=float(cfg_like),
            latent_image=lat, start_step=0, last_step=int(hires_steps),
            force_full_denoise=False, denoise_mask=None, sigmas=None,
            callback=base_cb, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=int(seed2 + it),
        )

    out = dict(latent_dict)
    out["samples"] = lat
    try:
        return out
    finally:
        _safe_cleanup(lat)

# -------------------------- main engine --------------------------

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
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.05}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "start_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "skip_tail": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "auto_flux_detect": ("BOOLEAN", {"default": True}),
                "purge_vram": ("BOOLEAN", {"default": False}),
                "allow_tf32": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "vae": ("VAE", {}),
                "special_cfg": ("DICT", {}),
                "special_noise": ("DICT", {}),
                "script": ("SCRIPT", {"tooltip": "Ultra HiRes Script output (or plain dict in same schema)."}),
                "emit_image": ("BOOLEAN", {"default": True}),
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
        script=None, vae=None, emit_image=True
    ):
        steps = int(steps)
        skip_tail = max(0, min(int(skip_tail), max(0, steps - 1)))
        total_steps = max(1, steps - skip_tail)

        device = getattr(model, "load_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        is_flux = bool(auto_flux_detect) and _is_flux_model(model)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # original sizes for final-output keep_original
        orig_lh, orig_lw = int(latent["samples"].shape[-2]), int(latent["samples"].shape[-1])
        orig_ph, orig_pw = orig_lh * 8, orig_lw * 8

        # ---------- sweeps ----------
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

        # ---------- CFG-Focus ----------
        cf = special_cfg if isinstance(special_cfg, dict) else {}
        base_cfg = float(cf.get("base_cfg", cfg if isinstance(cfg, (float, int)) else 7.0))
        focus_mode = str(cf.get("mode", "none"))
        focus_strength = float(cf.get("focus_strength", 0.0))
        focus_start = int(cf.get("focus_start", 0))
        focus_end = int(cf.get("focus_end", total_steps))
        focus_curve_template = _cfg_focus_curve(focus_mode, total_steps, focus_start, focus_end, focus_strength)

        # ---------- HiRes (Ultra-only) ----------
        hires_conf = _parse_ultra_hires(script)
        final_conf = hires_conf.get("final", {}) if isinstance(hires_conf, dict) else {}
        final_mode = str(final_conf.get("mode", "keep_original")).lower()
        final_w = int(final_conf.get("width", 0))
        final_h = int(final_conf.get("height", 0))
        final_le = int(final_conf.get("long_edge", 0))
        final_resampler = str(final_conf.get("resampler", "bicubic"))
        final_antialias = bool(final_conf.get("antialias", True))
        final_downscale_latent = bool(final_conf.get("downscale_latent", False))

        def _final_target_for(out_ph, out_pw):
            if final_mode == "hires_size":
                return None
            if final_mode == "keep_original":
                return (orig_pw, orig_ph)
            if final_mode == "custom":
                if final_w > 0 and final_h > 0:
                    return (final_w, final_h)
                elif final_w > 0:
                    h = int(round(final_w * (out_ph / out_pw)))
                    return (final_w, h)
                elif final_h > 0:
                    w = int(round(final_h * (out_pw / out_ph)))
                    return (w, final_h)
                elif final_le > 0:
                    long_is_w = out_pw >= out_ph
                    if long_is_w:
                        w = final_le; h = int(round(w * (out_ph / out_pw)))
                    else:
                        h = final_le; w = int(round(h * (out_pw / out_ph)))
                    return (w, h)
            return None

        def _img_resize_bhwc(img, W, H, mode, antialias=True):
            nchw = img.movedim(-1, 1)
            if mode == "nearest":
                out = F.interpolate(nchw, size=(H, W), mode="nearest")
            else:
                out = F.interpolate(nchw, size=(H, W), mode=mode,
                                    align_corners=False, antialias=bool(antialias))
            return out.movedim(1, -1).clamp(0, 1)

        def _latent_resize(lat, nh, nw, mode: str):
            mode = str(mode or "nearest-exact")
            if mode in ("nearest", "nearest-exact"):
                return F.interpolate(lat, size=(int(nh), int(nw)), mode="nearest-exact")
            elif mode in ("bilinear", "bicubic"):
                return F.interpolate(lat, size=(int(nh), int(nw)), mode=mode, align_corners=False)
            else:
                return F.interpolate(lat, size=(int(nh), int(nw)), mode="nearest-exact")

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
                        x.add_(inj * float(strength) * float(impact))
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

            if isinstance(hires_conf, dict):
                print("[UltraDuperSampler] HiRes script detected; applying second pass.")
                out_dict = _apply_hires(
                    model=model, vae=vae, positive=positive, negative=negative, cfg_like=_cfg_v,
                    sampler_name=_sampler, scheduler=_sched, base_denoise=_denoise,
                    latent_dict=out_dict, seed=_seed, hires_conf=hires_conf
                )

            return out_dict["samples"]

        interrupted = False
        prev_tf32_matmul = None
        prev_tf32_cudnn = None

        try:
            if allow_tf32 and torch.cuda.is_available():
                prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
                prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            def _decode_images(latent_tensor, final_size=None, mode="bicubic", antialias=True):
                if not bool(emit_image):
                    return None
                if vae is None:
                    print("[UltraDuperSampler] emit_image=True but no VAE connected; skipping decode.")
                    return None
                try:
                    dec = vae.decode(latent_tensor)
                    images = dec["images"] if isinstance(dec, dict) else dec
                    if isinstance(images, torch.Tensor):
                        images = images.clamp(0, 1)
                        if isinstance(final_size, tuple) and final_size[0] > 0 and final_size[1] > 0:
                            W, H = int(final_size[0]), int(final_size[1])
                            if mode == "nearest":
                                images = images.movedim(-1, 1)
                                images = F.interpolate(images, size=(H, W), mode="nearest").movedim(1, -1)
                            else:
                                images = images.movedim(-1, 1)
                                images = F.interpolate(images, size=(H, W), mode=mode,
                                                       align_corners=False, antialias=bool(antialias)).movedim(1, -1)
                        return images
                except Exception as e:
                    print(f"[UltraDuperSampler] VAE decode failed: {e}")
                return None

            params = {
                "steps": total_steps, "cfg": base_cfg, "sampler_name": sampler_name,
                "scheduler": scheduler, "denoise": denoise, "seed": seed
            }
            out_samples = _run_single(params)

            # ---- Final Output sizing (supersample → downscale) ----
            out_lh, out_lw = int(out_samples.shape[-2]), int(out_samples.shape[-1])
            out_ph, out_pw = out_lh * 8, out_lw * 8

            hires_conf_resampler = "nearest-exact"
            if isinstance(hires_conf, dict):
                hires_conf_resampler = str(hires_conf.get("latent_resampler", "nearest-exact"))

            final_conf = hires_conf.get("final", {}) if isinstance(hires_conf, dict) else {}
            final_mode = str(final_conf.get("mode", "keep_original")).lower()
            final_w = int(final_conf.get("width", 0))
            final_h = int(final_conf.get("height", 0))
            final_le = int(final_conf.get("long_edge", 0))
            final_resampler = str(final_conf.get("resampler", "bicubic"))
            final_antialias = bool(final_conf.get("antialias", True))
            final_downscale_latent = bool(final_conf.get("downscale_latent", False))

            def _final_target_for(out_ph, out_pw):
                if final_mode == "hires_size":
                    return None
                if final_mode == "keep_original":
                    return (orig_pw, orig_ph)
                if final_mode == "custom":
                    if final_w > 0 and final_h > 0:
                        return (final_w, final_h)
                    elif final_w > 0:
                        h = int(round(final_w * (out_ph / out_pw)))
                        return (final_w, h)
                    elif final_h > 0:
                        w = int(round(final_h * (out_pw / out_ph)))
                        return (w, final_h)
                    elif final_le > 0:
                        long_is_w = out_pw >= out_ph
                        if long_is_w:
                            w = final_le; h = int(round(w * (out_ph / out_pw)))
                        else:
                            h = final_le; w = int(round(h * (out_pw / out_ph)))
                        return (w, h)
                return None

            final_target = _final_target_for(out_ph, out_pw)

            if final_target and bool(final_downscale_latent):
                W, H = final_target
                nh, nw = int(max(8, H) // 8), int(max(8, W) // 8)
                out_samples = _latent_resize(out_samples, nh, nw, hires_conf_resampler)

            images = _decode_images(out_samples, final_size=final_target, mode=final_resampler, antialias=final_antialias)

            try:
                print(f"[UltraDuperSampler] Latent before: {orig_lw}x{orig_lh}  after: {out_lw}x{out_lh}")
                exp_w, exp_h = int(out_lw * 8), int(out_lh * 8)
                print(f"[UltraDuperSampler] Expected decoded image (hires): {exp_w}x{exp_h}")
                if final_target:
                    print(f"[UltraDuperSampler] Final output size: {final_target[0]}x{final_target[1]}  (mode={final_mode})")
                if isinstance(images, torch.Tensor):
                    b, h, w, c = images.shape
                    print(f"[UltraDuperSampler] Decoded tensor: {w}x{h} (B={b}, C={c})")
                else:
                    print("[UltraDuperSampler] No decoded image tensor (emit_image off or VAE missing).")
            except Exception as e:
                print(f"[UltraDuperSampler] Debug size print failed: {e}")

            out = dict(latent); out["samples"] = out_samples

            if is_flux:
                print("[UltraDuperSampler] Flux detected. Treating 'cfg' as guidance. Typical guidance 1.5–4.0; steps 12–24.")

            return (out, images, vae)

        except BaseException:
            interrupted = True
            raise
        finally:
            if prev_tf32_matmul is not None:
                torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
            if prev_tf32_cudnn is not None:
                torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn
            if interrupted or bool(purge_vram):
                _grid_cache_clear()
                _purge_vram(sync=True)


# -------------------------- benchmark + sampler matrix helpers --------------------------

class UltraSamplerMatrix:
    """
    Define sampler/scheduler rows. Automatically use all rows defined.
    No manual count input.
    """

    MAX = 6

    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        for i in range(1, cls.MAX + 1):
            opt[f"enable_{i}"] = (
                "BOOLEAN",
                {
                    "default": True if i == 1 else False,
                    "tooltip": f"Enable row {i} in sampler matrix",
                },
            )
            opt[f"sampler_{i}"] = (comfy.samplers.KSampler.SAMPLERS, {})
            opt[f"scheduler_{i}"] = (comfy.samplers.KSampler.SCHEDULERS, {})
            opt[f"label_{i}"] = (
                "STRING",
                {"default": f"S{i}", "tooltip": "Optional label for this row"},
            )

        return {"required": {}, "optional": opt}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("sampler_matrix",)
    FUNCTION = "build"
    CATEGORY = "sampling/ultra/tweaks"
    OUTPUT_NODE = False

    def build(self, **kwargs):
        pairs = []
        for i in range(1, self.MAX + 1):
            if not bool(kwargs.get(f"enable_{i}", False)):
                continue
            s = kwargs.get(f"sampler_{i}")
            sch = kwargs.get(f"scheduler_{i}")
            lab = kwargs.get(f"label_{i}", f"S{i}")
            if s is None or sch is None:
                continue
            pairs.append({"sampler": str(s), "scheduler": str(sch), "label": str(lab)})

        # Fallback to one row if none defined
        if not pairs:
            # Use first row defaults or provided sampler_name & scheduler
            pairs.append({"sampler": str(kwargs.get("sampler_1", "")),
                          "scheduler": str(kwargs.get("scheduler_1", "")),
                          "label": "S1"})

        return ({"pairs": pairs},)


class UltraBenchmarkTester:
    """
    Run UltraDuperSampler over:
      - sampler/scheduler pairs from UltraSamplerMatrix (outer loop)
      - sweep/script combos A..D (inner loop)

    Grid layout:
      X axis = sweep/script combos   (A / B / C / D / Base)
      Y axis = sampler_matrix rows   (S1 / S2 / ...)

    Outputs:
      - grid IMAGE
      - report STRING (tab-separated, suitable for .txt)
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
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.05}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "start_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "skip_tail": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "auto_flux_detect": ("BOOLEAN", {"default": True}),
                "allow_tf32": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "vae": ("VAE", {}),
                "special_cfg": ("DICT", {}),

                # sampler matrix (optional). If None, falls back to single sampler_name/scheduler.
                "sampler_matrix": ("DICT", {}),

                # up to 4 sweep/script combos
                "sweep_A": ("DICT", {}),
                "script_A": ("SCRIPT", {}),
                "sweep_B": ("DICT", {}),
                "script_B": ("SCRIPT", {}),
                "sweep_C": ("DICT", {}),
                "script_C": ("SCRIPT", {}),
                "sweep_D": ("DICT", {}),
                "script_D": ("SCRIPT", {}),

                "emit_image": ("BOOLEAN", {"default": True}),
                "purge_vram_each": ("BOOLEAN", {"default": False}),

                # TXT report controls
                "save_report": ("BOOLEAN", {"default": False}),
                "report_filename": ("STRING", {"default": "ultra_benchmark"}),
                "report_subfolder": ("STRING", {"default": "ultra_benchmark"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("grid", "report")
    FUNCTION = "run_tests"
    CATEGORY = "sampling/ultra"
    OUTPUT_NODE = False

    @torch.no_grad()
    def run_tests(
        self,
        model, positive, negative, latent, steps, cfg,
        sampler_name, scheduler, denoise, seed, start_noise, skip_tail,
        auto_flux_detect, allow_tf32,
        vae=None, special_cfg=None, sampler_matrix=None,
        sweep_A=None, script_A=None,
        sweep_B=None, script_B=None,
        sweep_C=None, script_C=None,
        sweep_D=None, script_D=None,
        emit_image=True, purge_vram_each=False,
        save_report=False, report_filename="ultra_benchmark", report_subfolder="ultra_benchmark"
    ):
        # ---- sweep/script combos (X axis) ----
        combo_defs = []
        for tag, sw, sc in [
            ("A", sweep_A, script_A),
            ("B", sweep_B, script_B),
            ("C", sweep_C, script_C),
            ("D", sweep_D, script_D),
        ]:
            if sw is not None or sc is not None:
                combo_defs.append({"tag": tag, "sweep": sw, "script": sc})
        if not combo_defs:
            combo_defs.append({"tag": "Base", "sweep": sweep_A, "script": script_A})

        # ---- sampler/scheduler pairs (Y axis) ----
        sampler_pairs = []
        if isinstance(sampler_matrix, dict) and isinstance(sampler_matrix.get("pairs"), list):
            for item in sampler_matrix["pairs"]:
                s = str(item.get("sampler", sampler_name))
                sch = str(item.get("scheduler", scheduler))
                lab = str(item.get("label", s))
                sampler_pairs.append({"sampler": s, "scheduler": sch, "label": lab})
        else:
            sampler_pairs.append({
                "sampler": str(sampler_name),
                "scheduler": str(scheduler),
                "label": "Base",
        })

        print(f"DEBUG: Number of columns (X) = {len(combo_defs)}; Number of rows (Y) = {len(sampler_pairs)}")

        uds = UltraDuperSampler()
        tiles = []
        meta_tiles = []
        const_seed = int(seed)

        # ---- nested loops: for each row (sampler_pair) then each column (combo_defs) ----
        for s_idx, sp in enumerate(sampler_pairs):
            s_name = sp["sampler"]
            sch_name = sp["scheduler"]
            s_label = sp["label"]
            for c_idx, combo in enumerate(combo_defs):
                tag = combo["tag"]
                sw_cfg = combo["sweep"]
                sc_cfg = combo["script"]

                # fresh latent copy each time
                local_latent = {
                    "samples": latent["samples"].clone(),
                    "batch_index": latent.get("batch_index"),
                }
                print(f"Running sampler {s_name}/{sch_name} row {s_idx}, combo {tag} col {c_idx}")
                out_latent, images, _ = uds.run(
                    model=model,
                    positive=positive,
                    negative=negative,
                    latent=local_latent,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=s_name,
                    scheduler=sch_name,
                    denoise=denoise,
                    seed=const_seed,
                    start_noise=start_noise,
                    skip_tail=skip_tail,
                    auto_flux_detect=auto_flux_detect,
                    purge_vram=bool(purge_vram_each),
                    allow_tf32=allow_tf32,
                    special_cfg=special_cfg,
                    special_noise=sw_cfg,
                    script=sc_cfg,
                    vae=vae,
                    emit_image=emit_image,
                )

                if not bool(emit_image):
                    raise RuntimeError("[UltraBenchmarkTester] emit_image=False; no IMAGE output.")
                if images is None or not isinstance(images, torch.Tensor):
                    raise RuntimeError("[UltraBenchmarkTester] UltraDuperSampler returned no image tensor.")
                if images.dim() != 4:
                    raise RuntimeError(f"[UltraBenchmarkTester] Expected 4D tensor, got {tuple(images.shape)}")

                b, h, w, c = images.shape
                for bi in range(b):
                    tiles.append(images[bi:bi+1].clone())
                    meta_tiles.append({
                        "tile_index": len(meta_tiles),
                        "sampler_index": s_idx,
                        "sampler": s_name,
                        "scheduler": sch_name,
                        "sampler_label": s_label,
                        "combo_index": c_idx,
                        "sweep_tag": tag,
                        "seed": const_seed,
                        "batch_index": bi,
                    })

                del local_latent, out_latent, images

        if not tiles:
            raise RuntimeError("[UltraBenchmarkTester] No tiles collected.")

        # ---- normalization (same as your version) ----
        heights = [t.shape[1] for t in tiles]
        widths = [t.shape[2] for t in tiles]
        min_h = min(heights)
        min_w = min(widths)
        if any(h != min_h or w != min_w for h, w in zip(heights, widths)):
            for i, t in enumerate(tiles):
                if t.shape[1] == min_h and t.shape[2] == min_w:
                    continue
                nchw = t.movedim(-1, 1)
                nchw = F.interpolate(nchw, size=(min_h, min_w), mode="bilinear", align_corners=False)
                tiles[i] = nchw.movedim(1, -1).clamp(0, 1)

        # ---- build grid ----
        sampler_ids = sorted({m["sampler_index"] for m in meta_tiles})
        combo_ids = sorted({m["combo_index"] for m in meta_tiles})
        batch_ids = sorted({m["batch_index"] for m in meta_tiles})
        if not batch_ids:
            batch_ids = [0]

        cell_map = {}
        for t, m in zip(tiles, meta_tiles):
            key = (m["sampler_index"], m["batch_index"], m["combo_index"])
            cell_map[key] = (t, m)

        blank = torch.zeros_like(tiles[0])
        rows_tensors = []
        report_lines = []
        report_lines.append("# row\tcol\tsampler_idx\tsampler_label\tsampler\tscheduler\tcombo_idx\tsweep_tag\tbatch_idx\tseed")

        row_idx = 0
        for s_idx in sampler_ids:
            base_meta = next((m for m in meta_tiles if m["sampler_index"] == s_idx), None)
            for b_idx in batch_ids:
                row_tiles = []
                col_idx = 0
                for c_idx in combo_ids:
                    key = (s_idx, b_idx, c_idx)
                    if key in cell_map:
                        tile, meta = cell_map[key]
                    else:
                        tile = blank
                        meta = base_meta
                    row_tiles.append(tile)
                    if meta is not None:
                        report_lines.append(
                            f"{row_idx}\t{col_idx}\t{s_idx}\t{meta['sampler_label']}\t"
                            f"{meta['sampler']}\t{meta['scheduler']}\t{c_idx}\t"
                            f"{meta['sweep_tag']}\t{b_idx}\t{meta['seed']}"
                        )
                    col_idx += 1
                row_cat = torch.cat(row_tiles, dim=2)
                rows_tensors.append(row_cat)
                row_idx += 1

        grid = torch.cat(rows_tensors, dim=1)
        report = "\n".join(report_lines)

        if save_report:
            try:
                try:
                    out_root = paths.get_output_directory()
                except Exception:
                    out_root = "."
                out_dir = os.path.join(out_root, report_subfolder) if report_subfolder else out_root
                os.makedirs(out_dir, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                base_name = report_filename or "ultra_benchmark"
                out_path = os.path.join(out_dir, f"{base_name}_{stamp}.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(report)
                # append path hint at end of report returned to UI
                report = report + f"\n# saved_to:\t{out_path}"
            except Exception as e:
                report = report + f"\n# save_report_failed: {e}"

        del tiles, rows_tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (grid, report)


class UltraBenchmarkSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "report": ("STRING", {}),
                "filename_prefix": ("STRING", {"default": "ultra_bench"}),
                "subfolder": ("STRING", {"default": "ultra_bench"}),
                "save_grid_image": ("BOOLEAN", {"default": True}),
                "save_individual_tiles": ("BOOLEAN", {"default": True}),
                "include_metadata": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("index_txt",)
    FUNCTION = "save"

    def _save_tensor_image(self, tensor, path):
        from PIL import Image
        img = (tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        img = Image.fromarray(img)
        img.save(path)

    def _parse_report(self, report):
        rows = []
        for line in report.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 10:
                rows.append({
                    "row": int(parts[0]), "col": int(parts[1]),
                    "sampler_label": parts[3], "sampler": parts[4], "scheduler": parts[5],
                    "sweep_tag": parts[7], "batch_idx": int(parts[8]), "seed": parts[9]
                })
        return rows

    def save(self, images, report, filename_prefix, subfolder, save_grid_image, save_individual_tiles, include_metadata):
        import os, time
        import folder_paths as paths

        if not isinstance(images, torch.Tensor) or images.dim() != 4:
            raise RuntimeError("Expected a 4D image tensor")

        images = images[:1]
        b, H, W, C = images.shape

        rows = self._parse_report(report)
        if not rows:
            raise RuntimeError("No report metadata found")

        max_row = max(r["row"] for r in rows)
        max_col = max(r["col"] for r in rows)
        rows_count = max_row + 1
        cols_count = max_col + 1

        tile_h = H // rows_count
        tile_w = W // cols_count

        try:
            base = paths.get_output_directory()
        except:
            base = "."

        out_dir = os.path.join(base, subfolder)
        os.makedirs(out_dir, exist_ok=True)

        stamp = time.strftime("%Y%m%d-%H%M%S")
        prefix = filename_prefix or "ultra_bench"
        index_lines = ["# row\tcol\tfilename" + ("\tsampler_label\tsampler\tscheduler\tsweep_tag\tbatch_idx\tseed" if include_metadata else "")]

        meta_map = {(r["row"], r["col"]): r for r in rows}

        if save_grid_image:
            grid_name = f"{prefix}_GRID_{stamp}.png"
            self._save_tensor_image(images[0], os.path.join(out_dir, grid_name))
            index_lines.append(f"# grid\t-\t{grid_name}")

        if save_individual_tiles:
            for r in range(rows_count):
                for c in range(cols_count):
                    tile = images[0, r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w, :]
                    m = meta_map.get((r, c), {"sampler_label":"Unknown","sampler":"","scheduler":"","sweep_tag":"","batch_idx":0,"seed":""})
                    fname = f"{prefix}_r{r}_c{c}_{m['sweep_tag']}_{m['sampler_label']}_{stamp}.png"
                    self._save_tensor_image(tile, os.path.join(out_dir, fname))
                    if include_metadata:
                        index_lines.append(f"{r}\t{c}\t{fname}\t{m['sampler_label']}\t{m['sampler']}\t{m['scheduler']}\t{m['sweep_tag']}\t{m['batch_idx']}\t{m['seed']}")
                    else:
                        index_lines.append(f"{r}\t{c}\t{fname}")

        index_path = os.path.join(out_dir, f"{prefix}_index_{stamp}.txt")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(index_lines))
        return (index_path,)
NODE_CLASS_MAPPINGS = {
    "UltraDuperSampler": UltraDuperSampler,
    "UltraSamplerMatrix": UltraSamplerMatrix,
    "UltraBenchmarkTester": UltraBenchmarkTester,
    "UltraBenchmarkSave": UltraBenchmarkSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraDuperSampler": "Ultra Duper Sampler (Flux-aware v3.2)",
    "UltraSamplerMatrix": "Ultra Sampler Matrix",
    "UltraBenchmarkTester": "Ultra Benchmark Tester",
    "UltraBenchmarkSave": "Ultra Benchmark Save",
}
