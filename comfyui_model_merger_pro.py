# comfyui_model_merger_pro.py
# -----------------------------------------------------------
# Model Merger Pro (SD/SDXL): high‑control, memory‑aware merging for
# Stable Diffusion 1.x/2.x/XL checkpoints with optional EMA preference,
# spherical (slerp) and norm‑matched blending, attention‑only filters,
# per‑stage UNet curves, overlap‑shape merging (experimental), metadata,
# GPU/CPU execution, multi‑merge (up to 6 models), and LoRA application.
#
# Nodes included:
# - ModelMergerPro: merge A (+ optional B..F) with rich controls → STATE_DICT
# - FromStateDictPro: instantiate (MODEL, CLIP, VAE) from a STATE_DICT
# - SaveMergedPro: save STATE_DICT to .safetensors with rich metadata
# - ApplyLoRAPro: apply one or more LoRA .safetensors onto a STATE_DICT
#
# Drop this file into: ComfyUI/custom_nodes/comfyui_model_merger_pro.py
# Restart ComfyUI.
# -----------------------------------------------------------

from __future__ import annotations
import os
import math
import time
from typing import Dict, Iterable, Callable, Optional, Tuple, List, Sequence

import torch

try:
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False

import folder_paths as paths

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def _resolve_checkpoint_path(name: str) -> str:
    """Resolve a checkpoint file name or path to an absolute path."""
    if not name:
        raise FileNotFoundError("Empty checkpoint name")
    if os.path.isabs(name) and os.path.exists(name):
        return name
    for ckpt_dir in paths.get_folder_paths("checkpoints"):
        if not os.path.isdir(ckpt_dir):
            continue
        for cand in (name, name + ".safetensors", name + ".ckpt"):
            p = os.path.join(ckpt_dir, cand)
            if os.path.exists(p):
                return os.path.abspath(p)
    raise FileNotFoundError(f"Could not find checkpoint: {name}")


def _load_state_dict_any(path: str) -> Dict[str, torch.Tensor]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        if not _HAS_SAFETENSORS:
            raise RuntimeError("safetensors not available; install safetensors")
        return load_safetensors(path)
    data = torch.load(path, map_location="cpu")
    return data.get("state_dict", data)


def _save_state_dict_safetensors(path: str, state_dict: Dict[str, torch.Tensor], meta: Optional[dict] = None):
    if not _HAS_SAFETENSORS:
        raise RuntimeError("safetensors not available; install safetensors")
    cpu_sd = {k: (v.detach().cpu().contiguous()) for k, v in state_dict.items()}
    metadata = {k: str(v) for k, v in (meta or {}).items()}
    save_safetensors(cpu_sd, path, metadata=metadata)


def _dtype_from_name(name: str):
    name = (name or "auto").lower()
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }.get(name, None)


def _cast_state_dict(sd: Dict[str, torch.Tensor], dtype):
    if dtype is None:
        return sd
    out = {}
    for k, v in sd.items():
        try:
            out[k] = v.to(dtype) if torch.is_floating_point(v) else v
        except Exception:
            out[k] = v
    return out


def _intersecting_keys(a: Iterable[str], b: Iterable[str]) -> Iterable[str]:
    sb = set(b)
    for k in a:
        if k in sb:
            yield k


# ---------- EMA handling ----------

def _prefer_ema(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """If EMA weights exist (keys prefixed by 'model_ema.'), swap them in place
    for their non‑EMA counterparts when shapes match."""
    if not any(k.startswith("model_ema.") for k in sd.keys()):
        return sd
    out = dict(sd)
    for k, v in sd.items():
        if k.startswith("model_ema."):
            base = k[len("model_ema."):]
            if base in sd and getattr(sd[base], 'shape', None) == getattr(v, 'shape', None):
                out[base] = v
    return out


# ---------- Merge primitives ----------

def _weighted_merge(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0.0:
        return a
    if alpha >= 1.0:
        return b
    return (1.0 - alpha) * a + alpha * b


def _diff_merge(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    # A + alpha * (B - A)
    return a + alpha * (b - a)


def _slerp_merge(a: torch.Tensor, b: torch.Tensor, alpha: float, eps: float = 1e-8) -> torch.Tensor:
    """Spherical interpolation across flattened vectors (compute in float32)."""
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    a_n = torch.linalg.norm(a_f) + eps
    b_n = torch.linalg.norm(b_f) + eps
    a_u = a_f / a_n
    b_u = b_f / b_n
    dot = torch.clamp(torch.dot(a_u, b_u), -1.0, 1.0)
    theta = torch.acos(dot)
    if theta < 1e-3:
        out = _weighted_merge(a_f, b_f, alpha)
        return out.reshape_as(a).to(a.dtype)
    s1 = torch.sin((1 - alpha) * theta)
    s2 = torch.sin(alpha * theta)
    out = (s1 * a_u + s2 * b_u) / torch.sin(theta)
    out = out * ((1 - alpha) * a_n + alpha * b_n)
    return out.reshape_as(a).to(a.dtype)


def _norm_matched_merge(a: torch.Tensor, b: torch.Tensor, alpha: float, eps: float = 1e-6) -> torch.Tensor:
    """Match B's tensor to A's L2 norm (compute in float32)."""
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)
    a_n = torch.linalg.norm(a_f.reshape(-1)) + eps
    b_n = torch.linalg.norm(b_f.reshape(-1)) + eps
    scale = a_n / b_n
    out = (1.0 - alpha) * a_f + alpha * (b_f * scale)
    return out.to(a.dtype)


# ---------- Shape handling (experimental) ----------

def _overlap_slices(shape_a: Sequence[int], shape_b: Sequence[int]) -> Optional[Tuple[Tuple[slice, ...], Tuple[slice, ...]]]:
    if len(shape_a) != len(shape_b):
        return None
    slices_a = []
    slices_b = []
    for da, db in zip(shape_a, shape_b):
        m = min(int(da), int(db))
        if m <= 0:
            return None
        slices_a.append(slice(0, m))
        slices_b.append(slice(0, m))
    return tuple(slices_a), tuple(slices_b)


def _merge_maybe_mismatched(a: torch.Tensor, b: torch.Tensor, alpha: float, mode: str, shape_mode: str) -> torch.Tensor:
    # Non‑floating (e.g., int position_ids): pick A/B discretely, no arithmetic.
    if not (torch.is_floating_point(a) and torch.is_floating_point(b)):
        return a if alpha < 0.5 else b

    if a.shape == b.shape:
        if mode == "weighted_sum":
            return _weighted_merge(a, b, alpha)
        if mode == "difference":
            return _diff_merge(a, b, alpha)
        if mode == "slerp":
            return _slerp_merge(a, b, alpha)
        if mode == "norm_matched":
            return _norm_matched_merge(a, b, alpha)
        return _weighted_merge(a, b, alpha)

    # Mismatch
    if shape_mode != "overlap":
        return a  # strict: keep A

    sl = _overlap_slices(a.shape, b.shape)
    if sl is None:
        return a
    sa, sb = sl
    out = a.clone()
    a_sub = a[sa]
    b_sub = b[sb]
    if mode == "weighted_sum":
        out[sa] = _weighted_merge(a_sub, b_sub, alpha)
    elif mode == "difference":
        out[sa] = _diff_merge(a_sub, b_sub, alpha)
    elif mode == "slerp":
        out[sa] = _slerp_merge(a_sub, b_sub, alpha)
    elif mode == "norm_matched":
        out[sa] = _norm_matched_merge(a_sub, b_sub, alpha)
    else:
        out[sa] = _weighted_merge(a_sub, b_sub, alpha)
    return out


# ---------- UNet stage utilities ----------

def _is_down_block(k: str) -> bool: return "input_blocks" in k

def _is_mid_block(k: str) -> bool: return "middle_block" in k

def _is_up_block(k: str) -> bool: return "output_blocks" in k

CURVES = ["constant", "a_to_b_ramp", "b_to_a_ramp", "centered"]


def _scan_unet_ranges(keys: Iterable[str]) -> Tuple[int, int]:
    max_in = -1
    max_out = -1
    for k in keys:
        if "input_blocks." in k:
            try:
                idx = int(k.split("input_blocks.")[1].split(".")[0])
                max_in = max(max_in, idx)
            except Exception:
                pass
        if "output_blocks." in k:
            try:
                idx = int(k.split("output_blocks.")[1].split(".")[0])
                max_out = max(max_out, idx)
            except Exception:
                pass
    return max_in, max_out


def _curve_factor_for_key(k: str, max_in: int, max_out: int, curve_mode: str, power: float) -> float:
    if curve_mode == "constant":
        return 1.0
    t = 0.0
    if "input_blocks." in k and max_in > 0:
        try:
            idx = int(k.split("input_blocks.")[1].split(".")[0])
            t = idx / max(1, max_in)
        except Exception:
            t = 0.0
    elif "output_blocks." in k and max_out > 0:
        try:
            idx = int(k.split("output_blocks.")[1].split(".")[0])
            t = idx / max(1, max_out)
        except Exception:
            t = 0.0
    elif "middle_block" in k:
        t = 0.5
    p = max(0.5, float(power))
    if curve_mode == "a_to_b_ramp":
        return t ** p
    if curve_mode == "b_to_a_ramp":
        return (1.0 - t) ** p
    if curve_mode == "centered":
        return 1.0 - abs(2.0 * t - 1.0) ** p
    return 1.0


# ---------- Attention filter ----------

def _attn_filter_factory(mode: str) -> Optional[Callable[[str], bool]]:
    mode = (mode or "all").lower()
    if mode == "all":
        return None
    if mode == "qkv_only":
        return lambda k: ("to_q" in k) or ("to_k" in k) or ("to_v" in k)
    if mode == "attn_only":
        # cover common attention terms across SD families
        return lambda k: ("attn" in k) or ("to_out.0" in k) or ("to_q" in k) or ("to_k" in k) or ("to_v" in k)
    if mode == "conv_only":
        return lambda k: ("conv" in k) or (".proj" in k)
    return None


# ---------- Prefixes (generic across SD families) ----------

PREFIXES = {
    "unet": [
        "model.diffusion_model.",
    ],
    "vae": [
        "first_stage_model.",
        "model_ema.first_stage_model.",
    ],
    # For SDXL, CLIP_L and CLIP_G coexist; for SD 1.x/2.x typically one CLIP encoder exists under cond_stage_model.
    "clip_l": [
        "cond_stage_model.transformer.text_model.",
        "cond_stage_model.clip_l.",
        "conditioner.embedders.0.transformer.text_model.",
    ],
    "clip_g": [
        "cond_stage_model.clip_g.",
        "conditioner.embedders.1.transformer.text_model.",
    ],
    # Fallback for SD 1.x/2.x single CLIP
    "clip_any": [
        "cond_stage_model.",
    ],
}


# ---------- Core merge ----------

def merge_models(
    sd_as: Sequence[Dict[str, torch.Tensor]],
    alphas_unet: Sequence[float],
    alpha_clip_l: float,
    alpha_clip_g: float,
    alpha_clip_any: float,
    alpha_vae: float,
    mode_unet: str = "weighted_sum",
    mode_others: str = "weighted_sum",
    unet_down: Optional[float] = None,
    unet_mid: Optional[float] = None,
    unet_up: Optional[float] = None,
    attn_filter: str = "all",
    spice: float = 0.0,
    clip_merge_mode: str = "both",
    curve_down: str = "constant",
    curve_mid: str = "constant",
    curve_up: str = "constant",
    curve_power: float = 1.0,
    shape_mode: str = "strict",
) -> Dict[str, torch.Tensor]:
    """Multi‑source merge (2..6 sources). sd_as[0] is the base A."""
    assert len(sd_as) >= 2, "Need at least two models to merge"
    assert len(alphas_unet) == len(sd_as) - 1

    merged = dict(sd_as[0])

    alpha_nudge = 0.2 * float(spice)

    def _pick(mode: str):
        return {
            "weighted_sum": _weighted_merge,
            "difference": _diff_merge,
            "slerp": _slerp_merge,
            "norm_matched": _norm_matched_merge,
        }.get(mode, _weighted_merge)

    fn_unet = _pick(mode_unet)
    fn_other = _pick(mode_others)

    key_filter = _attn_filter_factory(attn_filter)

    # CLIP mode
    clip_merge_mode = clip_merge_mode.lower()
    if clip_merge_mode == "none":
        alpha_l_eff = 0.0
        alpha_g_eff = 0.0
    elif clip_merge_mode in ("l only", "l_only"):
        alpha_l_eff = alpha_clip_l
        alpha_g_eff = 0.0
    elif clip_merge_mode in ("g only", "g_only"):
        alpha_l_eff = 0.0
        alpha_g_eff = alpha_clip_g
    else:
        alpha_l_eff = alpha_clip_l
        alpha_g_eff = alpha_clip_g

    # UNet keys for curve scanning based on base
    unet_keys = [k for k in sd_as[0].keys() if any(k.startswith(p) for p in PREFIXES["unet"])]
    max_in, max_out = _scan_unet_ranges(unet_keys)

    def _alpha_for_key(k: str, base_alpha: float) -> float:
        if _is_down_block(k) and unet_down is not None:
            base_alpha = unet_down
            curve = curve_down
        elif _is_mid_block(k) and unet_mid is not None:
            base_alpha = unet_mid
            curve = curve_mid
        elif _is_up_block(k) and unet_up is not None:
            base_alpha = unet_up
            curve = curve_up
        else:
            curve = "constant"
        fac = _curve_factor_for_key(k, max_in, max_out, curve, curve_power)
        return float(max(0.0, min(1.0, base_alpha * fac)))

    # ---------- UNet merge ----------
    base_keys = [k for k in sd_as[0].keys() if any(k.startswith(p) for p in PREFIXES["unet"])]
    for k in _intersecting_keys(base_keys, sd_as[1].keys()):
        if key_filter is not None and not key_filter(k):
            continue
        a = sd_as[0][k]
        acc = a
        for i, sd_b in enumerate(sd_as[1:]):
            if k not in sd_b:
                continue
            b = sd_b[k]
            alpha_eff = max(0.0, min(1.0, (alphas_unet[i] + alpha_nudge)))
            a_eff = _alpha_for_key(k, alpha_eff)
            acc = _merge_maybe_mismatched(acc, b, a_eff, mode_unet, shape_mode)
        merged[k] = acc

    # ---------- CLIPs & VAE ----------
    def _apply(prefixes: List[str], alpha: float, mode: str):
        if alpha is None or alpha == 0.0:
            return
        keys = [k for k in sd_as[0].keys() if any(k.startswith(p) for p in prefixes)]
        for k in keys:
            acc = sd_as[0].get(k)
            if acc is None:
                continue
            for sd_b in sd_as[1:]:
                if k in sd_b:
                    acc = _merge_maybe_mismatched(acc, sd_b[k], alpha, mode, shape_mode)
            merged[k] = acc

    _apply(PREFIXES["clip_l"], alpha_l_eff, mode_others)
    _apply(PREFIXES["clip_g"], alpha_g_eff, mode_others)
    _apply(PREFIXES["clip_any"], alpha_clip_any, mode_others)
    _apply(PREFIXES["vae"], alpha_vae, mode_others)

    return merged


# -----------------------------------------------------------
# ComfyUI Nodes
# -----------------------------------------------------------

class ModelMergerPro:
    @classmethod
    def INPUT_TYPES(cls):
        ckpt_names: List[str] = []
        for folder in paths.get_folder_paths("checkpoints"):
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".safetensors", ".ckpt")):
                    ckpt_names.append(fname)
        ckpt_names = sorted(list(set(ckpt_names)))
        ckpt_none = ["<none>"] + ckpt_names

        return {
            "required": {
                # Mandatory A + B, optional C..F (left as <none>)
                "ckpt_a": (ckpt_names, {"default": ckpt_names[0] if ckpt_names else ""}),
                "ckpt_b": (ckpt_names, {"default": ckpt_names[0] if ckpt_names else ""}),
                "ckpt_c": (ckpt_none, {"default": "<none>"}),
                "ckpt_d": (ckpt_none, {"default": "<none>"}),
                "ckpt_e": (ckpt_none, {"default": "<none>"}),
                "ckpt_f": (ckpt_none, {"default": "<none>"}),

                # Core alphas (UNet for each extra model B..F)
                "alpha_unet_b": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_unet_c": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_unet_d": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_unet_e": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_unet_f": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # CLIP/other alphas
                "alpha_clip_l": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_clip_g": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_clip_any": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_vae": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Modes
                "mode_unet": (["weighted_sum", "difference", "slerp", "norm_matched"], {"default": "weighted_sum"}),
                "mode_others": (["weighted_sum", "difference", "slerp", "norm_matched"], {"default": "weighted_sum"}),
                "attn_filter": (["all", "qkv_only", "attn_only", "conv_only"], {"default": "all"}),
                "clip_merge_mode": (["both", "L only", "G only", "none"], {"default": "both"}),

                # UNet per‑stage + curves
                "unet_down": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "unet_mid":  ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "unet_up":   ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "curve_down": (CURVES, {"default": "constant"}),
                "curve_mid":  (CURVES, {"default": "constant"}),
                "curve_up":   (CURVES, {"default": "constant"}),
                "curve_power": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1}),

                # Advanced
                "spice": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prefer_ema": ("BOOLEAN", {"default": True}),
                "shape_mode": (["strict", "overlap"], {"default": "strict"}),
                "target_dtype": (["auto", "float32", "float16", "bfloat16"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STATE_DICT", "STRING")
    RETURN_NAMES = ("state_dict", "summary")
    FUNCTION = "merge"
    CATEGORY = "model/merging"

    def merge(
        self,
        ckpt_a, ckpt_b, ckpt_c, ckpt_d, ckpt_e, ckpt_f,
        alpha_unet_b, alpha_unet_c, alpha_unet_d, alpha_unet_e, alpha_unet_f,
        alpha_clip_l, alpha_clip_g, alpha_clip_any, alpha_vae,
        mode_unet, mode_others, attn_filter, clip_merge_mode,
        unet_down, unet_mid, unet_up,
        curve_down, curve_mid, curve_up, curve_power,
        spice, prefer_ema, shape_mode, target_dtype,
    ):
        # Resolve
        path_a = _resolve_checkpoint_path(ckpt_a)
        path_b = _resolve_checkpoint_path(ckpt_b)

        def _maybe(p):
            return None if (p in (None, "<none>", "")) else _resolve_checkpoint_path(p)

        paths_opt = [
            _maybe(ckpt_c),
            _maybe(ckpt_d),
            _maybe(ckpt_e),
            _maybe(ckpt_f),
        ]

        # Load
        with torch.no_grad():
            sd_a = _load_state_dict_any(path_a)
            sd_b = _load_state_dict_any(path_b)
            sd_list = [sd_a, sd_b]
            for pp in paths_opt:
                if pp:
                    sd_list.append(_load_state_dict_any(pp))

            if prefer_ema:
                sd_list = [_prefer_ema(sd) for sd in sd_list]

            # Alphas per extra source
            alphas = [alpha_unet_b]
            for x in (alpha_unet_c, alpha_unet_d, alpha_unet_e, alpha_unet_f):
                if len(alphas) < len(sd_list) - 0:  # fill as many as needed; extra ignored later
                    alphas.append(x)
            # trim to sd_list
            alphas = alphas[: max(1, len(sd_list) - 1)]

            merged_sd = merge_models(
                sd_list,
                alphas_unet=alphas,
                alpha_clip_l=alpha_clip_l,
                alpha_clip_g=alpha_clip_g,
                alpha_clip_any=alpha_clip_any,
                alpha_vae=alpha_vae,
                mode_unet=mode_unet,
                mode_others=mode_others,
                unet_down=unet_down,
                unet_mid=unet_mid,
                unet_up=unet_up,
                attn_filter=attn_filter,
                spice=spice,
                clip_merge_mode=clip_merge_mode,
                curve_down=curve_down,
                curve_mid=curve_mid,
                curve_up=curve_up,
                curve_power=curve_power,
                shape_mode=shape_mode,
            )

            dtype = _dtype_from_name(target_dtype)
            if dtype is not None:
                merged_sd = _cast_state_dict(merged_sd, dtype)

        # Summary
        used = [os.path.basename(path_a), os.path.basename(path_b)]
        used += [os.path.basename(p) for p in paths_opt if p]
        summary = (
            f"Models: {', '.join(used)} | UNet alphas (B..): "
            f"{alpha_unet_b:.2f},{alpha_unet_c:.2f},{alpha_unet_d:.2f},{alpha_unet_e:.2f},{alpha_unet_f:.2f} "
            f"(down={unet_down:.2f}, mid={unet_mid:.2f}, up={unet_up:.2f}, curves={curve_down}/{curve_mid}/{curve_up}, p={curve_power}) "
            f"| clip(L={alpha_clip_l:.2f}, G={alpha_clip_g:.2f}, any={alpha_clip_any:.2f}, mode={clip_merge_mode}) "
            f"| vae={alpha_vae:.2f} | mode_unet={mode_unet} | mode_others={mode_others} | attn_filter={attn_filter} "
            f"| spice={spice:.2f} | ema={prefer_ema} | shape={shape_mode} | dtype={target_dtype}"
        )
        return (merged_sd, summary)


class FromStateDictPro:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"state_dict": ("STATE_DICT", {})}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "instantiate"
    CATEGORY = "model/merging"

    def instantiate(self, state_dict):
        from comfy.sd import load_checkpoint_guess_config
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="merge_pro_")
        tmp_path = os.path.join(tmp_dir, f"{time.time_ns()}.safetensors")
        _save_state_dict_safetensors(tmp_path, state_dict, meta={"source": "FromStateDictPro"})
        try:
            model, clip, vae, _ = load_checkpoint_guess_config(tmp_path, output_vae=True, output_clip=True)
            return (model, clip, vae)
        finally:
            try:
                if os.path.exists(tmp_path): os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except Exception:
                pass


class SaveMergedPro:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state_dict": ("STATE_DICT", {}),
                "file_name": ("STRING", {"default": "merged_pro"}),
                "subfolder": ("STRING", {"default": "merged"}),
                "meta_comment": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "model/merging"

    def save(self, state_dict, file_name, subfolder, meta_comment):
        ckpt_dirs = [d for d in paths.get_folder_paths("checkpoints") if os.path.isdir(d)]
        out_root = ckpt_dirs[0] if ckpt_dirs else "."
        out_dir = os.path.join(out_root, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(out_dir, f"{file_name}_{stamp}.safetensors")
        meta = {"comment": meta_comment, "timestamp": stamp}
        _save_state_dict_safetensors(out_path, state_dict, meta=meta)
        return (out_path,)


# ---------- LoRA application ----------

def _load_lora(path: str) -> Dict[str, torch.Tensor]:
    if not _HAS_SAFETENSORS:
        raise RuntimeError("safetensors not available; install safetensors for LoRA")
    return load_safetensors(path)


def _apply_lora_once(sd: Dict[str, torch.Tensor], lora_sd: Dict[str, torch.Tensor], strength: float) -> Dict[str, torch.Tensor]:
    out = dict(sd)
    alphas = {k: v.item() for k, v in lora_sd.items() if k.endswith(".alpha") and v.ndim == 0}

    def find_pairs(prefix: str):
        pairs = []
        for k in lora_sd.keys():
            if k.endswith(".lora_up.weight") and k.startswith(prefix):
                base = k[:-len(".lora_up.weight")]
                up = k
                down = base + ".lora_down.weight"
                if down in lora_sd:
                    r = lora_sd[down].shape[0]
                    pairs.append((base, up, down, r))
        return pairs

    for base, upk, downk, r in find_pairs(""):
        up = lora_sd[upk]
        down = lora_sd[downk]
        Wk = base + ".weight"
        if Wk not in out:
            continue
        W = out[Wk]
        up2 = up.reshape(up.shape[0], -1)
        down2 = down.reshape(down.shape[0], -1)
        try:
            delta = torch.matmul(up2, down2)
        except Exception:
            continue
        delta = delta.reshape_as(W)
        alpha = alphas.get(base + ".alpha", r)
        scale = (alpha / max(1, r)) * float(strength)
        out[Wk] = W + delta.to(W.dtype) * scale
    return out


class ApplyLoRAPro:
    @classmethod
    def INPUT_TYPES(cls):
        # Guard against non‑existent lora paths; do not throw at import time.
        lora_names: List[str] = []
        try:
            search_dirs = [d for d in paths.get_folder_paths("loras") if os.path.isdir(d)]
        except Exception:
            search_dirs = []
        for folder in search_dirs:
            try:
                for fname in os.listdir(folder):
                    if fname.lower().endswith(".safetensors"):
                        lora_names.append(fname)
            except Exception:
                pass
        lora_names = sorted(list(set(lora_names)))
        if not lora_names:
            lora_names = ["<provide path>"]

        return {
            "required": {
                "state_dict": ("STATE_DICT", {}),
                "lora_file": (lora_names, {"default": (lora_names[0])}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "custom_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STATE_DICT", "STRING")
    RETURN_NAMES = ("state_dict", "summary")
    FUNCTION = "apply"
    CATEGORY = "model/merging"

    def apply(self, state_dict, lora_file, strength, custom_path):
        path = None
        search_dirs = []
        try:
            search_dirs = [d for d in paths.get_folder_paths("loras") if os.path.isdir(d)]
        except Exception:
            search_dirs = []

        if lora_file and lora_file != "<provide path>":
            for folder in search_dirs:
                cand = os.path.join(folder, lora_file)
                if os.path.exists(cand):
                    path = cand
                    break
        if not path and custom_path and os.path.exists(custom_path):
            path = custom_path
        if not path:
            raise FileNotFoundError("LoRA file not found; select from dropdown or provide custom_path")

        lora_sd = _load_lora(path)
        out = _apply_lora_once(state_dict, lora_sd, strength)
        summary = f"Applied LoRA: {os.path.basename(path)} @ strength {strength:.2f}"
        return (out, summary)


# ---------- Node registration ----------
NODE_CLASS_MAPPINGS = {
    "ModelMergerPro": ModelMergerPro,
    "FromStateDictPro": FromStateDictPro,
    "SaveMergedPro": SaveMergedPro,
    "ApplyLoRAPro": ApplyLoRAPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMergerPro": "Model Merger Pro (SD/SDXL)",
    "FromStateDictPro": "Instantiate from STATE_DICT (Pro)",
    "SaveMergedPro": "Save Merged (Pro, metadata)",
    "ApplyLoRAPro": "Apply LoRA to STATE_DICT",
}
