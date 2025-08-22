# ComfyUI Custom Nodes: SDXLDualMerger (+ extras)
# -----------------------------------------------------------
# A flexible SDXL merger for any two SDXL checkpoints.
# - SDXLDualMerger: computes an in-memory merge and returns STATE_DICT.
# - SDXLFromStateDict: instantiates MODEL/CLIP/VAE from a STATE_DICT (temp file).
# - SDXLMergedSaver: saves a STATE_DICT to .safetensors.
#
# Upgrades in this refactor:
# - **True dropdowns (COMBO)** for modes/filters (no more STRING boxes).
# - **CLIP merge mode**: both / L only / G only / none.
# - **UNet per-stage curves** (down/mid/up): constant, A→B ramp, B→A ramp, centered (with power).
# - **Target dtype cast**: auto / float32 / float16 / bfloat16 (applied to merged params).
# - Cleaned duplicate class, safer file ops, small summaries.
#
# Install:
#   Save as: ComfyUI/custom_nodes/sdxl_dual_merger.py  and restart ComfyUI.
# -----------------------------------------------------------

import os
import time
import math
from typing import Dict, Iterable, Callable, Optional, Tuple, List

import torch

try:
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False

# ComfyUI internals (safe at import time)
import folder_paths as paths


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def _resolve_checkpoint_path(name: str) -> str:
    """Resolve a checkpoint file name or path to an absolute path."""
    if os.path.isabs(name) and os.path.exists(name):
        return name
    # Known ComfyUI checkpoints folder(s)
    for ckpt_dir in paths.get_folder_paths("checkpoints"):
        candidates = [
            os.path.join(ckpt_dir, name),
            os.path.join(ckpt_dir, name + ".safetensors"),
            os.path.join(ckpt_dir, name + ".ckpt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
    raise FileNotFoundError(f"Could not find checkpoint: {name}")


def _load_state_dict_any(path: str) -> Dict[str, torch.Tensor]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        if not _HAS_SAFETENSORS:
            raise RuntimeError("safetensors not available; install safetensors")
        return load_safetensors(path)
    else:
        data = torch.load(path, map_location="cpu")
        return data.get("state_dict", data)


def _save_state_dict_safetensors(path: str, state_dict: Dict[str, torch.Tensor]):
    if not _HAS_SAFETENSORS:
        raise RuntimeError("safetensors not available; install safetensors")
    cpu_sd = {k: (v.detach().cpu().contiguous()) for k, v in state_dict.items()}
    save_safetensors(cpu_sd, path)


def _intersecting_keys(a: Iterable[str], b: Iterable[str]) -> Iterable[str]:
    set_b = set(b)
    for k in a:
        if k in set_b:
            yield k


def _weighted_merge(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0.0:
        return a
    if alpha >= 1.0:
        return b
    return (1.0 - alpha) * a + alpha * b


def _diff_merge(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    # A + alpha * (B - A)
    return a + alpha * (b - a)


def _dtype_from_name(name: str):
    name = (name or "auto").lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return None  # auto


def _cast_state_dict(sd: Dict[str, torch.Tensor], dtype):
    if dtype is None:
        return sd
    out = {}
    for k, v in sd.items():
        try:
            if torch.is_floating_point(v):
                out[k] = v.to(dtype)
            else:
                out[k] = v
        except Exception:
            out[k] = v
    return out


# Common SDXL module prefixes
SDXL_PREFIXES = {
    "unet": [
        "model.diffusion_model.",
    ],
    "vae": [
        "first_stage_model.",
        "model_ema.first_stage_model.",
    ],
    # SDXL ships with two text encoders: clip_l (OpenCLIP big G) and clip_g (OpenCLIP ViT-G/14)
    "clip_l": [
        "cond_stage_model.transformer.text_model.",
        "cond_stage_model.clip_l.",
        "conditioner.embedders.0.transformer.text_model.",
    ],
    "clip_g": [
        "cond_stage_model.clip_g.",
        "conditioner.embedders.1.transformer.text_model.",
    ],
}


# UNet sub-block helpers

def _is_down_block(k: str) -> bool:
    return "input_blocks" in k

def _is_mid_block(k: str) -> bool:
    return "middle_block" in k

def _is_up_block(k: str) -> bool:
    return "output_blocks" in k


def _attn_filter_factory(mode: str) -> Optional[Callable[[str], bool]]:
    mode = (mode or "all").lower()
    if mode == "all":
        return None
    if mode == "qkv_only":
        def _f(k: str) -> bool:
            return ("to_q" in k) or ("to_k" in k) or ("to_v" in k)
        return _f
    if mode == "attn_only":
        def _f(k: str) -> bool:
            return ("attn" in k) or ("to_out.0" in k) or ("to_q" in k) or ("to_k" in k) or ("to_v" in k)
        return _f
    if mode == "conv_only":
        def _f(k: str) -> bool:
            return "conv" in k or ".proj" in k
        return _f
    return None


def _apply_merge_for_prefix(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    merged: Dict[str, torch.Tensor],
    prefix_filters: Iterable[str],
    alpha: float,
    fn_merge,
    key_filter: Optional[Callable[[str], bool]] = None,
):
    if alpha is None:
        return
    keys_a = [k for k in sd_a.keys() if any(k.startswith(p) for p in prefix_filters)]
    keys_b = sd_b.keys()
    for k in _intersecting_keys(keys_a, keys_b):
        if key_filter is not None and not key_filter(k):
            continue
        try:
            if sd_a[k].shape == sd_b[k].shape:
                merged[k] = fn_merge(sd_a[k], sd_b[k], alpha)
        except Exception:
            pass


# ---------- Curves across UNet depth ----------

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


def merge_sdxl(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    alpha_unet: float,
    alpha_clip_l: float,
    alpha_clip_g: float,
    alpha_vae: float,
    mode_unet: str = "weighted_sum",
    mode_others: str = "weighted_sum",
    unet_down: Optional[float] = None,
    unet_mid: Optional[float] = None,
    unet_up: Optional[float] = None,
    attn_filter: str = "all",
    spice: float = 0.0,
    # new:
    clip_merge_mode: str = "both",
    curve_down: str = "constant",
    curve_mid: str = "constant",
    curve_up: str = "constant",
    curve_power: float = 1.0,
) -> Dict[str, torch.Tensor]:

    merged = dict(sd_a)  # start from A

    # Macro spice: gently push UNet alpha toward B by +spice*0.2
    alpha_unet_eff = max(0.0, min(1.0, alpha_unet + 0.2 * spice))

    fn_unet = _weighted_merge if mode_unet == "weighted_sum" else _diff_merge
    fn_other = _weighted_merge if mode_others == "weighted_sum" else _diff_merge

    key_filter = _attn_filter_factory(attn_filter)

    # Adjust CLIP mode
    if clip_merge_mode == "none":
        alpha_clip_l_eff = 0.0
        alpha_clip_g_eff = 0.0
    elif clip_merge_mode == "L only":
        alpha_clip_l_eff = alpha_clip_l
        alpha_clip_g_eff = 0.0
    elif clip_merge_mode == "G only":
        alpha_clip_l_eff = 0.0
        alpha_clip_g_eff = alpha_clip_g
    else:
        alpha_clip_l_eff = alpha_clip_l
        alpha_clip_g_eff = alpha_clip_g

    # Pre-scan UNet ranges for curves
    unet_keys = [k for k in sd_a.keys() if any(k.startswith(p) for p in SDXL_PREFIXES["unet"])]
    max_in, max_out = _scan_unet_ranges(unet_keys)

    # Helper to choose base alpha by block and apply curve
    def _alpha_for_key(k: str, base: float) -> float:
        # per-stage override
        if _is_down_block(k) and unet_down is not None:
            base = unet_down
            curve = curve_down
        elif _is_mid_block(k) and unet_mid is not None:
            base = unet_mid
            curve = curve_mid
        elif _is_up_block(k) and unet_up is not None:
            base = unet_up
            curve = curve_up
        else:
            curve = "constant"
        # curve factor
        fac = _curve_factor_for_key(k, max_in, max_out, curve, curve_power)
        return float(max(0.0, min(1.0, base * fac)))

    # UNet merge with per-key effective alpha
    keys = unet_keys
    for k in _intersecting_keys(keys, sd_b.keys()):
        if key_filter is not None and not key_filter(k):
            continue
        try:
            if sd_a[k].shape == sd_b[k].shape:
                a_eff = _alpha_for_key(k, alpha_unet_eff)
                merged[k] = fn_unet(sd_a[k], sd_b[k], a_eff)
        except Exception:
            pass

    # CLIPs & VAE
    _apply_merge_for_prefix(sd_a, sd_b, merged, SDXL_PREFIXES["clip_l"], alpha_clip_l_eff, fn_other)
    _apply_merge_for_prefix(sd_a, sd_b, merged, SDXL_PREFIXES["clip_g"], alpha_clip_g_eff, fn_other)
    _apply_merge_for_prefix(sd_a, sd_b, merged, SDXL_PREFIXES["vae"], alpha_vae, fn_other)

    return merged


# -----------------------------------------------------------
# ComfyUI Nodes
# -----------------------------------------------------------

class SDXLDualMerger:
    @classmethod
    def INPUT_TYPES(cls):
        # Build a list of visible checkpoint names using ComfyUI's path helper
        ckpt_names = []
        for folder in paths.get_folder_paths("checkpoints"):
            for fname in os.listdir(folder):
                if fname.lower().endswith((".safetensors", ".ckpt")):
                    ckpt_names.append(fname)
        ckpt_names = sorted(list(set(ckpt_names)))

        return {
            "required": {
                # Checkpoint pickers (true dropdowns via list type)
                "ckpt_a": (ckpt_names, {"default": ckpt_names[0] if ckpt_names else ""}),
                "ckpt_b": (ckpt_names, {"default": ckpt_names[0] if ckpt_names else ""}),

                # Core alphas
                "alpha_unet": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_clip_l": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_clip_g": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_vae": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Modes (COMBOs)
                "mode_unet": (["weighted_sum", "difference"], {"default": "weighted_sum"}),
                "mode_others": (["weighted_sum", "difference"], {"default": "weighted_sum"}),
                "attn_filter": (["all", "qkv_only", "attn_only", "conv_only"], {"default": "all"}),

                # CLIP merge mode
                "clip_merge_mode": (["both", "L only", "G only", "none"], {"default": "both"}),

                # UNet per-stage overrides + curves
                "unet_down": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "unet_mid": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "unet_up":  ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),

                "curve_down": (["constant","a_to_b_ramp","b_to_a_ramp","centered"], {"default": "constant"}),
                "curve_mid":  (["constant","a_to_b_ramp","b_to_a_ramp","centered"], {"default": "constant"}),
                "curve_up":   (["constant","a_to_b_ramp","b_to_a_ramp","centered"], {"default": "constant"}),
                "curve_power": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1}),

                # Macro spice
                "spice": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Target dtype
                "target_dtype": (["auto","float32","float16","bfloat16"], {"default": "auto"}),
            },
        }

    # Output only STATE_DICT to avoid cache issues; instantiate in a separate node
    RETURN_TYPES = ("STATE_DICT", "STRING")
    RETURN_NAMES = ("state_dict", "summary")
    FUNCTION = "merge"
    CATEGORY = "model/merging"

    def merge(
        self,
        ckpt_a, ckpt_b,
        alpha_unet, alpha_clip_l, alpha_clip_g, alpha_vae,
        mode_unet, mode_others, attn_filter, clip_merge_mode,
        unet_down, unet_mid, unet_up,
        curve_down, curve_mid, curve_up, curve_power,
        spice, target_dtype,
    ):
        # Resolve absolute paths
        path_a = _resolve_checkpoint_path(ckpt_a)
        path_b = _resolve_checkpoint_path(ckpt_b)

        # Load state dicts
        sd_a = _load_state_dict_any(path_a)
        sd_b = _load_state_dict_any(path_b)

        # Merge to in-memory dict
        merged_sd = merge_sdxl(
            sd_a, sd_b,
            alpha_unet=alpha_unet,
            alpha_clip_l=alpha_clip_l,
            alpha_clip_g=alpha_clip_g,
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
        )

        # Optional dtype cast
        dtype = _dtype_from_name(target_dtype)
        if dtype is not None:
            merged_sd = _cast_state_dict(merged_sd, dtype)

        # Summary string
        summary = (
            f"A: {os.path.basename(path_a)} | B: {os.path.basename(path_b)} | "
            f"unet={alpha_unet:.2f} (down={unet_down:.2f}, mid={unet_mid:.2f}, up={unet_up:.2f}, curves={curve_down}/{curve_mid}/{curve_up}, p={curve_power}) | "
            f"clip(L={alpha_clip_l:.2f}, G={alpha_clip_g:.2f}, mode={clip_merge_mode}) | "
            f"vae={alpha_vae:.2f} | mode_unet={mode_unet} | mode_others={mode_others} | attn_filter={attn_filter} | spice={spice:.2f} | dtype={target_dtype}"
        )

        return (merged_sd, summary)


class SDXLFromStateDict:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state_dict": ("STATE_DICT", {}),
            },
            "optional": {
                "preview": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "instantiate"
    CATEGORY = "model/merging"

    def instantiate(self, state_dict, preview=True):
        from comfy.sd import load_checkpoint_guess_config
        import tempfile, os, time
        # Always unique temp path
        tmp_dir = tempfile.mkdtemp(prefix="sdxl_merge_")
        tmp_path = os.path.join(tmp_dir, f"{time.time_ns()}.safetensors")
        _save_state_dict_safetensors(tmp_path, state_dict)
        try:
            model, clip, vae, _ = load_checkpoint_guess_config(tmp_path, output_vae=True, output_clip=True)
            return (model, clip, vae)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except Exception:
                pass


class SDXLMergedSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state_dict": ("STATE_DICT", {}),
                "file_name": ("STRING", {"default": "sdxl_merged"}),
                "subfolder": ("STRING", {"default": "merged"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "model/merging"

    def save(self, state_dict, file_name, subfolder):
        ckpt_dirs = paths.get_folder_paths("checkpoints")
        out_root = ckpt_dirs[0] if ckpt_dirs else "."
        out_dir = os.path.join(out_root, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(out_dir, f"{file_name}_{stamp}.safetensors")
        _save_state_dict_safetensors(out_path, state_dict)
        return (out_path,)


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SDXLDualMerger": SDXLDualMerger,
    "SDXLFromStateDict": SDXLFromStateDict,
    "SDXLMergedSaver": SDXLMergedSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLDualMerger": "SDXL Dual Model + Dual CLIP Merger (v2)",
    "SDXLFromStateDict": "Instantiate SDXL from STATE_DICT",
    "SDXLMergedSaver": "Save SDXL Merged Checkpoint",
}