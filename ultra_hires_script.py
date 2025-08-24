# ultra_hires_script.py
# Ultra HiRes Script — smart HiRes + final output sizing for Ultra Duper Sampler

class UltraHiResScript:
    UPSCALE_TYPES = ["latent", "pixel", "both"]
    LATENT_RESAMPLERS = ["nearest-exact", "bilinear", "bicubic"]
    PIXEL_RESAMPLERS  = ["bicubic", "bilinear"]
    FINAL_OUTPUT_MODES = ["hires_size", "keep_original", "custom"]
    FINAL_RESAMPLERS   = ["bicubic", "bilinear", "nearest"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_type": (cls.UPSCALE_TYPES, {"default": "both"}),

                # internal HiRes sizing — pick ONE method; others 0/ignored
                "upscale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 8.0, "step": 0.05}),
                "target_long_edge": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip":"0 = ignore"}),
                "target_short_edge": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip":"0 = ignore"}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "max_megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.1, "tooltip":"0 = ignore"}),

                # refine
                "hires_steps": ("INT", {"default": 12, "min": 1, "max": 200}),
                "denoise": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 4}),
                "use_same_seed": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                # internal quality knobs
                "latent_resampler": (cls.LATENT_RESAMPLERS, {"default": "nearest-exact"}),
                "pixel_resampler": (cls.PIXEL_RESAMPLERS, {"default": "bicubic"}),

                # beautify (pixel stage only)
                "enable_beautify": ("BOOLEAN", {"default": True}),
                "sharpen_amount": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sharpen_sigma": ("FLOAT", {"default": 0.8, "min": 0.2, "max": 2.5, "step": 0.05}),
                "micro_contrast": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.6, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.6, "step": 0.01}),

                # final output controls (supersample → downscale)
                "final_output": (cls.FINAL_OUTPUT_MODES, {"default": "keep_original"}),
                "final_width": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip":"custom mode only; 0 = auto"}),
                "final_height": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip":"custom mode only; 0 = auto"}),
                "final_long_edge": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip":"custom mode only; 0 = ignore"}),
                "final_resampler": (cls.FINAL_RESAMPLERS, {"default": "bicubic"}),
                "final_antialias": ("BOOLEAN", {"default": True, "tooltip":"use antialias=True on downscale"}),
                "final_downscale_latent": ("BOOLEAN", {"default": False, "tooltip":"also shrink latent to final size"}),
            }
        }

    RETURN_TYPES = ("SCRIPT",)
    RETURN_NAMES = ("script",)
    FUNCTION = "build"
    CATEGORY = "sampling/ultra/scripts"
    OUTPUT_NODE = False
    DESCRIPTION = "Ultra HiRes SCRIPT: smart internal upscaling + optional final downscale to your chosen output size."

    def build(
        self, upscale_type, upscale_by, target_long_edge, target_short_edge, target_width, target_height, max_megapixels,
        hires_steps, denoise, iterations, use_same_seed, seed,
        latent_resampler, pixel_resampler,
        enable_beautify, sharpen_amount, sharpen_sigma, micro_contrast, saturation,
        final_output, final_width, final_height, final_long_edge, final_resampler, final_antialias, final_downscale_latent
    ):
        cfg = {
            "upscale_type": str(upscale_type),

            # internal HiRes sizing
            "upscale_by": float(upscale_by),
            "target_long_edge": int(target_long_edge),
            "target_short_edge": int(target_short_edge),
            "target_width": int(target_width),
            "target_height": int(target_height),
            "max_megapixels": float(max_megapixels),

            # refine
            "hires_steps": int(hires_steps),
            "denoise": float(denoise),
            "iterations": int(iterations),
            "use_same_seed": bool(use_same_seed),
            "seed": int(seed),

            "latent_resampler": str(latent_resampler),
            "pixel_resampler": str(pixel_resampler),

            "beautify": {
                "enable": bool(enable_beautify),
                "sharpen_amount": float(sharpen_amount),
                "sharpen_sigma": float(sharpen_sigma),
                "micro_contrast": float(micro_contrast),
                "saturation": float(saturation),
            },

            # final output controls
            "final": {
                "mode": str(final_output),
                "width": int(final_width),
                "height": int(final_height),
                "long_edge": int(final_long_edge),
                "resampler": str(final_resampler),
                "antialias": bool(final_antialias),
                "downscale_latent": bool(final_downscale_latent),
            }
        }
        wrapped = ("Ultra HiRes Script", cfg)
        return (wrapped,)

NODE_CLASS_MAPPINGS = {"UltraHiResScript": UltraHiResScript}
NODE_DISPLAY_NAME_MAPPINGS = {"UltraHiResScript": "Ultra HiRes Script"}
