"""Image-and-geometry texture generation with an explicit low-memory schedule.

This is the SD2.1 ig2mv model (``mvadapter_ig2mv_sd21.safetensors``) plus the
existing multiview bake. It is not a recolor, a reference projection, or an
SDXL run. SDXL ig2mv is refused here: upstream documents it as a >16 GB path,
and the SDXL UNet weights alone are about 5 GB in fp16.

Nothing in this file measures an 8 GB card. The JSON report records the GPU
that actually ran it. A machine without CUDA exits before any weight download.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

from mvadapter.utils.lowmem import (
    cuda_peak_mib,
    release_cuda_module,
    validate_square_resolution,
)
from mvadapter.utils.mesh_utils.cv_ops import active_cv_backend

REPO_ROOT = Path(__file__).resolve().parents[1]

PRESETS = {
    "8gb": {
        "height": 512,
        "width": 512,
        "offload": "model",
        "split_cfg": True,
        "vae_tiling": True,
        "uv_size": 2048,
        "view_upscale": True,
        "uv_unwarp": True,
        "scheduler": "ddpm",
        "steps": 50,
        "context_type": "cuda",
        "staged_checkpoints": True,
    },
    "8gb-tight": {
        "height": 384,
        "width": 384,
        "offload": "sequential",
        "split_cfg": True,
        "vae_tiling": True,
        "uv_size": 1024,
        "view_upscale": False,
        "uv_unwarp": True,
        "scheduler": "ddpm",
        "steps": 50,
        "context_type": "cuda",
        "staged_checkpoints": True,
    },
}


def _gpu_facts():
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "gpu_name": None,
            "gpu_total_vram_mib": None,
            "device_count": 0,
        }
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_vram_mib": round(props.total_memory / (1024 * 1024), 1),
        "device_count": torch.cuda.device_count(),
    }


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Report written to {path}")


def _stage_remove_bg(image_path: str, out_path: Path, height: int, width: int, device: str):
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    from scripts.inference_ig2mv_sd import preprocess_image, remove_bg

    print("Stage background removal: loading ZhengPeng7/BiRefNet, then freeing it.")
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    birefnet.to(device)
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    cutout = remove_bg(image, birefnet, transform_image, device)
    prepared = preprocess_image(cutout, height, width)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.save(out_path)
    del birefnet, cutout, image
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SD2.1 image+geometry multiview texture with CPU offload and split CFG. "
            "Does not claim an 8 GB measurement."
        )
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="8gb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mesh", default=str(REPO_ROOT / "assets/lowmem/neutral_cylinder.glb"))
    parser.add_argument(
        "--image", default=str(REPO_ROOT / "assets/lowmem/neutral_reference.png")
    )
    parser.add_argument("--text", default="high quality")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default=str(REPO_ROOT / "outputs" / "lowmem_neutral"))
    parser.add_argument("--save-name", default="neutral")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--offload", choices=["none", "model", "sequential"], default=None)
    parser.add_argument("--uv-size", type=int, default=None)
    parser.add_argument("--context", choices=["cuda", "gl"], default=None)
    parser.add_argument("--remove-bg", action="store_true")
    parser.add_argument("--keep-uv", action="store_true")
    parser.add_argument("--no-upscale", action="store_true")
    parser.add_argument("--preprocess-mesh", action="store_true")
    parser.add_argument(
        "--multiview-only",
        action="store_true",
        help="Stop after the six generated views. Skips UV bake, LaMa, and RealESRGAN.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Upstream ig2mv default is 3.0. Values below 1 disable CFG.",
    )
    args = parser.parse_args()

    preset = dict(PRESETS[args.preset])
    height = args.height or preset["height"]
    width = args.width or preset["width"]
    offload = args.offload or preset["offload"]
    steps = args.steps or preset["steps"]
    uv_size = args.uv_size or preset["uv_size"]
    context_type = args.context or preset["context_type"]
    view_upscale = preset["view_upscale"] and not args.no_upscale
    uv_unwarp = preset["uv_unwarp"] and not args.keep_uv
    split_cfg = preset["split_cfg"] and args.guidance_scale > 1.0

    warning = validate_square_resolution(height, width)
    if warning:
        print("WARNING:", warning)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / f"{args.save_name}_report.json"
    report = {
        "experiment": "mvadapter-ig2mv-sd21-lowmem",
        "status": "started",
        "what_this_is": (
            "SD2.1 image-and-geometry multiview diffusion, then projection of "
            "those generated views into a UV texture. Not a flat color fill."
        ),
        "hardware": _gpu_facts(),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": sys.platform,
        "settings": {
            "preset": args.preset,
            "height": height,
            "width": width,
            "offload": offload,
            "split_cfg": split_cfg,
            "steps": steps,
            "scheduler": preset["scheduler"],
            "guidance_scale": args.guidance_scale,
            "uv_size": uv_size,
            "view_upscale": view_upscale,
            "uv_unwarp": uv_unwarp,
            "context_type": context_type,
            "staged_checkpoints": preset["staged_checkpoints"],
            "remove_bg": args.remove_bg,
            "multiview_only": args.multiview_only,
            "base_model": "stabilityai/stable-diffusion-2-1-base",
            "adapter": "huanngzh/mv-adapter/mvadapter_ig2mv_sd21.safetensors",
            "cv_backend": None,
            "mesh": str(Path(args.mesh).resolve()) if Path(args.mesh).exists() else args.mesh,
            "image": args.image,
            "text": args.text,
            "seed": args.seed,
        },
        "stages": [],
        "outputs": {},
        "warnings": [warning] if warning else [],
        "not_measured_here": [
            "No result in this file is an 8 GB certification unless hardware.gpu_name is an 8 GB GPU and status is completed on that machine.",
        ],
    }

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        report["status"] = "blocked_no_cuda"
        report["error"] = (
            "CUDA is not available in this process. Refusing to download weights "
            "or run a CPU imitation of the diffusion model."
        )
        _write_report(report_path, report)
        print(report["error"], file=sys.stderr)
        return 2

    if not Path(args.mesh).is_file():
        report["status"] = "blocked_missing_mesh"
        report["error"] = f"Mesh not found: {args.mesh}"
        _write_report(report_path, report)
        print(report["error"], file=sys.stderr)
        return 2
    if not Path(args.image).is_file():
        report["status"] = "blocked_missing_image"
        report["error"] = f"Reference image not found: {args.image}"
        _write_report(report_path, report)
        print(report["error"], file=sys.stderr)
        return 2

    upscaler = REPO_ROOT / "checkpoints" / "RealESRGAN_x2plus.pth"
    lama = REPO_ROOT / "checkpoints" / "big-lama.pt"
    if not args.multiview_only:
        needed = [lama]
        if view_upscale:
            needed.append(upscaler)
        missing = [str(p) for p in needed if not p.is_file()]
        if missing:
            report["status"] = "blocked_missing_checkpoints"
            report["error"] = (
                "Texture bake checkpoints are missing. Download them before the "
                "GPU run, or pass --multiview-only to test generation first. Missing: "
                + ", ".join(missing)
            )
            _write_report(report_path, report)
            print(report["error"], file=sys.stderr)
            return 2

    os.environ["MVADAPTER_RASTER_BACKEND"] = context_type
    try:
        report["settings"]["cv_backend"] = active_cv_backend()
    except Exception as exc:
        report["settings"]["cv_backend"] = f"unavailable: {exc}"

    print("Effective settings:")
    print(json.dumps(report["settings"], indent=2))
    print(
        "GPU:",
        report["hardware"]["gpu_name"],
        f"({report['hardware']['gpu_total_vram_mib']} MiB total)",
    )

    image_for_model = args.image
    pipe = None
    try:
        if args.remove_bg:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            image_for_model = _stage_remove_bg(
                args.image,
                save_dir / f"{args.save_name}_cutout.png",
                height,
                width,
                args.device,
            )
            report["stages"].append(
                {
                    "name": "background_removal",
                    "seconds": round(time.perf_counter() - started, 3),
                    "peak_allocated_mib": None
                    if cuda_peak_mib() is None
                    else round(cuda_peak_mib(), 1),
                }
            )
            report["outputs"]["cutout"] = image_for_model

        try:
            from mvadapter.pipelines.pipeline_texture import (
                ModProcessConfig,
                TexturePipeline,
            )
            from mvadapter.utils import make_image_grid
            from scripts.inference_ig2mv_sd import prepare_pipeline, run_pipeline
        except ImportError as exc:
            report["status"] = "blocked_import"
            report["error"] = (
                "Could not import the geometry-conditioned pipeline. nvdiffrast "
                "has to be built in this environment before a real run. "
                f"Detail: {exc}"
            )
            _write_report(report_path, report)
            print(report["error"], file=sys.stderr)
            return 2

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        print("Stage diffusion: loading SD2.1 ig2mv. This downloads weights on first run.")
        pipe = prepare_pipeline(
            base_model="stabilityai/stable-diffusion-2-1-base",
            vae_model=None,
            unet_model=None,
            lora_model=None,
            adapter_path="huanngzh/mv-adapter",
            scheduler=preset["scheduler"],
            num_views=6,
            device=args.device,
            dtype=torch.float16,
            offload=offload,
            vae_tiling=preset["vae_tiling"],
        )
        generator_device = "cpu" if offload != "none" else args.device
        images, pos_images, normal_images, reference_image = run_pipeline(
            pipe,
            mesh_path=args.mesh,
            num_views=6,
            text=args.text,
            image=image_for_model,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            reference_conditioning_scale=1.0,
            negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
            device=args.device,
            remove_bg_fn=None,
            context_type=context_type,
            split_cfg=split_cfg,
            generator_device=generator_device,
        )
        mv_path = save_dir / f"{args.save_name}.png"
        make_image_grid(images, rows=1).save(mv_path)
        make_image_grid(pos_images, rows=1).save(save_dir / f"{args.save_name}_pos.png")
        make_image_grid(normal_images, rows=1).save(save_dir / f"{args.save_name}_nor.png")
        reference_image.save(save_dir / f"{args.save_name}_reference.png")
        report["stages"].append(
            {
                "name": "ig2mv_sd21",
                "seconds": round(time.perf_counter() - started, 3),
                "peak_allocated_mib": round(cuda_peak_mib(), 1),
            }
        )
        report["outputs"]["multiview"] = str(mv_path)
        print(f"Multiview image saved to {mv_path}")

        release_cuda_module(pipe)
        pipe = None

        if args.multiview_only:
            report["status"] = "completed_multiview_only"
            report["note"] = (
                "Generation finished. Texture bake was not run. "
                "Inspect the six views before spending the bake step."
            )
            _write_report(report_path, report)
            return 0

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        print("Stage bake: project the generated views. Diffusion weights were released.")
        texture_pipe = TexturePipeline(
            upscaler_ckpt_path=str(upscaler) if view_upscale else None,
            inpaint_ckpt_path=str(lama),
            device=args.device,
            context_type=context_type,
            staged_checkpoints=preset["staged_checkpoints"],
            # The bake calls cam_proj with poisson_blending=False. torch-native
            # avoids JIT-compiling the unused CUDA Poisson solver on Windows.
            pb_backend="torch-native",
        )
        out = texture_pipe(
            mesh_path=args.mesh,
            save_dir=str(save_dir),
            save_name=args.save_name,
            uv_unwarp=uv_unwarp,
            preprocess_mesh=args.preprocess_mesh,
            uv_size=uv_size,
            rgb_path=str(mv_path),
            rgb_process_config=ModProcessConfig(
                view_upscale=view_upscale, inpaint_mode="view"
            ),
            camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        )
        report["stages"].append(
            {
                "name": "texture_bake",
                "seconds": round(time.perf_counter() - started, 3),
                "peak_allocated_mib": round(cuda_peak_mib(), 1),
            }
        )
        report["outputs"]["shaded_glb"] = out.shaded_model_save_path
        report["status"] = "completed"
        print(f"Textured GLB: {out.shaded_model_save_path}")
        _write_report(report_path, report)
        return 0
    except RuntimeError as exc:
        oom = "out of memory" in str(exc).lower()
        report["status"] = "cuda_oom" if oom else "failed"
        report["error"] = str(exc)
        if oom:
            report["next_step"] = (
                "If this was preset 8gb at 512, rerun with --preset 8gb-tight. "
                "That uses 384 and a smaller UV. It is a quality tradeoff, not a "
                "second model. If 384 also OOMs, stop. Do not describe a partial "
                "file as a finished texture."
            )
            print(report["next_step"], file=sys.stderr)
        _write_report(report_path, report)
        raise
    finally:
        if pipe is not None:
            release_cuda_module(pipe)


if __name__ == "__main__":
    sys.exit(main())
