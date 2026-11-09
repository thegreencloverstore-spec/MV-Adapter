"""CPU checks for the low-memory handoff.

These checks do not load Stable Diffusion and do not use a GPU. A passing run
only means the OpenCV inpaint fallback and the GLB texture writer work. It is
not a texture-quality result.
"""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _neutral_mesh():
    import trimesh

    mesh = trimesh.creation.cylinder(radius=0.35, height=1.2, sections=48)
    vertices = np.asarray(mesh.vertices)
    theta = np.arctan2(vertices[:, 0], vertices[:, 2])
    u = (theta + np.pi) / (2.0 * np.pi)
    height = vertices[:, 1]
    span = float(height.max() - height.min()) + 1e-8
    v = (height - height.min()) / span
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[180, 180, 180, 255],
        metallicFactor=0.0,
        roughnessFactor=0.9,
        name="neutral_gray",
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.stack([u, v], axis=1).astype(np.float32),
        material=material,
    )
    return mesh


def _reference_image(size: int = 512) -> Image.Image:
    image = Image.new("RGB", (size, size), (127, 127, 127))
    pixels = image.load()
    left, right = int(size * 0.32), int(size * 0.68)
    top, bottom = int(size * 0.12), int(size * 0.88)
    sand = (217, 199, 163)
    slate = (76, 106, 138)
    for y in range(top, bottom):
        band = (y - top) / float(bottom - top)
        color = sand if band < 0.28 or band > 0.72 else slate
        for x in range(left, right):
            pixels[x, y] = color
    return image


def _checker(size: int = 64) -> Image.Image:
    image = Image.new("RGB", (size, size), (217, 199, 163))
    pixels = image.load()
    slate = (76, 106, 138)
    for y in range(size // 2, size):
        for x in range(size):
            pixels[x, y] = slate
    return image


def _run_blender(src: Path, dst: Path) -> str:
    blender = shutil.which("blender")
    if blender is None:
        return "skipped: blender is not on PATH"
    script = """
import bpy, sys
src, dst = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
if len(bpy.data.materials) < 1:
    raise SystemExit("imported GLB has no material")
if len(bpy.data.images) < 1:
    raise SystemExit("imported GLB has no image")
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
"""
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "roundtrip.py"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            [
                blender,
                "--background",
                "--python",
                str(script_path),
                "--",
                str(src),
                str(dst),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode != 0:
        tail = (completed.stdout + "\n" + completed.stderr)[-2000:]
        raise RuntimeError(f"Blender roundtrip failed:\n{tail}")
    return f"completed with {blender}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emit-fixtures",
        default="",
        help="Directory for the neutral cylinder GLB and reference PNG.",
    )
    args = parser.parse_args()

    lowmem = _load("mvadapter_lowmem_standalone", "mvadapter/utils/lowmem.py")
    cv_ops = _load("mvadapter_cv_ops_standalone", "mvadapter/utils/mesh_utils/cv_ops.py")
    gltf_export = _load(
        "mvadapter_gltf_export_standalone", "mvadapter/utils/mesh_utils/gltf_export.py"
    )

    checks = []

    warning = lowmem.validate_square_resolution(512, 512)
    assert warning is None
    warning_384 = lowmem.validate_square_resolution(384, 384)
    assert warning_384 and "384" in warning_384
    for bad in ((512, 768), (500, 500), (768, 768), (128, 128)):
        try:
            lowmem.validate_square_resolution(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"resolution {bad} should have been rejected")
    checks.append("resolution guards accept 512 and 384 and reject rectangles, 768, and 128")

    import os

    os.environ["MVADAPTER_CV_BACKEND"] = "opencv"
    assert cv_ops.active_cv_backend() == "opencv"
    import torch

    image = torch.ones(32, 32, 3)
    image[12:20, 12:20] = 0
    mask = torch.zeros(32, 32, dtype=torch.bool)
    mask[12:20, 12:20] = True
    filled = cv_ops.inpaint_cvc(image, mask, 3)
    outside = filled[~mask]
    inside = filled[mask]
    if not torch.allclose(outside, torch.ones_like(outside)):
        raise AssertionError("OpenCV inpaint changed pixels outside the mask")
    if float(inside.mean()) < 0.2:
        raise AssertionError("OpenCV inpaint left the hole near black")
    eroded = cv_ops.batch_erode(mask[None].float(), kernel_size=3)
    if int(eroded.sum()) >= int(mask.sum()):
        raise AssertionError("erode fallback did not shrink the mask")
    checks.append("opencv telea inpaint preserves known pixels and fills a hole; erode shrinks a mask")

    import trimesh

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bare = _neutral_mesh()
        bare_path = tmp_path / "neutral_cylinder.glb"
        bare.export(bare_path)
        reloaded = trimesh.load(bare_path, force="mesh", process=False)
        if reloaded.faces.shape[0] < 10:
            raise AssertionError("cylinder GLB has too few faces")
        if reloaded.visual.uv is None or len(reloaded.visual.uv) != len(reloaded.vertices):
            raise AssertionError("cylinder GLB is missing per-vertex UVs")

        texture = _checker()
        textured_path = tmp_path / "textured.glb"
        gltf_export.embed_textures_in_glb(
            str(bare_path),
            str(textured_path),
            texture,
            task_id="selftest",
        )
        painted = trimesh.load(textured_path, force="mesh", process=False)
        base = painted.visual.material.baseColorTexture
        if base is None:
            raise AssertionError("textured GLB has no baseColorTexture")
        array = np.asarray(base.convert("RGB"), dtype=np.float32)
        split = array.shape[0] // 2
        top = array[:split].mean(axis=(0, 1))
        bottom = array[split:].mean(axis=(0, 1))
        # Sand is the high-red band. Slate is the darker lower band.
        # JPEG is lossy, so this checks the bands survived, not exact bytes.
        if top[0] < 180 or bottom[0] > 140:
            raise AssertionError(
                f"texture colors did not survive JPEG embedding: top {top}, bottom {bottom}"
            )
        blender_note = _run_blender(textured_path, tmp_path / "blender_roundtrip.glb")
        if blender_note.startswith("completed"):
            roundtrip = trimesh.load(
                tmp_path / "blender_roundtrip.glb", force="mesh", process=False
            )
            if roundtrip.visual.material.baseColorTexture is None:
                raise AssertionError("Blender re-export dropped the base-color texture")
            checks.append("blender import/export kept the base-color texture")
        else:
            checks.append(blender_note)

        if args.emit_fixtures:
            dest = Path(args.emit_fixtures)
            dest.mkdir(parents=True, exist_ok=True)
            bare.export(dest / "neutral_cylinder.glb")
            _reference_image().save(dest / "neutral_reference.png")
            checks.append(f"wrote fixtures to {dest}")

    report = {
        "kind": "cpu_selftest",
        "gpu_inference": False,
        "checks": checks,
        "authoring_hardware_note": (
            "This process does not report a GPU memory measurement."
        ),
    }
    print(json.dumps(report, indent=2))
    print("SELFTEST: passed CPU export and inpaint checks. Diffusion was not run.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"SELFTEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
