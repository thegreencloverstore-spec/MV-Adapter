"""Report whether this machine can start the isolated low-memory experiment.

Does not download weights and does not run the diffusion model.
"""

import importlib
import json
import platform
import sys


CHECKS = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "peft",
    "einops",
    "cv2",
    "trimesh",
    "gltflib",
    "open3d",
    "pymeshlab",
    "spandrel",
    "nvdiffrast",
    "nvdiffrast.torch",
    "cvcuda",
]


def _probe(name: str):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"import": False, "detail": f"{type(exc).__name__}: {exc}"}
    version = getattr(module, "__version__", None)
    return {"import": True, "version": version}


def main():
    torch_info = _probe("torch")
    cuda = {
        "available": False,
        "gpu_name": None,
        "gpu_total_vram_mib": None,
        "torch_cuda": None,
    }
    if torch_info["import"]:
        import torch

        cuda["torch_cuda"] = torch.version.cuda
        cuda["available"] = bool(torch.cuda.is_available())
        if cuda["available"]:
            props = torch.cuda.get_device_properties(0)
            cuda["gpu_name"] = torch.cuda.get_device_name(0)
            cuda["gpu_total_vram_mib"] = round(props.total_memory / (1024 * 1024), 1)

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda": cuda,
        "imports": {name: _probe(name) for name in CHECKS},
        "ready_for_gpu_test": bool(cuda["available"] and torch_info["import"]),
        "notes": [
            "cvcuda is optional. The low-memory path uses OpenCV Telea when it is absent.",
            "nvdiffrast is required for geometry conditioning. A failed import blocks the real test.",
            "This script does not download Stable Diffusion weights.",
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if report["ready_for_gpu_test"]:
        print("PREFLIGHT: CUDA is visible. This does not mean the 8 GB run fits.")
        return 0
    print("PREFLIGHT: CUDA is not available. Do not start the weight download here.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
