"""Memory placement for the SD2.1 image-and-geometry pipeline.

The multiview UNet denoises all six views together. Row/column attention reads
views 0-3 as a row and views 0,2,4,5 as a column, so views cannot be generated
one at a time without changing the model. Classifier-free guidance can be split
into two six-view forwards. That preserves the guidance formula and cuts the
UNet batch from 12 to 6. It does not change the trained weights.
"""

from typing import Optional

import torch


TRAINED_SD21_SIDE = 512


def validate_square_resolution(
    height: int,
    width: int,
    trained: int = TRAINED_SD21_SIDE,
    max_side: int = TRAINED_SD21_SIDE,
) -> Optional[str]:
    """Reject resolutions the row/column attention reshape cannot represent.

    Returns a warning string when the resolution is valid but is not the
    training resolution. The SD2.1 geometry adapter was trained at 512.
    """

    if height != width:
        raise ValueError(
            f"Image-geometry MV-Adapter attention requires a square resolution, "
            f"got {height}x{width}. The row/column rearrange takes the integer "
            "square root of each feature-map length and will mis-shape a rectangle."
        )
    if height % 8 != 0:
        raise ValueError(
            f"Resolution must be divisible by 8 so the VAE latent is an integer "
            f"grid, got {height}."
        )
    if height < 256:
        raise ValueError(
            f"Resolution {height} is below the minimum this low-memory path allows "
            "(256). Smaller grids still run the same network, but they are not a "
            "supported test setting."
        )
    if height > max_side:
        raise ValueError(
            f"Resolution {height} is above {max_side}. The SD2.1 image-geometry "
            "weights were trained at 512. 768 is the SDXL setting and is not a "
            "low-memory mode for these weights."
        )
    if height != trained:
        return (
            f"Resolution {height} is not the SD2.1 training resolution {trained}. "
            "The network can run because its convolutions and square attention "
            "reshape accept this grid. Color fidelity relative to 512 was not "
            "measured. Use 512 unless 512 exhausts GPU memory."
        )
    return None


def place_pipeline(
    pipe,
    device: str,
    dtype: torch.dtype,
    offload: str = "none",
    vae_tiling: bool = False,
) -> str:
    """Move an already-loaded pipeline. ``offload='none'`` matches the stock scripts.

    ``cond_encoder`` is not one of the diffusers components registered for
    offload hooks, so it stays resident on the GPU. It is the T2I-Adapter that
    ingests the position/normal control images. The UNet, VAE, and text encoder
    participate in the offload chain.
    """

    if offload not in {"none", "model", "sequential"}:
        raise ValueError(f"Unknown offload mode {offload!r}.")

    pipe.enable_vae_slicing()
    if vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if offload == "none":
        pipe.to(device=device, dtype=dtype)
        pipe.cond_encoder.to(device=device, dtype=dtype)
        return "resident fp16 on the requested device, VAE slicing enabled"

    # Cast in place while the weights are still on CPU, then install hooks.
    pipe.to(dtype=dtype)
    pipe.cond_encoder.to(dtype=dtype)
    _enable_offload(pipe, offload, device)
    # The condition encoder is invoked between the reference UNet pass and the
    # denoising loop and is not in model_cpu_offload_seq.
    pipe.cond_encoder.to(device=device, dtype=dtype)

    if offload == "model":
        return (
            "model CPU offload for text encoder, UNet, and VAE; "
            "condition encoder left on GPU; VAE slicing enabled"
        )
    return (
        "sequential CPU offload for diffusers components; "
        "condition encoder left on GPU; VAE slicing enabled"
    )


def _enable_offload(pipe, offload: str, device: str) -> None:
    # diffusers 0.32 takes device=. Older builds take gpu_id=.
    gpu_id = torch.device(device).index or 0
    if offload == "model":
        try:
            pipe.enable_model_cpu_offload(device=device)
        except TypeError:
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        return
    try:
        pipe.enable_sequential_cpu_offload(device=device)
    except TypeError:
        pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)


def release_cuda_module(module) -> None:
    import gc

    if module is None:
        return
    remove_hooks = getattr(module, "remove_all_hooks", None)
    if callable(remove_hooks):
        try:
            remove_hooks()
        except Exception:
            pass
    del module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_peak_mib() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)
