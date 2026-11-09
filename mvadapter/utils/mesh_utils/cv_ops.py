"""Image morphology and inpaint used by UV padding.

The upstream baker calls NVIDIA CV-CUDA. CV-CUDA does not publish a native
Windows wheel (upstream MV-Adapter issue 51; CV-CUDA issue 92). When CV-CUDA
is missing, or when ``MVADAPTER_CV_BACKEND=opencv``, these functions use
OpenCV's Telea inpaint and a PyTorch morphology fallback.

OpenCV ``INPAINT_TELEA`` implements the same Telea 2004 algorithm that
``cvcuda.inpaint`` documents. The two are not bit-identical.
"""

from typing import Optional

import torch
import torch.nn.functional as F

_BACKEND_ENV = "MVADAPTER_CV_BACKEND"


def active_cv_backend() -> str:
    import os

    choice = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    if choice not in {"auto", "cvcuda", "opencv"}:
        raise ValueError(
            f"{_BACKEND_ENV} must be auto, cvcuda, or opencv, got {choice!r}."
        )
    if choice == "opencv":
        return "opencv"
    cvcuda = _import_cvcuda()
    if cvcuda is not None:
        return "cvcuda"
    if choice == "cvcuda":
        raise ImportError(
            "MVADAPTER_CV_BACKEND=cvcuda but the cvcuda module is not importable. "
            "Native Windows has no official CV-CUDA wheel. Leave the variable unset "
            "or set MVADAPTER_CV_BACKEND=opencv to use the OpenCV Telea fallback."
        )
    return "opencv"


def _import_cvcuda():
    try:
        import cvcuda
    except Exception:
        return None
    return cvcuda


def _as_uint8_image(image: torch.Tensor) -> torch.Tensor:
    if image.dtype == torch.uint8:
        return image
    return (image.clamp(0, 1) * 255).round().to(torch.uint8)


def _as_uint8_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.uint8:
        return mask
    if mask.dtype == torch.bool:
        return mask.to(torch.uint8) * 255
    return (mask > 0).to(torch.uint8) * 255


def _inpaint_cvcuda(
    image_u8: torch.Tensor,
    mask_u8: torch.Tensor,
    padding_size: int,
    cvcuda,
):
    image_cvc = cvcuda.as_tensor(image_u8, "HWC")
    mask_cvc = cvcuda.as_tensor(mask_u8, "HW")
    output_cvc = cvcuda.inpaint(image_cvc, mask_cvc, padding_size)
    return torch.tensor(output_cvc.cuda(), device=image_u8.device)


def _inpaint_opencv(
    image_u8: torch.Tensor, mask_u8: torch.Tensor, padding_size: int
) -> torch.Tensor:
    try:
        import cv2
    except Exception as exc:
        raise ImportError(
            "UV padding needs either CV-CUDA or opencv-python. "
            "Install opencv-python in the isolated MV-Adapter environment."
        ) from exc

    if int(padding_size) <= 0:
        raise ValueError("inpaint radius must be a positive integer.")

    image_np = image_u8.detach().cpu().numpy()
    mask_np = mask_u8.detach().cpu().numpy()
    if image_np.ndim != 3 or image_np.shape[-1] not in (1, 3):
        raise ValueError(
            f"OpenCV inpaint expected HWC with 1 or 3 channels, got {image_np.shape}."
        )
    if mask_np.shape != image_np.shape[:2]:
        raise ValueError(
            f"Inpaint mask shape {mask_np.shape} does not match image {image_np.shape[:2]}."
        )

    hole = mask_np > 0
    if not hole.any():
        filled = image_np
    else:
        source = image_np if image_np.shape[-1] == 3 else image_np[:, :, 0]
        painted = cv2.inpaint(
            source,
            (hole.astype("uint8") * 255),
            int(padding_size),
            cv2.INPAINT_TELEA,
        )
        if painted.ndim == 2:
            painted = painted[:, :, None]
        filled = image_np.copy()
        filled[hole] = painted[hole]

    return torch.from_numpy(filled).to(device=image_u8.device)


def inpaint_cvc(
    image: torch.Tensor,
    mask: torch.Tensor,
    padding_size: int,
    return_dtype: Optional[torch.dtype] = None,
):
    input_dtype = image.dtype
    input_device = image.device
    image_u8 = _as_uint8_image(image.detach())
    mask_u8 = _as_uint8_mask(mask.detach())

    backend = active_cv_backend()
    if backend == "cvcuda" and image_u8.is_cuda and mask_u8.is_cuda:
        output = _inpaint_cvcuda(image_u8, mask_u8, padding_size, _import_cvcuda())
    else:
        output = _inpaint_opencv(image_u8, mask_u8, padding_size)

    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return output.to(device=input_device, dtype=torch.uint8)
    return output.to(device=input_device, dtype=input_dtype) / 255.0


def batch_inpaint_cvc(
    images: torch.Tensor,
    masks: torch.Tensor,
    padding_size: int,
    return_dtype: Optional[torch.dtype] = None,
):
    output = torch.stack(
        [
            inpaint_cvc(image, mask, padding_size, return_dtype)
            for (image, mask) in zip(images, masks)
        ],
        axis=0,
    )
    return output


def _torch_binary_morph(masks_u8: torch.Tensor, kernel_size: int, erode: bool):
    if kernel_size % 2 != 1:
        raise ValueError(
            "The PyTorch morphology fallback expects an odd kernel size. "
            f"Got {kernel_size}."
        )
    values = (masks_u8 > 0).to(dtype=torch.float32).unsqueeze(1)
    if erode:
        values = 1.0 - values
    pad = kernel_size // 2
    pooled = F.max_pool2d(values, kernel_size=kernel_size, stride=1, padding=pad)
    if erode:
        pooled = 1.0 - pooled
    return pooled[:, 0] > 0.5


def _morph_cvcuda(masks: torch.Tensor, kernel_size: int, erode: bool):
    cvcuda = _import_cvcuda()
    kind = cvcuda.MorphologyType.ERODE if erode else cvcuda.MorphologyType.DILATE
    masks_cvc = cvcuda.as_tensor(masks[..., None], "NHWC")
    morphed = cvcuda.morphology(
        masks_cvc,
        kind,
        maskSize=(kernel_size, kernel_size),
        anchor=(-1, -1),
    )
    return torch.tensor(morphed.cuda(), device=masks.device)[..., 0]


def _batch_morph(
    masks: torch.Tensor, kernel_size: int, erode: bool, return_dtype: Optional[torch.dtype]
):
    input_dtype = masks.dtype
    input_device = masks.device
    masks_u8 = _as_uint8_mask(masks.detach())
    use_cvcuda = (
        active_cv_backend() == "cvcuda" and masks_u8.is_cuda and masks_u8.ndim == 3
    )
    if use_cvcuda:
        morphed = _morph_cvcuda(masks_u8, kernel_size, erode=erode)
    else:
        morphed = _torch_binary_morph(masks_u8, kernel_size, erode=erode)
        morphed = morphed.to(dtype=torch.uint8) * 255

    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return morphed.to(device=input_device, dtype=torch.uint8)
    return (morphed > 0).to(device=input_device, dtype=input_dtype)


def batch_erode(
    masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None
):
    return _batch_morph(masks, kernel_size, erode=True, return_dtype=return_dtype)


def batch_dilate(
    masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None
):
    return _batch_morph(masks, kernel_size, erode=False, return_dtype=return_dtype)
