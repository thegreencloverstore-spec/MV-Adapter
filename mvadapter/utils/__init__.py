"""Lazy exports. Importing this package must not import nvdiffrast."""

import importlib

_EXPORTS = {
    "get_camera": (".mesh_utils.camera", "get_camera"),
    "get_orthogonal_camera": (".mesh_utils.camera", "get_orthogonal_camera"),
    "get_plucker_embeds_from_cameras_ortho": (
        ".geometry",
        "get_plucker_embeds_from_cameras_ortho",
    ),
    "make_image_grid": (".saving", "make_image_grid"),
    "tensor_to_image": (".saving", "tensor_to_image"),
    "image_to_tensor": (".saving", "image_to_tensor"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
