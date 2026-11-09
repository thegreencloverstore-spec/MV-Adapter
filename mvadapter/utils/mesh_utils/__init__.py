"""Lazy exports so `cv_ops` can import without building nvdiffrast.

The rasterizer, mesh loader, and projection stack still load on first use.
"""

import importlib

_EXPORTS = {
    "Camera": (".camera", "Camera"),
    "get_c2w": (".camera", "get_c2w"),
    "get_camera": (".camera", "get_camera"),
    "get_orthogonal_camera": (".camera", "get_orthogonal_camera"),
    "get_orthogonal_projection_matrix": (".camera", "get_orthogonal_projection_matrix"),
    "get_projection_matrix": (".camera", "get_projection_matrix"),
    "TexturedMesh": (".mesh", "TexturedMesh"),
    "load_mesh": (".mesh", "load_mesh"),
    "replace_mesh_texture_and_save": (".mesh", "replace_mesh_texture_and_save"),
    "CameraProjection": (".projection", "CameraProjection"),
    "CameraProjectionOutput": (".projection", "CameraProjectionOutput"),
    "DepthControlNetNormalization": (".render", "DepthControlNetNormalization"),
    "DepthNormalizationStrategy": (".render", "DepthNormalizationStrategy"),
    "NVDiffRastContextWrapper": (".render", "NVDiffRastContextWrapper"),
    "RenderOutput": (".render", "RenderOutput"),
    "SimpleNormalization": (".render", "SimpleNormalization"),
    "Zero123PlusPlusNormalization": (".render", "Zero123PlusPlusNormalization"),
    "render": (".render", "render"),
    "SmartPainter": (".smart_paint", "SmartPainter"),
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
