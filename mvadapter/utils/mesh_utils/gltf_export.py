"""Embed textures into an existing GLB without importing the rasterizer.

``mesh.py`` imports nvdiffrast, which is unavailable until that package is
built on the target machine. The texture write itself only needs gltflib.
"""

import io
from typing import Optional

from PIL import Image


def embed_textures_in_glb(
    input_path: str,
    output_path: str,
    texture: Image.Image,
    metallic_roughness_texture: Optional[Image.Image] = None,
    normal_texture: Optional[Image.Image] = None,
    texture_format: str = "JPEG",
    normal_strength: float = 1.0,
    task_id: str = "",
) -> None:
    from gltflib import GLTF
    from gltflib import Image as gltfImage
    from gltflib import NormalTextureInfo, Texture, TextureInfo

    def add_texture(gltf: GLTF, image: Image.Image) -> int:
        buffer = io.BytesIO()
        image.save(buffer, format=texture_format)
        payload = bytearray(buffer.getvalue())
        _, offset, bytelen = gltf._create_or_extend_glb_resource(payload)
        buffer_view = gltf._create_embedded_image_buffer_view(offset, bytelen)
        image_entry = gltfImage(mimeType="image/jpeg", bufferView=buffer_view)
        if gltf.model.images:
            gltf.model.images.append(image_entry)
        else:
            gltf.model.images = [image_entry]
        texture_entry = Texture(source=len(gltf.model.images) - 1)
        if gltf.model.textures:
            gltf.model.textures.append(texture_entry)
        else:
            gltf.model.textures = [texture_entry]
        return len(gltf.model.textures) - 1

    gltf = GLTF.load(input_path)
    if not gltf.model.materials:
        raise ValueError(
            "GLB has no material. The baker writes baseColorTexture onto "
            "materials[0]. Export the mesh with a PBR material, or re-unwrap it "
            "so the intermediate mesh contains one."
        )
    material = gltf.model.materials[0]
    pbr = material.pbrMetallicRoughness
    if pbr is None:
        raise ValueError(
            "GLB material 0 has no pbrMetallicRoughness block, so a base-color "
            "texture cannot be attached. Re-export with a PBR material."
        )

    if texture.mode != "RGB":
        texture = texture.convert("RGB")

    pbr.baseColorTexture = TextureInfo(index=add_texture(gltf, texture))
    pbr.baseColorFactor = [1, 1, 1, 1]

    if metallic_roughness_texture is not None:
        pbr.metallicRoughnessTexture = TextureInfo(
            index=add_texture(gltf, metallic_roughness_texture.convert("RGB"))
        )
        pbr.metallicFactor = 1.0
        pbr.roughnessFactor = 1.0
    else:
        pbr.metallicFactor = 0.0
        pbr.roughnessFactor = 0.9

    if normal_texture is not None:
        material.normalTexture = NormalTextureInfo(
            index=add_texture(gltf, normal_texture.convert("RGB")),
            scale=normal_strength,
        )

    if gltf.model.nodes:
        gltf.model.nodes[0].name = f"mvadapter_node_{task_id}"
    if gltf.model.meshes:
        gltf.model.meshes[0].name = f"mvadapter_mesh_{task_id}"
    material.name = f"mvadapter_material_{task_id}"
    if gltf.model.asset is not None:
        gltf.model.asset.generator = "https://github.com/huanngzh/MV-Adapter"
    gltf.export(output_path)
