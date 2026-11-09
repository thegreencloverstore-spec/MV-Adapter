# Low-memory image-and-geometry texturing

This fork adds an isolated experiment that runs the published SD2.1 image-and-geometry MV-Adapter (`mvadapter_ig2mv_sd21.safetensors`) and then bakes those generated views into a GLB. The goal is a local test on a Windows machine with an RTX 3060 Ti 8 GB. This document is the handoff. It is not a claim that the 8 GB card has already succeeded.

## Status

| Item | State |
| --- | --- |
| Code path | SD2.1 ig2mv, six views, split classifier-free guidance, CPU offload, staged bake |
| Authoring machine | Linux cloud VM, no NVIDIA GPU (`nvidia-smi` absent), about 15 GiB RAM, CPython 3.12.3. Diffusion was not run. |
| 8 GB measurement | Not done. Do not treat a CPU self-test, a mock, or a larger GPU as an 8 GB result. |
| Weights downloaded here | No. `stabilityai/stable-diffusion-2-1-base` returned HTTP 401 without a token on 2026-11-09. |
| TripoSG install | Not modified. Do not point this installer at that interpreter. |
| ComfyUI | Not modified. |
| Upstream GitHub | No issue, comment, or pull request. |

The CPU self-test covers OpenCV inpaint and GLB texture embedding only. The JSON report from a real GPU run is the only place a VRAM number belongs, and only if `hardware.gpu_name` and `hardware.gpu_total_vram_mib` were filled by that run.

Authoring checks on 2026-11-09, Linux, CPython 3.12.3, torch 2.5.1+cpu, diffusers 0.32.2:

- The SD2.1 pipeline class imports, and `split_cfg` is on its call signature.
- The CPU self-test passed: resolution guards, OpenCV Telea inpaint, GLB base-color reload.
- Blender 4.0.2, installed on that Linux VM, imported the synthetic textured GLB and exported it with the base-color image still attached. That GLB was a hand-painted checker, not a diffusion result, and the test was not Windows.
- Preflight exited 2 because CUDA was absent. The texture script exited 2 with `blocked_no_cuda` and did not download weights.
- `open3d`, `pymeshlab`, `spandrel`, and `nvdiffrast` were not installed in that CPU environment, so unwrap and generation were not exercised.

## What the model is

Image-and-geometry generation conditions the diffusion UNet on:

- a reference image, through reference attention
- rendered position and normal maps of the mesh, through the T2I-Adapter condition encoder
- the six orthographic views inside row/column multiview attention

The bake then projects those six generated images onto a UV atlas and fills unseen texels. That projection step is how the published texture pipeline works. It is not a substitute for the diffusion step. This patch does not paint the reference color onto the mesh and call that generation.

The row/column processor is hard-wired to six views (front, right, back, left, top, bottom). The script refuses other view counts and non-square resolutions. Generating one view at a time would drop the multiview attention, so that is not offered as a memory mode.

## Why this is the SD2.1 variant

Upstream README, as shipped in this repo:

- SDXL image-geometry: use when VRAM is greater than 16 GB, 768 resolution
- SD2.1 image-geometry: the path it points to when VRAM is under 10 GB, 512 resolution, DDPM scheduler
- SD2.1 texture script: `--variant sd21` when VRAM is under 10 GB

Those sentences are upstream guidance, not a measurement from this patch.

Published weight sizes checked on 2026-11-09:

| File | Size | How it was checked |
| --- | --- | --- |
| `huanngzh/mv-adapter` `mvadapter_ig2mv_sd21.safetensors` | 708,310,064 bytes | Hugging Face commit `6de4033` |
| `mvadapter_ig2mv_sdxl.safetensors` | listed as 3.6 GB | Hugging Face file listing, not a local checksum |
| `ZhengPeng7/BiRefNet` `model.safetensors` | 444,473,596 bytes | Hugging Face API |
| `RealESRGAN_x2plus.pth` | 67,061,725 bytes | GitHub release `Content-Length` |
| `big-lama.pt` | 205,669,692 bytes | GitHub release `Content-Length` |

`stabilityai/stable-diffusion-2-1` and `stable-diffusion-2-1-base` returned HTTP 401 to an unauthenticated request, so their snapshot size was not measured here. Plan for several gigabytes.

SDXL's UNet is about 2.6 billion parameters, on the order of 5 GB in fp16 before activations, the two text encoders, the VAE, and the 3.6 GB adapter file. CPU offload can move idle modules to system RAM. It cannot shrink the activation memory of one UNet forward below what that forward allocates. This patch therefore does not offer an SDXL 8 GB mode. The SD2.1 UNet is on the order of 865 million parameters, about 1.7 GB in fp16, plus the 708 MB adapter file loaded into the UNet and the condition encoder. That is small enough that an 8 GB card is a plausible attempt. It is not evidence the attempt fits.

Upstream issue 66 reports one person ran the SD2.1 pipeline on 32 GB system RAM and an RTX 3090 24 GB, and had the SDXL texture run killed after 31 GB of system RAM. That is not an 8 GB data point.

## What the low-memory schedule changes

Preset `8gb` (the one to run first):

- base `stabilityai/stable-diffusion-2-1-base`
- adapter `mvadapter_ig2mv_sd21.safetensors`
- 512×512, which is the SD2.1 training resolution in `configs/view-guidance/mvadapter_i2mv_sd21.yaml` and in `scripts/inference_ig2mv_sd.py`
- 50 steps, guidance scale 3.0, DDPM plus the existing shift-SNR wrapper
- model CPU offload for the text encoder, UNet, and VAE
- the condition encoder stays on GPU because diffusers does not put it in the offload component list
- classifier-free guidance is two forwards of six views instead of one forward of twelve. The guidance formula is the same. Views inside a branch still attend to each other
- VAE slicing and VAE tiling
- nvdiffrast CUDA context instead of the OpenGL context
- background removal, if requested, is loaded and deleted before the diffusion weights
- diffusion weights are deleted before RealESRGAN and LaMa load
- RealESRGAN is dropped before LaMa when both are used
- UV padding uses OpenCV Telea when CV-CUDA is missing
- Poisson blending is not used by this bake. The script selects the PyTorch Jacobi backend so Windows does not JIT-compile the unused CUDA solver

Preset `8gb-tight` is only for a CUDA out-of-memory error at 512:

- 384×384. The network can run because the maps stay square. It is not the training resolution. Do not describe it as the same quality
- sequential CPU offload
- UV size 1024 and no 2× view upscale

Attention slicing is not enabled. The multiview processor calls `scaled_dot_product_attention` itself, so the diffusers slicing switch would not cut that cost and can replace the custom processors.

## Licensing and downloads

Accept these before the first GPU run. Do not use an unofficial mirror to skip a license prompt.

| Piece | License | Action |
| --- | --- | --- |
| This repository | Apache-2.0 | Already in `LICENSE` |
| `huanngzh/mv-adapter` weights | Apache-2.0. API reported `gated: false` on 2026-11-09 | Downloaded by the script |
| `stabilityai/stable-diffusion-2-1-base` | CreativeML Open RAIL++-M. Unauthenticated API returned HTTP 401 | Open the model page, accept the license, create a read token, `huggingface-cli login` inside the isolated venv |
| `ZhengPeng7/BiRefNet` | MIT. Only if you pass `--remove-bg` | Public on 2026-11-09 |
| RealESRGAN x2plus | BSD-3-Clause, upstream xinntao/Real-ESRGAN | Download command below. Required for the full bake with upscale |
| `big-lama.pt` | LaMa's code repository is Apache-2.0 (`advimman/lama`). The `.pt` file is the Sanster conversion used by the upstream README | Read those project terms, then download. Required for the view-inpaint bake |
| nvdiffrast | NVIDIA Source Code License | Built from `NVlabs/nvdiffrast` tag `v0.3.3` |
| PyMeshLab | GPL-3.0 | Used only if the mesh is re-unwrapped. Stays inside this venv |
| Open3D | MIT | UV atlas during re-unwrap |
| gltflib | MIT | Writes the textured GLB |

CV-CUDA is not installed. Upstream issue 51 and CV-CUDA issue 92: there is no official native Windows wheel. `MVADAPTER_CV_BACKEND=opencv` selects Telea inpaint. It is the same published algorithm as `cvcuda.inpaint`, not a bit-identical replacement.

Disk, before the first run, on the drive that holds `C:\venvs\mvadapter-lowmem`:

- torch 2.5.1+cu121 Windows wheel is 2,449,372,784 bytes compressed. The unpacked environment is larger
- torch 2.5.1+cu118 wheel is 2,700,130,584 bytes compressed
- adapter 708 MB, BiRefNet 424 MB if used, RealESRGAN 64 MB, LaMa 196 MB
- SD2.1 snapshot size was not measured because of the 401. Allow 20 GB free for the venv, the Hugging Face cache, and outputs

RAM: offload holds weights in system memory. Upstream issue 66's SDXL failure was a 32 GB machine. Close TripoSG and ComfyUI before this run. 32 GB of system RAM is the safer host. This patch was not run on 16 GB, so it does not set a measured minimum.

## Install

Use the existing Windows CPython 3.10.11. Do not install into TripoSG. Do not install into ComfyUI. Do not run `pip install -e .` in this clone: `setup.py` depends on `cvcuda_cu12`.

From PowerShell, in the clone of this fork:

```powershell
nvidia-smi
powershell -ExecutionPolicy Bypass -File .\scripts\install_lowmem_windows.ps1 -Cuda cu121
```

Use `-Cuda cu118` when `nvidia-smi` reports a CUDA driver version below 12.1. A driver that reports 12.x can run the cu121 wheel. The CUDA toolkit still has to be installed to compile nvdiffrast, and it should match the wheel (12.1 or 11.8). Visual Studio Build Tools with the C++ workload are required for that compile. Git for Windows is required for the nvdiffrast URL.

If nvdiffrast fails to build, stop. Geometry conditioning renders position and normal maps with nvdiffrast. There is no fallback that still conditions the UNet on the mesh.

Then, inside the venv:

```powershell
C:\venvs\mvadapter-lowmem\Scripts\python.exe -m pip install huggingface_hub==0.27.1
C:\venvs\mvadapter-lowmem\Scripts\huggingface-cli.exe login
```

The installer already pins `huggingface_hub`. The login command is the license step.

Download the bake checkpoints only when you are ready for the texture stage. From the repo root:

```powershell
New-Item -ItemType Directory -Force checkpoints | Out-Null
Invoke-WebRequest https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -OutFile checkpoints\RealESRGAN_x2plus.pth
Invoke-WebRequest https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt -OutFile checkpoints\big-lama.pt
```

The Hugging Face cache is `C:\venvs\mvadapter-lowmem\hf-cache`, set by the run script. It is not the TripoSG cache.

## 1. Preflight

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lowmem_neutral.ps1 -Stage preflight
```

Read the JSON. `ready_for_gpu_test` is true only when CUDA is visible to this venv. `cvcuda.import` may be false. `nvdiffrast.torch.import` must be true before the next step.

## 2. CPU self-test

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lowmem_neutral.ps1 -Stage selftest
```

This checks resolution guards, OpenCV inpaint, a UV-bearing GLB, and a JPEG base-color texture reloaded by trimesh. If `blender` is on `PATH`, it also imports and re-exports that GLB. On the authoring Linux VM, Blender 4.0.2 completed that synthetic roundtrip. Run it again on Windows; a Linux result is not a Windows result.

The committed neutral inputs are:

- `assets/lowmem/neutral_cylinder.glb` — gray PBR material, cylindrical UVs, no generated texture
- `assets/lowmem/neutral_reference.png` — synthetic 512 image, gray field with sand and slate bands. Not a photograph

## 3. Neutral GPU test

Close TripoSG and ComfyUI. First generate the six views only:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lowmem_neutral.ps1 -Stage multiview -Preset 8gb
```

That writes, under `outputs\lowmem_neutral`:

- `neutral.png` — six generated views in one row
- `neutral_pos.png`, `neutral_nor.png` — the geometry condition
- `neutral_reference.png` — the image the model actually saw
- `neutral_report.json`

Open `neutral_report.json`. Record `hardware.gpu_name`, `hardware.gpu_total_vram_mib`, `stages[ig2mv_sd21].peak_allocated_mib`, and `status`. `peak_allocated_mib` is PyTorch's allocator peak for that stage, not a full-card profiler.

If status is `completed_multiview_only` and the six views are coherent color (not noise and not a flat copy of the position render), run the bake:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lowmem_neutral.ps1 -Stage texture -Preset 8gb
```

Expected file: `outputs\lowmem_neutral\neutral_shaded.glb`.

Then, in Blender, import that GLB, confirm a material and a base-color image, and export it again. This patch does not automate that step unless `blender` is on `PATH` for the CPU self-test, which uses a synthetic texture rather than the diffusion result.

If the multiview stage raises CUDA out of memory, the report status is `cuda_oom`. Rerun once with `-Preset 8gb-tight`. If 384 also fails, stop. The blocker is that the six-view UNet forward does not fit, and further resolution cuts are outside the test this patch is willing to call supported.

## Integration boundary

The phone upload, TripoSG, Blender, and viewer stay as they are. This experiment reads two files and writes one GLB:

- reference image: whatever file the current app already saved. This repo does not call a phone API
- mesh: the OBJ or GLB TripoSG already wrote

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lowmem_neutral.ps1 -Stage texture -Preset 8gb -Mesh D:\path\to\triposg.glb -Image D:\path\to\reference.png
```

Do not put this command behind the existing synchronous HTTP handler. A 50-step six-view run with CPU offload is minutes, not a request timeout. The boundary is files in, a GLB out, started as its own process. Wiring a new API is out of scope.

TripoSG orientation may not match the orthographic rig in `scripts/inference_ig2mv_sd.py` (elevations `[0,0,0,0,89.99,-89.99]`, azimuths derived from `[0,90,180,270,180,180]`, distance 1.8). Run the neutral cylinder first. If a TripoSG mesh comes out sideways, the camera constants are the adjustment. Do not retrain.

`--remove-bg` loads BiRefNet, writes a cutout, and frees it before diffusion. The neutral PNG is already on gray and does not need it. An image that already has an alpha channel is handled inside the existing preprocess without BiRefNet.

`--keep-uv` skips Open3D re-unwrap and keeps the mesh UVs. The upstream baker re-unwraps. Use the default unwrap for the neutral test. Try `--keep-uv` only after that succeeds, if TripoSG UVs should be preserved.

## Rollback

1. Close any process using `C:\venvs\mvadapter-lowmem`.
2. Delete `C:\venvs\mvadapter-lowmem`. That removes the interpreter, the Hugging Face cache, and the installed packages.
3. Delete `checkpoints\` and `outputs\lowmem_neutral\` in this clone if you want the downloads gone.
4. Leave the TripoSG environment and the ComfyUI environment where they are.
5. To drop the code, check out the previous branch. Nothing in this patch edits a TripoSG checkout.

## What a finished local report should contain

Send back `neutral_report.json` plus whether `neutral_shaded.glb` imported in Blender. A useful report has `status` of `completed` or `cuda_oom` or `failed`, the GPU name, the total VRAM, and the per-stage peak. A report with `cuda_available: false` only repeats the preflight.
