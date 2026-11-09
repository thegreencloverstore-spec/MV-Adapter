# Isolated installer for the MV-Adapter low-memory experiment.
# Does not modify an existing TripoSG or ComfyUI environment.
# Run from the clone of this fork, in Windows PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\scripts\install_lowmem_windows.ps1
param(
    [string]$Python = "py",
    [string]$Venv = "C:\venvs\mvadapter-lowmem",
    [ValidateSet("cu121", "cu118")]
    [string]$Cuda = "cu121"
)

$ErrorActionPreference = "Stop"

$lower = $Venv.ToLowerInvariant()
if ($lower.Contains("triposg") -or $lower.Contains("comfy")) {
    throw "Refusing to create the venv inside a path that looks like TripoSG or ComfyUI: $Venv"
}

Write-Host "Creating isolated venv at $Venv"
if ($Python -eq "py") {
    & py -3.10 -m venv $Venv
} else {
    & $Python -m venv $Venv
}

$py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "venv python was not created at $py"
}

& $py -c "import sys; print(sys.version); assert sys.version.startswith('3.10'), 'This pin set is for CPython 3.10'"

$env:HF_HOME = Join-Path $Venv "hf-cache"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

& $py -m pip install --upgrade "pip==24.3.1"
& $py -m pip install --upgrade "setuptools==75.6.0" "wheel==0.45.1"

$index = "https://download.pytorch.org/whl/$Cuda"
Write-Host "Installing torch 2.5.1 from $index"
& $py -m pip install "torch==2.5.1" "torchvision==0.20.1" --index-url $index

$req = Join-Path $PSScriptRoot ".." "requirements-lowmem-windows.txt"
$req = (Resolve-Path $req).Path
& $py -m pip install -r $req

Write-Host "Building nvdiffrast. This needs Git, a CUDA toolkit matching $Cuda, and MSVC Build Tools."
Write-Host "If this step fails, the geometry-conditioned model cannot run. Do not replace it with a color fill."
& $py -m pip install "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"

& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"

Write-Host ""
Write-Host "Install finished."
Write-Host "HF cache for this env: $env:HF_HOME"
Write-Host "Do not pip install -e . and do not install this into the TripoSG interpreter."
Write-Host "Next: docs\LOWMEM_IG2MV_WINDOWS.md"
