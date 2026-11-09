# Runs the neutral-object low-memory test in the isolated venv.
# Close TripoSG and ComfyUI first so they are not holding the GPU or RAM.
param(
    [string]$Venv = "C:\venvs\mvadapter-lowmem",
    [ValidateSet("preflight", "selftest", "multiview", "texture")]
    [string]$Stage = "preflight",
    [ValidateSet("8gb", "8gb-tight")]
    [string]$Preset = "8gb",
    [string]$Mesh = "",
    [string]$Image = "",
    [string]$Repo = ""
)

$ErrorActionPreference = "Stop"
$py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "Isolated interpreter not found: $py. Run scripts\install_lowmem_windows.ps1 first."
}

if (-not $Repo) {
    $Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
Set-Location $Repo

$env:HF_HOME = Join-Path $Venv "hf-cache"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:MVADAPTER_CV_BACKEND = "opencv"
$env:MVADAPTER_RASTER_BACKEND = "cuda"

if ($Stage -eq "preflight") {
    & $py -m scripts.lowmem_preflight
    exit $LASTEXITCODE
}
if ($Stage -eq "selftest") {
    & $py -m scripts.lowmem_export_selftest
    exit $LASTEXITCODE
}

$cmdArgs = @(
    "-m", "scripts.texture_i2tex_lowmem",
    "--preset", $Preset,
    "--save-dir", (Join-Path $Repo "outputs\lowmem_neutral"),
    "--save-name", "neutral"
)
if ($Stage -eq "multiview") {
    $cmdArgs += "--multiview-only"
}
if ($Mesh) { $cmdArgs += @("--mesh", $Mesh) }
if ($Image) { $cmdArgs += @("--image", $Image) }

& $py @cmdArgs
exit $LASTEXITCODE
