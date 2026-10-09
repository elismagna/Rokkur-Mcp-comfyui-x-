<#
.SYNOPSIS
  Move the Wan 2.1 VACE model files into ComfyUI's models folder.
  Usage: .\scripts\install-models.ps1 [-From <folder>] [-ModelsDir <ComfyUI models folder>]
  -From defaults to your Downloads folder. -ModelsDir is found automatically by locating a model
  ComfyUI already has; pass it if that fails. Files already in place are left alone; nothing is deleted.
#>
param(
  [string]$From = (Join-Path $HOME "Downloads"),
  [string]$ModelsDir = ""
)
$ErrorActionPreference = "Stop"

# file name -> models subfolder (must match workflows/v2v_3070_quality/params.yaml)
$Files = [ordered]@{
  "wan2.1_vace_1.3B_fp16.safetensors"      = "diffusion_models"
  "umt5_xxl_fp8_e4m3fn_scaled.safetensors" = "text_encoders"
  "wan_2.1_vae.safetensors"                = "vae"
}

function Find-ModelsDir {
  # A model ComfyUI already lists (from data/audit.json) marks the models folder in use.
  $known = "z_image_turbo_bf16.safetensors"
  $roots = @(
    (Join-Path $HOME "Documents\ComfyUI"),
    (Join-Path $env:LOCALAPPDATA "Comfy-Desktop"),
    (Join-Path $env:APPDATA "Comfy Desktop"),
    (Join-Path $HOME "Documents")
  ) | Where-Object { Test-Path $_ }
  foreach ($root in $roots) {
    $hit = Get-ChildItem -Path $root -Filter $known -Recurse -File -Depth 6 -ErrorAction SilentlyContinue |
      Select-Object -First 1
    if ($hit) { return $hit.Directory.Parent.FullName }  # ...\models\diffusion_models\x -> ...\models
  }
  return $null
}

if (-not $ModelsDir) { $ModelsDir = Find-ModelsDir }
if (-not $ModelsDir -or -not (Test-Path $ModelsDir)) {
  Write-Error ("Could not find ComfyUI's models folder. Open it from ComfyUI Desktop ('Open Models Folder') " +
               "and rerun with -ModelsDir '<that path>'.")
}
Write-Host "ComfyUI models folder: $ModelsDir"
Write-Host "Looking for downloads in: $From"

$missing = 0
foreach ($name in $Files.Keys) {
  $destDir = Join-Path $ModelsDir $Files[$name]
  $dest = Join-Path $destDir $name
  if (Test-Path $dest) { Write-Host "  [in place] $($Files[$name])\$name"; continue }
  $src = Get-ChildItem -Path $From -Filter $name -Recurse -File -Depth 3 -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if (-not $src) { Write-Host "  [not found] $name (not in $From)"; $missing++; continue }
  New-Item -ItemType Directory -Force -Path $destDir | Out-Null
  Move-Item -Path $src.FullName -Destination $dest
  Write-Host "  [moved] $name -> $($Files[$name])\"
}
# Optional: the RTX3070_DRAFT profile's Self-Forcing LoRA (Apache-2.0), from
# huggingface.co/Kijai/WanVideo_comfy -> LoRAs/Wan2_1_self_forcing_1_3B/
$Optional = [ordered]@{ "Wan2_1_self_forcing_dmd_1_3B_lora_rank_32_fp16.safetensors" = "loras" }
foreach ($name in $Optional.Keys) {
  $dest = Join-Path (Join-Path $ModelsDir $Optional[$name]) $name
  if (Test-Path $dest) { Write-Host "  [in place] $($Optional[$name])\$name (optional)"; continue }
  $src = Get-ChildItem -Path $From -Filter $name -Recurse -File -Depth 3 -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if (-not $src) { Write-Host "  [optional, not found] $name (only RTX3070_DRAFT needs it)"; continue }
  New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
  Move-Item -Path $src.FullName -Destination $dest
  Write-Host "  [moved] $name -> $($Optional[$name])\"
}
if ($missing) { Write-Host "$missing file(s) still missing. Download them, then run this again."; exit 1 }
Write-Host "All three models are in place. Restart ComfyUI (or press R in its browser tab), then run:"
Write-Host "  .\scripts\studio.ps1 comfy-check"
