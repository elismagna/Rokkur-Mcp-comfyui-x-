<#
.SYNOPSIS
  Windows equivalent of the Makefile. Usage: .\scripts\studio.ps1 <command>
  Commands: up, down, logs, test, migrate, studio, worker, comfy-check, youtube-auth,
            smoke-test, audit, lint, build, install-models, comfy-render, render, agent-check, publish,
            prompt-schedule, youtube-playlists
  publish: .\scripts\studio.ps1 publish <project id> [--privacy unlisted] [--dry-run]
           [--at "2026-10-09 18:00" | --at next] [--playlist "Shorts"]
  render: .\scripts\studio.ps1 render myclip.mp4 --theme "1970s claymation" --rights USER_OWNED --evidence "I filmed it"
          (the file must be in the media folder; extra options are passed to rokkur-studio render,
          e.g. --character NEO)
  prompt-schedule: .\scripts\studio.ps1 prompt-schedule <project id>   (FizzNodes Batch Prompt Schedule text)
#>
# Plain (non-advanced) param block on purpose: options like --theme land in $args
# instead of being rejected as unknown PowerShell parameters.
param([string]$Command)
$Rest = @($args)
if (-not $Command) { Write-Error "usage: .\scripts\studio.ps1 <command>"; exit 1 }
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Compose { docker compose @args; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
function InApi { Compose run --rm api rokkur-studio @args }

switch ($Command) {
  "build"        { Compose build migrate }
  "up"           { Compose build migrate; Compose up -d }
  "down"         { Compose down }
  "logs"         { Compose logs -f --tail=200 }
  "migrate"      { InApi migrate }
  "studio"       { Compose up -d api }
  "worker"       { Compose up -d worker }
  "comfy-check"  { InApi comfy-check }
  "agent-check"  { InApi agent-check @Rest }
  "youtube-auth" {
    # Google sends the browser back to 127.0.0.1:8401, so that port must reach the container.
    if (-not (Test-Path "secrets")) { New-Item -ItemType Directory "secrets" | Out-Null }
    Compose run --rm -p 127.0.0.1:8401:8401 api rokkur-studio youtube-auth @Rest
  }
  "publish"      { InApi publish @Rest }
  "youtube-playlists" { InApi youtube-playlists }
  "prompt-schedule" { InApi prompt-schedule @Rest }
  "smoke-test"   { InApi smoke-test --inject-fault }
  "comfy-render" { InApi smoke-test --renderer comfyui --profile PREVIEW --timeout 3600 }
  "render"       {
    if ($Rest.Count -eq 0) { Write-Error "usage: render <file in media folder> --theme ... --rights USER_OWNED --evidence ..."; exit 2 }
    $src = $Rest[0]
    if (-not $src.StartsWith("/")) { $src = "/media/" + (Split-Path -Leaf $src) }
    InApi render $src @($Rest | Select-Object -Skip 1)
  }
  "audit"        { InApi audit }
  "install-models" { & (Join-Path $PSScriptRoot "install-models.ps1") }
  "test"         { python -m pytest }
  "lint"         { ruff check src tests; mypy }
  default        { Write-Error "unknown command '$Command'"; exit 1 }
}
