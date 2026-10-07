<#
.SYNOPSIS
  Windows equivalent of the Makefile. Usage: .\scripts\studio.ps1 <command>
  Commands: up, down, logs, test, migrate, studio, worker, comfy-check, youtube-auth,
            smoke-test, audit, lint, build
#>
param([Parameter(Mandatory = $true)][string]$Command)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Compose { docker compose @args; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
function InApi { Compose run --rm api rokkur-studio @args }

switch ($Command) {
  "build"        { Compose build }
  "up"           { Compose up -d --build }
  "down"         { Compose down }
  "logs"         { Compose logs -f --tail=200 }
  "migrate"      { InApi migrate }
  "studio"       { Compose up -d api }
  "worker"       { Compose up -d worker }
  "comfy-check"  { InApi comfy-check }
  "youtube-auth" { InApi youtube-auth }
  "smoke-test"   { InApi smoke-test --inject-fault }
  "audit"        { InApi audit }
  "test"         { python -m pytest }
  "lint"         { ruff check src tests; mypy }
  default        { Write-Error "unknown command '$Command'"; exit 1 }
}
