<#
.SYNOPSIS
  Put a Rökkur Studio icon on the desktop. Double-clicking it runs scripts\launch.ps1:
  Docker, the studio, the question about the cloud GPU, then the dashboard.
  Run:  .\scripts\studio.ps1 shortcut
#>
$Root = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath("Desktop")  # follows a desktop moved to OneDrive
$link = Join-Path $desktop "Rökkur Studio.lnk"
$launcher = Join-Path $Root "scripts\launch.ps1"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
# Bypass applies to this one launch only; it does not change the machine's script policy.
$shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$launcher`""
$shortcut.WorkingDirectory = $Root
$shortcut.IconLocation = "$(Join-Path $Root 'assets\rokkur.ico'),0"
$shortcut.Description = "Start Rökkur Studio"
$shortcut.Save()
Write-Host "Made $link"
Write-Host "Double-click it to start Rökkur Studio. Right-click it > Pin to Start or Pin to taskbar if you like."
