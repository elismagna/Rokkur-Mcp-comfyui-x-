<#
.SYNOPSIS
  Start Rökkur Studio: Docker, the studio and, if you say yes, the RunPod cloud GPU.
  The desktop icon runs this. Make the icon with:  .\scripts\studio.ps1 shortcut

  -Cloud    start the cloud GPU without asking
  -NoCloud  do not ask about the cloud GPU
  -Off      stop the cloud GPU and close its tunnel, then exit (studio.ps1 cloud-off)

  The cloud GPU part needs STUDIO_CLOUD__RUNPOD_API_KEY and STUDIO_CLOUD__RUNPOD_POD_ID in
  .env (docs/cloud.md). Without them the studio starts as before and asks nothing.
#>
param([switch]$Cloud, [switch]$NoCloud, [switch]$Off)

# The same behaviour from the desktop icon and from studio.ps1 (which sets Stop): problems are
# handled below and shown in a message box.
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Dashboard = "http://127.0.0.1:8400/ui"
$PidFile = Join-Path $Root "data\cloud-tunnel.pid"
# ssh reads UserKnownHostsFile as a list split at spaces, so this path must have none
# (the studio folder "rokkur studio" has one).
$KnownHosts = Join-Path $env:USERPROFILE ".ssh\rokkur_runpod_known_hosts"
try { $Host.UI.RawUI.WindowTitle = "Rökkur Studio" } catch { }
Add-Type -AssemblyName System.Windows.Forms

function Say([string]$Text) { Write-Host "  $Text" }

function Show-Box([string]$Text, $Buttons, $Icon) {
  # an owner form that stays on top, so the question never hides behind other windows
  $owner = New-Object System.Windows.Forms.Form -Property @{ TopMost = $true }
  try { return [System.Windows.Forms.MessageBox]::Show($owner, $Text, "Rökkur Studio", $Buttons, $Icon) }
  finally { $owner.Dispose() }
}
function Ask([string]$Text) {
  $answer = Show-Box $Text ([System.Windows.Forms.MessageBoxButtons]::YesNo) ([System.Windows.Forms.MessageBoxIcon]::Question)
  return $answer -eq [System.Windows.Forms.DialogResult]::Yes
}
function Tell([string]$Text, [string]$Icon = "Information") {
  Show-Box $Text ([System.Windows.Forms.MessageBoxButtons]::OK) ([System.Windows.Forms.MessageBoxIcon]::$Icon) | Out-Null
}

# Windows PowerShell 5.1 turns a program's stderr into errors when it is redirected; run
# programs with errors set to Continue and judge them by their exit code instead.
function Native([scriptblock]$Block) {
  $old = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try { & $Block } finally { $ErrorActionPreference = $old }
}

function Test-Docker { Native { docker info *> $null }; return $LASTEXITCODE -eq 0 }

function Test-Http([string]$Url) {
  try { return (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 $Url).StatusCode -eq 200 }
  catch { return $false }
}

# Ask the studio (inside Docker) about the pod; returns the parsed JSON line or $null.
function Invoke-CloudPod([string[]]$CliArgs) {
  $lines = Native { docker compose exec -T api rokkur-studio cloud-pod @CliArgs --json 2>$null }
  $json = @($lines) | Where-Object { "$_".StartsWith('{"configured"') } | Select-Object -Last 1
  if (-not $json) {  # the api container is not running: use a one-off container
    $lines = Native { docker compose run --rm -T api rokkur-studio cloud-pod @CliArgs --json 2>$null }
    $json = @($lines) | Where-Object { "$_".StartsWith('{"configured"') } | Select-Object -Last 1
  }
  if (-not $json) { return $null }
  return $json | ConvertFrom-Json
}

function Stop-Tunnel {
  if (-not (Test-Path $PidFile)) { return }
  $old = Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($old -match '^\d+$') {
    Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue |
      Where-Object { $_.ProcessName -eq "ssh" } | Stop-Process -Force -ErrorAction SilentlyContinue
  }
  Remove-Item $PidFile -ErrorAction SilentlyContinue
}

function Connect-Cloud($Pod) {
  if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    throw "The Windows SSH client is missing. Add it under Settings > System > Optional features > OpenSSH Client."
  }
  if ($Pod.running) { Say "Connecting to the cloud GPU..." }
  else { Say "Starting the cloud GPU on RunPod. This takes a minute or two..." }
  $info = Invoke-CloudPod @("start", "--wait")
  if (-not $info) { throw "The studio did not answer (is Docker running?)." }
  if ($info.error) { throw $info.error }

  $key = if ($info.ssh_key) { $info.ssh_key } else { Join-Path $env:USERPROFILE ".ssh\id_ed25519" }
  if (-not (Test-Path $key)) { throw "There is no SSH key at $key. Set STUDIO_CLOUD__SSH_KEY in .env (docs/cloud.md)." }
  New-Item -ItemType Directory -Force (Split-Path $KnownHosts) | Out-Null
  $target = "$($info.ssh_user)@$($info.ip)"
  # A restarted pod can get an address it had before with a new host key: forget the old key
  # for this address. The address itself comes from RunPod's API over HTTPS.
  if (Test-Path $KnownHosts) { Native { ssh-keygen -R "[$($info.ip)]:$($info.ssh_port)" -f $KnownHosts *> $null } }
  $ssh = @("-p", "$($info.ssh_port)", "-i", $key, "-o", "UserKnownHostsFile=$KnownHosts",
           "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15")

  Say "Starting ComfyUI on the cloud GPU..."
  $started = $false
  for ($try = 1; $try -le 6 -and -not $started; $try++) {  # sshd can need a moment after boot
    Native { ssh @ssh $target $info.remote_start *> $null }
    $started = $LASTEXITCODE -eq 0
    if (-not $started) { Start-Sleep -Seconds 10 }
  }
  if (-not $started) {
    throw "Could not log in to the pod with SSH ($target, port $($info.ssh_port)). Check that your SSH key is on the pod (docs/cloud.md, RunPod notes)."
  }

  $port = [int]$info.local_port
  $local = "http://127.0.0.1:$port/system_stats"
  $tunnel = $null
  if (-not (Test-Http $local)) {
    Stop-Tunnel
    # Start-Process does not quote arguments, so the paths are quoted here.
    $argLine = "-N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes " +
               "-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KnownHosts " +
               "-i `"$key`" -p $($info.ssh_port) -L $($port):127.0.0.1:$($info.remote_port) $target"
    $tunnel = Start-Process -FilePath "ssh" -ArgumentList $argLine -WindowStyle Hidden -PassThru -ErrorAction Stop
    Set-Content -Path $PidFile -Value $tunnel.Id
  }

  Say "Waiting for ComfyUI on the cloud GPU..."
  $deadline = (Get-Date).AddMinutes(4)
  while (-not (Test-Http $local)) {
    if ($tunnel -and $tunnel.HasExited) {
      throw "The SSH tunnel closed at once. Another program may be using port $port, such as a tunnel window you opened by hand; close it and try again."
    }
    if ((Get-Date) -gt $deadline) {
      throw "ComfyUI on the pod did not answer within 4 minutes. Its log is /workspace/comfyui.log on the pod."
    }
    Start-Sleep -Seconds 4
  }
  return $info
}

# Anything unexpected: say so in a box, since the icon's window closes when the script ends.
trap {
  Tell "Rökkur Studio could not start:`n$($_.Exception.Message)`n`nRun .\scripts\studio.ps1 launch in PowerShell to see more." "Error"
  exit 1
}

Write-Host ""
Write-Host "  RÖKKUR STUDIO" -ForegroundColor Yellow
Write-Host ""

# -- 1. Docker ------------------------------------------------------------------------------
if (-not (Test-Docker)) {
  $desktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
  if (-not (Test-Path $desktop)) {
    Tell "Docker is not running. Start Docker Desktop, then open Rökkur Studio again." "Error"
    exit 1
  }
  Say "Starting Docker Desktop..."
  Start-Process $desktop
  $deadline = (Get-Date).AddMinutes(4)
  while (-not (Test-Docker)) {
    if ((Get-Date) -gt $deadline) {
      Tell "Docker Desktop did not start within 4 minutes. Start it yourself, then open Rökkur Studio again." "Error"
      exit 1
    }
    Start-Sleep -Seconds 3
  }
}

# -- cloud-off: stop the pod and the tunnel, nothing else -----------------------------------
if ($Off) {
  Stop-Tunnel
  $result = Invoke-CloudPod @("stop")
  if ($result -and $result.stopped) { Say "Stopping the cloud GPU. RunPod stops charging its hourly rate once it has stopped." }
  elseif ($result -and $result.error) { Say "RunPod: $($result.error)"; exit 1 }
  else { Say "RunPod start/stop is not set up (docs/cloud.md)."; exit 1 }
  exit 0
}

# -- 2. The studio --------------------------------------------------------------------------
Say "Starting the studio..."
& (Join-Path $PSScriptRoot "studio.ps1") up
if ($LASTEXITCODE -ne 0) {
  Tell "The studio did not start. Run .\scripts\studio.ps1 up in PowerShell to see why." "Error"
  exit 1
}

# -- 3. The cloud GPU -----------------------------------------------------------------------
$cloudNote = ""
$pod = if ($NoCloud) { $null } else { Invoke-CloudPod @("status") }
if ($pod -and $pod.configured) {
  $want = $false
  if ($pod.error) {
    Tell "Could not check the cloud GPU:`n$($pod.error)`n`nThe studio opens and renders on this PC." "Warning"
  } elseif ($pod.running -or $Cloud) {
    $want = $true
  } elseif ($pod.ask_on_launch) {
    $gpu = if ($pod.gpu) { " ($($pod.gpu))" } else { "" }
    $price = if ($pod.price_per_hour) { " It costs about `$$($pod.price_per_hour) an hour while it is on." } else { "" }
    $idle = if ($pod.auto_stop_idle_minutes -gt 0) { " The studio turns it off after $($pod.auto_stop_idle_minutes) minutes without cloud work." } else { "" }
    $want = Ask ("Turn on the cloud GPU$gpu?$price`n`nYes: start it now. It takes a minute or two.$idle`nNo: render on this PC only.")
  }
  if ($want) {
    try {
      $info = Connect-Cloud $pod
      $cloudNote = "The cloud GPU is on. Pick Cloud server under Where to render. Stop it on the System page when you are done."
      Say "Cloud GPU ready: $($info.label)"
    } catch {
      Tell ("The cloud GPU could not be started:`n$($_.Exception.Message)`n`nThe studio opens and renders on this PC.") "Warning"
    }
  }
}

# -- 4. Open the dashboard ------------------------------------------------------------------
Say "Opening the dashboard..."
$deadline = (Get-Date).AddMinutes(2)
while (-not (Test-Http "$Dashboard/status") -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 2 }
Start-Process $Dashboard
Write-Host ""
Say "Rökkur Studio is open in your browser. You can close this window."
if ($cloudNote) { Say $cloudNote }
Start-Sleep -Seconds 6
