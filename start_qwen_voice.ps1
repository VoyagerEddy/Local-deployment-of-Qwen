[CmdletBinding()]
param(
    [switch] $NoBrowser
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$projectDir = $PSScriptRoot
$webUrl = "http://127.0.0.1:7860/"
$webHealthUrl = "http://127.0.0.1:7860/health"
$defaults = Get-Content -LiteralPath (Join-Path $projectDir "runtime_defaults.json") -Raw | ConvertFrom-Json
$backendPort = if ($env:QWEN3_GGUF_PORT) { [int]$env:QWEN3_GGUF_PORT } else { [int]$defaults.QWEN3_GGUF_PORT }
$backendBaseUrl = if ($env:QWEN3_GGUF_BASE_URL) { $env:QWEN3_GGUF_BASE_URL.TrimEnd('/') } else { "http://127.0.0.1:$backendPort/v1" }
$backendHealthUrl = "$backendBaseUrl/models"
$logDir = Join-Path $projectDir "QwenTemp"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$env:PATH = "$machinePath;$userPath"
if (-not $env:QWEN3_GGUF_GPU_LAYER_CANDIDATES) {
    $env:QWEN3_GGUF_GPU_LAYER_CANDIDATES = "13,12,11,10,9,8,7,6,5,4,3,2,1,0"
}

New-Item -ItemType Directory -Force $logDir | Out-Null

function Test-LocalPort {
    param([int] $Port)
    return $null -ne (Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Start-HiddenScript {
    param(
        [string] $ScriptPath,
        [string] $LogName
    )

    $stdoutLog = Join-Path $logDir "$LogName-$timestamp.out.log"
    $stderrLog = Join-Path $logDir "$LogName-$timestamp.err.log"
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ScriptPath) `
        -WorkingDirectory $projectDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog | Out-Null
}

function Wait-HttpEndpoint {
    param(
        [string] $Uri,
        [int] $TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 3 | Out-Null
            return $true
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    return $false
}

try {
    if (-not (Test-LocalPort -Port $backendPort)) {
        Start-HiddenScript -ScriptPath (Join-Path $projectDir "run_qwen3_llamacpp.ps1") -LogName "qwen-backend"
    }

    if (-not (Test-LocalPort -Port 7860)) {
        Start-HiddenScript -ScriptPath (Join-Path $projectDir "run_voice_chat.ps1") -LogName "qwen-web"
    }

    if (-not (Wait-HttpEndpoint -Uri $webHealthUrl -TimeoutSeconds 90)) {
        throw "The web service timed out. Check the newest qwen-web logs in $logDir."
    }
    if (-not (Wait-HttpEndpoint -Uri $backendHealthUrl -TimeoutSeconds 300)) {
        throw "The model service timed out. Check the newest qwen-backend logs in $logDir."
    }

    if (-not $NoBrowser) {
        Start-Process -FilePath $webUrl | Out-Null
    }
} catch {
    $message = "$(Get-Date -Format o) $($_.Exception.Message)"
    Add-Content -LiteralPath (Join-Path $logDir "qwen-shortcut-error.log") -Value $message -Encoding UTF8
    try {
        Add-Type -AssemblyName PresentationFramework
        [System.Windows.MessageBox]::Show(
            $_.Exception.Message,
            "Qwen local launcher failed",
            "OK",
            "Error"
        ) | Out-Null
    } catch {
        Write-Error $message
    }
    exit 1
}
