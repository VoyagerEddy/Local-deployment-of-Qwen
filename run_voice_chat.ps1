$ErrorActionPreference = "Stop"
$modelsRoot = Join-Path $PSScriptRoot "QwenModels"
$envsRoot = Join-Path $PSScriptRoot "QwenEnvs"
$dataRoot = Join-Path $PSScriptRoot "QwenData"
$tempRoot = Join-Path $PSScriptRoot "QwenTemp"
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:HF_HOME = Join-Path $modelsRoot "hf_cache"
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
if (-not $env:QWEN_VOICE_DATA_DIR) {
    $env:QWEN_VOICE_DATA_DIR = Join-Path $dataRoot "qwen-voice"
}

New-Item -ItemType Directory -Force $env:TEMP, $env:QWEN_VOICE_DATA_DIR | Out-Null

$vadAssetsDir = Join-Path $env:QWEN_VOICE_DATA_DIR "vad-assets"
$legacyVadAssetsDir = Join-Path $dataRoot "qwen25-omni-voice\vad-assets"
New-Item -ItemType Directory -Force $vadAssetsDir | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $vadAssetsDir "bundle.min.js"))) {
    if (-not (Test-Path -LiteralPath $legacyVadAssetsDir)) {
        throw "Silero VAD assets were not found: $vadAssetsDir"
    }

    Copy-Item -Path (Join-Path $legacyVadAssetsDir "*") -Destination $vadAssetsDir -Recurse -Force
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "ffmpeg is required but was not found on PATH."
}

$pythonExe = Join-Path $envsRoot "qwen25omni\Scripts\python.exe"
& $pythonExe "$PSScriptRoot\app.py"
