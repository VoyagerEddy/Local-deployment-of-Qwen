$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:HF_HOME = "D:\CodexModels\hf_cache"
$env:TEMP = "D:\CodexTemp"
$env:TMP = "D:\CodexTemp"
$env:QWEN25_OMNI_MNN_CONFIG = "D:\CodexModels\Qwen2.5-Omni-3B-MNN\config.json"
$env:QWEN25_OMNI_DATA_DIR = "D:\CodexData\qwen25-omni-voice"

New-Item -ItemType Directory -Force $env:TEMP, $env:QWEN25_OMNI_DATA_DIR | Out-Null

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "ffmpeg is required but was not found on PATH."
}

& "D:\CodexEnvs\qwen25omni\Scripts\python.exe" "$PSScriptRoot\app.py"
