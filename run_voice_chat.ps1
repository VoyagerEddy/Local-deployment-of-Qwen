$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:HF_HOME = "D:\QwenModels\hf_cache"
$env:TEMP = "D:\QwenTemp"
$env:TMP = "D:\QwenTemp"
$env:QWEN25_OMNI_MNN_CONFIG = "D:\QwenModels\Qwen2.5-Omni-3B-MNN\config.json"
$env:QWEN25_OMNI_DATA_DIR = "D:\QwenData\qwen25-omni-voice"

New-Item -ItemType Directory -Force $env:TEMP, $env:QWEN25_OMNI_DATA_DIR | Out-Null

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "ffmpeg is required but was not found on PATH."
}

& "D:\QwenEnvs\qwen25omni\Scripts\python.exe" "$PSScriptRoot\app.py"
