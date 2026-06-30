$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:HF_HOME = "D:\QwenModels\hf_cache"
$env:TEMP = "D:\QwenTemp"
$env:TMP = "D:\QwenTemp"
$env:QWEN_VOICE_DEFAULT_BACKEND = "qwen3-gguf"
$env:QWEN_VOICE_DATA_DIR = "D:\QwenData\qwen-voice"
$env:QWEN3_GGUF_BASE_URL = "http://127.0.0.1:8080/v1"
$env:QWEN3_GGUF_MODEL = "qwen3-omni-30b-a3b-instruct-q4km"
$env:QWEN3_GGUF_CONTEXT = "2048"
$env:QWEN3_GGUF_GPU_LAYERS = "8"
$env:QWEN25_OMNI_MNN_CONFIG = "D:\QwenModels\Qwen2.5-Omni-3B-MNN\config.json"

New-Item -ItemType Directory -Force $env:TEMP, $env:QWEN_VOICE_DATA_DIR | Out-Null

$vadAssetsDir = Join-Path $env:QWEN_VOICE_DATA_DIR "vad-assets"
$legacyVadAssetsDir = "D:\QwenData\qwen25-omni-voice\vad-assets"
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

& "D:\QwenEnvs\qwen25omni\Scripts\python.exe" "$PSScriptRoot\app.py"
