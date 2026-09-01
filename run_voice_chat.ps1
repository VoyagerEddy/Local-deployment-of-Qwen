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
$env:QWEN_VOICE_DEFAULT_BACKEND = "qwen3-gguf"
$env:QWEN_VOICE_DATA_DIR = Join-Path $dataRoot "qwen-voice"
$env:QWEN3_GGUF_BASE_URL = "http://127.0.0.1:8080/v1"
$env:QWEN3_GGUF_MODEL = "qwen3-omni-30b-a3b-instruct-q4km"
$env:QWEN3_GGUF_CONTEXT = "768"
$env:QWEN3_GGUF_GPU_LAYERS = "auto"
$env:QWEN3_GGUF_MIN_FREE_VRAM_MB = "768"
$env:QWEN3_GGUF_CACHE_TYPE_K = "q8_0"
$env:QWEN3_GGUF_CACHE_TYPE_V = "q8_0"
$env:QWEN3_GGUF_FLASH_ATTN = "on"
$env:QWEN3_GGUF_MLOCK = "off"
$env:QWEN3_GGUF_PARALLEL = "1"
$env:QWEN3_GGUF_BATCH_SIZE = "256"
$env:QWEN3_GGUF_UBATCH_SIZE = "64"
$env:QWEN3_GGUF_PROMPT_CACHE = "off"
$env:QWEN3_GGUF_PROMPT_CACHE_RAM_MB = "0"
$env:QWEN3_GGUF_CTX_CHECKPOINTS = "0"
$env:QWEN3_GGUF_MIN_FREE_RAM_BEFORE_LAUNCH_MB = "4096"
$env:QWEN3_GGUF_MIN_FREE_RAM_MB = "1024"
$env:QWEN3_GGUF_MIN_REQUEST_RAM_MB = "1024"
$env:QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB = "512"
$env:QWEN3_GGUF_MAX_PAGE_READS_PER_SEC = "120"
$env:QWEN3_GGUF_MAX_PAGE_WRITES_PER_SEC = "80"
$env:QWEN3_GGUF_TRIM_WORKING_SET = "on"
$env:QWEN3_GGUF_TRIM_BELOW_RAM_MB = "1536"
$env:QWEN3_GGUF_MAX_TEXT_TOKENS = "96"
$env:QWEN3_GGUF_MAX_AUDIO_TOKENS = "96"
$env:QWEN3_LLAMA_STATE_FILE = Join-Path $tempRoot "qwen3-llamacpp-state.json"
$env:QWEN25_OMNI_MNN_CONFIG = Join-Path $modelsRoot "Qwen2.5-Omni-3B-MNN\config.json"
$env:QWEN_VOICE_STOP_ON_BROWSER_CLOSE = "on"
$env:QWEN_VOICE_BROWSER_HEARTBEAT_TIMEOUT_SECONDS = "45"
$env:QWEN_VOICE_BROWSER_SHUTDOWN_GRACE_SECONDS = "8"

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
