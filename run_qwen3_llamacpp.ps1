$ErrorActionPreference = "Stop"

$env:HF_HOME = "D:\QwenModels\hf_cache"
$env:TEMP = "D:\QwenTemp"
$env:TMP = "D:\QwenTemp"
New-Item -ItemType Directory -Force $env:HF_HOME, $env:TEMP | Out-Null

$apiModel = $env:QWEN3_GGUF_MODEL
if (-not $apiModel) {
    $apiModel = "qwen3-omni-30b-a3b-instruct-q4km"
}

$hfRepo = $env:QWEN3_GGUF_HF_REPO
if (-not $hfRepo) {
    $hfRepo = "ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF:Q4_K_M"
}

$ctx = $env:QWEN3_GGUF_CONTEXT
if (-not $ctx) { $ctx = "2048" }

$ngl = $env:QWEN3_GGUF_GPU_LAYERS
if (-not $ngl) { $ngl = "8" }

$port = $env:QWEN3_GGUF_PORT
if (-not $port) { $port = "8080" }

$modelDir = $env:QWEN3_GGUF_MODEL_DIR
if (-not $modelDir) {
    $modelDir = "D:\QwenModels\Qwen3-Omni-30B-A3B-Instruct-GGUF"
}

$localModel = Join-Path $modelDir "Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"
$localMmproj = Join-Path $modelDir "mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf"
$localModelAria2 = "$localModel.aria2"
$minimumModelBytes = 15GB
$useLocalModel = $false

if (Test-Path -LiteralPath $localModel) {
    if (Test-Path -LiteralPath $localModelAria2) {
        throw "Local Qwen3 GGUF is still downloading: $localModelAria2. Wait for the download to finish, then run this script again."
    }

    $modelFile = Get-Item -LiteralPath $localModel
    if ($modelFile.Length -lt $minimumModelBytes) {
        throw "Local Qwen3 GGUF is incomplete: $localModel. Wait for the download to finish, then run this script again."
    }

    if (-not (Test-Path -LiteralPath $localMmproj)) {
        throw "Qwen3 multimodal projector was not found: $localMmproj"
    }

    $useLocalModel = $true
}

$llama = Get-Command llama -ErrorAction SilentlyContinue
$llamaServer = Get-Command llama-server -ErrorAction SilentlyContinue
$wingetLlamaServer = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"

if ($llama) {
    if ($useLocalModel) {
        & $llama.Source serve -m $localModel -mm $localMmproj -a $apiModel --host 127.0.0.1 --port $port -c $ctx -ngl $ngl --jinja
    } else {
        & $llama.Source serve -hf $hfRepo -a $apiModel --host 127.0.0.1 --port $port -c $ctx -ngl $ngl --jinja
    }
    exit $LASTEXITCODE
}

if ($llamaServer) {
    if ($useLocalModel) {
        & $llamaServer.Source -m $localModel -mm $localMmproj -a $apiModel --host 127.0.0.1 --port $port -c $ctx -ngl $ngl --jinja
    } else {
        & $llamaServer.Source -hf $hfRepo -a $apiModel --host 127.0.0.1 --port $port -c $ctx -ngl $ngl --jinja
    }
    exit $LASTEXITCODE
}

if (Test-Path -LiteralPath $wingetLlamaServer) {
    if ($useLocalModel) {
        & $wingetLlamaServer -m $localModel -mm $localMmproj -a $apiModel --host 127.0.0.1 --port $port -c $ctx -ngl $ngl --jinja
    } else {
        & $wingetLlamaServer -hf $hfRepo -a $apiModel --host 127.0.0.1 --port $port -c $ctx -ngl $ngl --jinja
    }
    exit $LASTEXITCODE
}

throw "llama.cpp was not found. Install it first, for example: winget install llama.cpp"
