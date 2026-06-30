$ErrorActionPreference = "Stop"

$env:OLLAMA_MODELS = "D:\QwenModels\ollama"
New-Item -ItemType Directory -Force $env:OLLAMA_MODELS | Out-Null

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama was not found. Install it first from https://ollama.com/download/windows"
}

$modelName = "qwen3-omni-30b-a3b-instruct-q4km"
$modelfile = Join-Path $PSScriptRoot "Modelfile.qwen3-omni-q4km"

ollama create $modelName -f $modelfile
ollama serve
