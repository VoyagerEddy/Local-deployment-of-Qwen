# Qwen Local Voice Chat

This app defaults to `Qwen3-Omni-30B-A3B-Instruct` through GGUF quantization and a local llama.cpp/Ollama-compatible API.

## Paths

- Python environment: `D:\QwenEnvs\qwen25omni`
- Qwen3 GGUF and cache: `D:\QwenModels`
- Qwen3 local files: `D:\QwenModels\Qwen3-Omni-30B-A3B-Instruct-GGUF`
- Runtime data and temporary recordings: `D:\QwenData\qwen-voice`
- Browser VAD assets: `D:\QwenData\qwen-voice\vad-assets`
- App: `D:\QwenData\github-sync\Local-deployment-of-Qwen`

## Qwen3 GGUF Backend

Recommended route for realtime voice is llama.cpp with multimodal support:

```powershell
cd D:\QwenData\github-sync\Local-deployment-of-Qwen
.\run_qwen3_llamacpp.ps1
```

The script prefers local files when they exist:

```text
D:\QwenModels\Qwen3-Omni-30B-A3B-Instruct-GGUF\Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf
D:\QwenModels\Qwen3-Omni-30B-A3B-Instruct-GGUF\mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf
```

If `Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf.aria2` still exists, the download is not finished yet.

The script starts:

```text
ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF:Q4_K_M
API model name: qwen3-omni-30b-a3b-instruct-q4km
context: 2048
gpu layers: 8
server: http://127.0.0.1:8080/v1
```

Q4_K_M is about 18.6 GB, so it cannot fit entirely in 6 GB VRAM. Keep most weights on CPU/RAM and tune GPU offload with:

```powershell
$env:QWEN3_GGUF_GPU_LAYERS = "4"   # safer
$env:QWEN3_GGUF_GPU_LAYERS = "8"   # default
$env:QWEN3_GGUF_GPU_LAYERS = "12"  # try if VRAM allows
```

If you see OOM, lower `QWEN3_GGUF_GPU_LAYERS`. Keep `QWEN3_GGUF_CONTEXT=2048` because KV cache also consumes VRAM and RAM.

Ollama route:

```powershell
cd D:\QwenData\github-sync\Local-deployment-of-Qwen
.\run_qwen3_ollama.ps1
```

Ollama compatibility for audio input depends on its current multimodal support. llama.cpp is the preferred backend for Qwen3-Omni realtime voice.

## App

Start the web app in another terminal:

```powershell
cd D:\QwenData\github-sync\Local-deployment-of-Qwen
.\run_voice_chat.ps1
```

Open:

```text
http://127.0.0.1:7860
```

## Notes

- Qwen3 GGUF is the default model option.
- The browser uses Silero VAD v5, waits for about 420 ms of silence, then sends the whole 16 kHz WAV segment to Qwen3 through the GGUF backend.
- The app does not load the existing Qwen2.5 MNN model at startup.
- Qwen2.5 MNN remains only as a selectable fallback in the UI.
