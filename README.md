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
context: 768
gpu layers: auto
server: http://127.0.0.1:8080/v1
```

Q4_K_M is about 18.6 GB, so it cannot fit entirely in 6 GB VRAM. The launcher defaults to automatic GPU offload:

```powershell
$env:QWEN3_GGUF_GPU_LAYERS = "auto"
```

In `auto` mode it tries higher GPU offload first and falls back if llama.cpp fails to start or if free VRAM is below the safety margin. The default candidate list is:

```text
999,96,80,64,48,40,36,32,30,28,26,24,20,18,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
```

Useful overrides:

```powershell
$env:QWEN3_GGUF_GPU_LAYER_CANDIDATES = "999,96,80,64,48,40,36,32,30,28,26,24"
$env:QWEN3_GGUF_MIN_FREE_VRAM_MB = "768"
$env:QWEN3_GGUF_GPU_LAYERS = "8"  # fixed mode, no automatic fallback
```

KV cache quantization is enabled by default to reduce VRAM pressure:

```powershell
$env:QWEN3_GGUF_CACHE_TYPE_K = "q8_0"
$env:QWEN3_GGUF_CACHE_TYPE_V = "q8_0"
```

Flash Attention is enabled by default for the llama.cpp backend:

```powershell
$env:QWEN3_GGUF_FLASH_ATTN = "on"
```

The launcher also uses RAM-safe defaults to reduce Windows paging:

```powershell
$env:QWEN3_GGUF_MLOCK = "off"                 # optional; on can overpressure 16 GB RAM
$env:QWEN3_GGUF_PARALLEL = "1"                # single local user, fewer KV/cache slots
$env:QWEN3_GGUF_CONTEXT = "768"               # short realtime turns, smaller KV cache
$env:QWEN3_GGUF_BATCH_SIZE = "256"            # lower prompt-processing RAM spikes
$env:QWEN3_GGUF_UBATCH_SIZE = "64"
$env:QWEN3_GGUF_PROMPT_CACHE = "off"
$env:QWEN3_GGUF_PROMPT_CACHE_RAM_MB = "0"     # disable llama.cpp's RAM prompt cache
$env:QWEN3_GGUF_CTX_CHECKPOINTS = "0"
$env:QWEN3_GGUF_MIN_FREE_RAM_BEFORE_LAUNCH_MB = "4096"
$env:QWEN3_GGUF_MIN_FREE_RAM_MB = "1024"
$env:QWEN3_GGUF_MIN_REQUEST_RAM_MB = "1024"   # soft guard
$env:QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB = "512"
$env:QWEN3_GGUF_MAX_PAGE_READS_PER_SEC = "120"
$env:QWEN3_GGUF_MAX_PAGE_WRITES_PER_SEC = "80"
$env:QWEN3_GGUF_TRIM_WORKING_SET = "on"       # trim llama-server after heavy turns
$env:QWEN3_GGUF_TRIM_BELOW_RAM_MB = "1536"
$env:QWEN3_GGUF_MAX_TEXT_TOKENS = "96"
$env:QWEN3_GGUF_MAX_AUDIO_TOKENS = "96"
```

If the physical-RAM guard fails, close other applications or lower the context before starting Qwen3. The request-time guard is softer: after the first audio request, low Available RAM is allowed if Windows is not actively reading/writing the page file. When Available RAM drops below `QWEN3_GGUF_TRIM_BELOW_RAM_MB`, the Flask app asks Windows to trim `llama-server`'s working set after the request completes. Lowering the hard guard lets the model run longer, but Windows may start using `pagefile.sys`, which is much slower. `QWEN3_GGUF_MLOCK=on` is available as an experiment, but on a 16 GB RAM machine it can increase memory pressure during load.

For a more aggressive experiment, try `q4_0`, but quality may drop:

```powershell
$env:QWEN3_GGUF_CACHE_TYPE_K = "q4_0"
$env:QWEN3_GGUF_CACHE_TYPE_V = "q4_0"
```

The selected runtime settings are written to `D:\QwenTemp\qwen3-llamacpp-state.json`.

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
