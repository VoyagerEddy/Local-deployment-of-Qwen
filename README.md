# Qwen Local Voice Chat

This app defaults to `Qwen3-Omni-30B-A3B-Instruct` through GGUF quantization and a local llama.cpp/Ollama-compatible API.

`runtime_defaults.json` contains the shared defaults for the Flask app and llama.cpp launcher. Environment variables override these defaults; `run_voice_chat.ps1` preserves those overrides. If `QWEN3_GGUF_BASE_URL` is unset, the app uses the configured `QWEN3_GGUF_PORT` on localhost. Request-time paging counters are queried only when available RAM is below the soft threshold.

## Paths

- Python environment: `D:\download\Qwen\_local\QwenEnvs\qwen25omni`
- Qwen3 GGUF and cache: `D:\download\Qwen\_local\QwenModels`
- Qwen3 local files: `D:\download\Qwen\_local\QwenModels\Qwen3-Omni-30B-A3B-Instruct-GGUF`
- Runtime data and temporary recordings: `D:\download\Qwen\_local\QwenData\qwen-voice`
- Browser VAD assets: `D:\download\Qwen\_local\QwenData\qwen-voice\vad-assets`
- Temporary files and logs: `D:\download\Qwen\_local\QwenTemp`
- App: `D:\download\Qwen\_local`

## Qwen3 GGUF Backend

Recommended route for realtime voice is llama.cpp with multimodal support:

```powershell
cd D:\download\Qwen\_local
.\run_qwen3_llamacpp.ps1
```

The script prefers local files when they exist:

```text
D:\download\Qwen\_local\QwenModels\Qwen3-Omni-30B-A3B-Instruct-GGUF\Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf
D:\download\Qwen\_local\QwenModels\Qwen3-Omni-30B-A3B-Instruct-GGUF\mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf
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

The selected runtime settings are written to `D:\download\Qwen\_local\QwenTemp\qwen3-llamacpp-state.json`.

Ollama route:

```powershell
cd D:\download\Qwen\_local
.\run_qwen3_ollama.ps1
```

Ollama compatibility for audio input depends on its current multimodal support. llama.cpp is the preferred backend for Qwen3-Omni realtime voice.

## App

### Desktop shortcut (recommended)

Create the desktop shortcut once:

```powershell
cd D:\download\Qwen\_local
.\install_shortcut.ps1
```

Double-click the Qwen desktop shortcut. The launcher starts the llama.cpp backend and Flask app in hidden processes, waits until both are ready, and opens `http://127.0.0.1:7860` in the default browser. Reopening the shortcut while the project is already running only opens the page; it does not start duplicate services.

The page sends a local heartbeat to Flask. When all Qwen pages are closed, the app waits 8 seconds (so a refresh does not stop it), then stops both Flask and the Qwen llama.cpp backend. If the browser exits unexpectedly without sending a close notification, the missed-heartbeat fallback also shuts them down.

Relevant controls:

```powershell
$env:QWEN_VOICE_STOP_ON_BROWSER_CLOSE = "on"
$env:QWEN_VOICE_BROWSER_HEARTBEAT_TIMEOUT_SECONDS = "45"
$env:QWEN_VOICE_BROWSER_SHUTDOWN_GRACE_SECONDS = "8"
```

Start the web app in another terminal:

```powershell
cd D:\download\Qwen\_local
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

## Generation Speed

The top of the page and the Run status area display generation speed (`token/s`), while the sidebar also shows the number of generated tokens. Statistics refresh every second for text and audio requests, and the average speed for the request remains visible after completion. The status indicates Preparing while waiting for the first token; statistics for the request are cleared on failure.

For llama.cpp, the app uses [the actual generation counts and timings returned by `timings_per_token`](https://github.com/ggml-org/llama.cpp/blob/b10711/tools/server/README.md). Speed is the cumulative average during generation and excludes input processing, speech synthesis, and playback. Token counts for Qwen3 audio requests include the model's transcription, answer, and JSON formatting. When a compatible backend does not return native timings, such as Ollama's OpenAI interface, the app estimates statistics from output chunks during generation and clearly marks them as approximate. The final `usage.completion_tokens` corrects the token count, but speed remains labeled as an estimate.

For the MNN fallback model, the status indicates Reported after completion. The app reads the native token count and decoding time after the response finishes generating; audio requests show statistics for the final answer stage. On a normal stop, the MNN count excludes the EOS token that was not output. When the generation limit is reached, the full count is retained.

`POST /ask-text` and `POST /ask-audio` accept a unique `generation_id`. `GET /generation-stats/<generation_id>` can retrieve statistics during processing without waiting for the inference lock. The response endpoints still return JSON, with the added `generation_id` and `generation` fields. Statistics are isolated per request. Completed records are kept for up to 15 minutes, with a total limit of 128 records.

## Standalone Qwen3-TTS Local Voice Output

Voice output uses the separate `Qwen3-TTS-12Hz-1.7B-Base` model to synthesize Qwen3-Omni's answer text into WAV audio. It requires its own GGUF and mmproj; the Talker weights in the existing Omni GGUF cannot directly serve as this TTS model. The browser receives and plays the audio returned by the backend. Every segment uses the same official Chelsie sample for reference voice cloning, preventing the voice from drifting between segments without speaker conditioning. This is a standalone TTS clone of the Chelsie reference voice, rather than output from Qwen3-Omni's native Talker with `speaker="Chelsie"`.

The page has a separate Voice output selector offering `Web TTS` and `Qwen3-TTS`. It defaults to Web TTS on the first visit and remembers subsequent selections. Local model synthesis is invoked only when Qwen3-TTS is selected. Flask's `GET /tts/status` returns `engine: "qwen3-tts"`, `ready`, `busy`, `error`, the model name, GPU layer count, and the status of the fixed Chelsie reference. `POST /tts` accepts `{"text": "Your text here", "language": "zh"}` and returns WAV audio on success. Synthesis runs through the local CLI, without a separate TTS HTTP service or an additional port.

First, run the setup script manually once:

```powershell
cd D:\download\Qwen\_local
.\setup_qwen3_tts.ps1
```

The script checks for an existing `llama-tts.exe`, downloads the models from [ggml-org's official GGUF repository](https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF), and downloads the [Chinese voice sample](https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20251126/jnleoh/Chelsie_ZH.wav) from the Chelsie entry in [Alibaba Cloud's official voice list](https://help.aliyun.com/zh/model-studio/multimodal-timbre-list). It verifies the complete file size and SHA256 of all three files:

```text
QwenModels\Qwen3-TTS-12Hz-1.7B-Base-GGUF\Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf
QwenModels\Qwen3-TTS-12Hz-1.7B-Base-GGUF\mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
QwenModels\Qwen3-TTS-12Hz-1.7B-Base-GGUF\voices\Chelsie.wav
```

The two model files total about 1.48 GB. The script pins the official repository revision to `ca27d74bc954b73dadab5b71ca265d87fc861a7c`. The Chelsie sample is 347,564 bytes, 24 kHz, mono, 16-bit PCM, and 7.24 seconds long, with SHA256 `e2461d0e0fc2bf1e083ea7af40c3b6849e5912b2744162cab6e151c366097b53`. Verified files are not downloaded again. Interrupted downloads retain a `.partial` file for resuming. The script supports an existing aria2c installation or Windows curl and reads the Windows system proxy. `QwenModels` and generated WAV files are local runtime data and are not committed to Git. Starting the app normally does not automatically run the setup script or download models.

Verify existing files only, or run a short Chinese synthesis check:

```powershell
.\setup_qwen3_tts.ps1 -VerifyOnly
.\setup_qwen3_tts.ps1 -VerifyOnly -Test
```

`-VerifyOnly` checks both model files and the Chelsie sample; a missing sample or a failed verification produces an error. `-Test` writes a Chinese sentence meaning "Hello, this is a local voice test." to a UTF-8 text file without a BOM, synthesizes it on the CPU with the fixed `--tts-speaker-file` and `--seed 42`, and creates a uniquely named WAV in `QwenTemp`. It does not play audio, record the microphone, or restart an already running Omni service. `-ModelDir` or the `QWEN3_TTS_MODEL_DIR` environment variable can change the model directory, and `-TestText` can change the test text. The Chelsie sample always remains at the `voices\Chelsie.wav` path shown above under the app directory; changes to the model directory environment configuration do not move it.

The default is CPU synthesis to avoid competing with Omni for GPU memory. Available path and runtime overrides:

```powershell
$env:QWEN3_TTS_MODEL_DIR = "D:\download\Qwen\_local\QwenModels\Qwen3-TTS-12Hz-1.7B-Base-GGUF"
# Optional absolute paths; normally inferred from the directory above:
$env:QWEN3_TTS_MODEL = "$env:QWEN3_TTS_MODEL_DIR\Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"
$env:QWEN3_TTS_MMPROJ = "$env:QWEN3_TTS_MODEL_DIR\mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
# Set only if llama-tts.exe cannot be found automatically:
$env:QWEN3_TTS_LLAMA_TTS = "C:\path\to\llama-tts.exe"
$env:QWEN3_TTS_GPU_LAYERS = "0"
$env:QWEN3_TTS_THREADS = "4"
$env:QWEN3_TTS_MAX_FRAMES = "120"
$env:QWEN3_TTS_TIMEOUT_SECONDS = "600"
$env:QWEN3_TTS_MIN_FREE_RAM_MB = "4096"
```

Restart the Flask app after changing environment variables; the already running Omni service on port 8080 can continue to be used. `llama-tts.exe` is a separate executable, and the local b10711 build of `llama.exe` has no `tts` subcommand. The [upstream TTS documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md) demonstrates the Base model and reference audio workflow; this does not establish that CustomVoice or VoiceDesign modes are integrated.

The Chelsie sample is required. The app reports a clear error if it is missing or fails verification; it does not omit reference conditioning to generate a random voice or automatically switch back to Web TTS. The old `QWEN3_TTS_SPEAKER_FILE` custom reference setting is ignored, and all segments use the fixed official Chelsie sample. `--seed 42` fixes the sampling randomness, but a seed alone cannot select the Chelsie voice. Voice conditioning comes from the speaker embedding in the reference audio. The current llama.cpp path clones the voice using that embedding, without the full ICL conditioning that includes the reference transcript. [Qwen's official guidance](https://github.com/QwenLM/Qwen3-TTS#voice-clone) notes that this mode can reduce cloning quality. Intonation and prosody still vary with the text, so reference cloning does not guarantee a voice identical to Omni's native Chelsie.

The app splits long answers at punctuation and spaces. Segments containing Chinese are limited to 24 characters; other segments are limited to 80 characters. Each segment is synthesized separately by the CLI, then concatenated into a single WAV response. This avoids sending the entire long answer to the waveform decoder at once, though long answers still take more time. `QWEN3_TTS_MAX_FRAMES` defaults to 120 and limits the number of frames per segment, about 10 seconds at 12 Hz. Setting it too low may truncate speech; setting it too high permits greater memory use. The total synthesis timeout defaults to 600 seconds.

The app's CPU settings also disable GPU offload for mmproj, operators, and the KV cache. `QWEN3_TTS_MIN_FREE_RAM_MB` defaults to 4096 as a target for the memory protection strategy, rather than a hard minimum requirement. On Windows, if available RAM is low after Omni answers, the app first trims the local llama.cpp process's working set, then checks whether TTS can start; the model process keeps running. Question answering and TTS share an inference lock. Other inference requests quickly return a busy response, while status endpoints remain accessible.

Selecting Stop speaking or changing the voice output stops browser playback and discards the current audio result. Synthesis that has already started on the server continues until completion or timeout, and new inference requests report busy during that time. VAD resumes after playback ends to avoid treating synthesized speech as a user recording.

On 2026-10-01, a real two-segment Chinese synthesis test using the local b10711 build, the fixed Chelsie reference, and seed 42 through Flask's test client returned HTTP 200. The concatenated WAV was 24 kHz, mono, 16-bit PCM, and 8.74 seconds long. Wall-clock time, including two model loads, was 21.17 seconds. The test confirmed that both segments shared the reference path and produced valid audio; it did not establish perceptual identity with native Chelsie or the memory limits for long answers. The app processes TTS serially and uses conservative segmentation under the limits above. The setup script's `-Test` uses 120 frames, the fixed Chelsie reference, and seed 42, with GPU offload disabled for mmproj, operators, and the KV cache.
