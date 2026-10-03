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

Double-click `Qwen 本地多模态` on the desktop. The launcher starts the llama.cpp backend and Flask app in hidden processes, waits until both are ready, and opens `http://127.0.0.1:7860` in the default browser. Reopening the shortcut while the project is already running only opens the page; it does not start duplicate services.

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

## 生成速度

页面顶部和“运行状态”显示生成速度（`token/s`），侧栏同时显示已生成 token 数。文字和语音请求每秒刷新统计，完成后保留本次生成的平均速度；等待首个 token 时显示“准备中”，失败时清除本次统计。

llama.cpp 使用 [`timings_per_token` 返回的真实生成计数和计时](https://github.com/ggml-org/llama.cpp/blob/b10711/tools/server/README.md)，速度为生成阶段的累计平均值，不包含输入处理、语音合成和播放。Qwen3 语音请求中的 token 数包含模型输出的转写、回答及 JSON 格式。兼容后端未返回原生计时（如 Ollama 的 OpenAI 接口）时，运行中按输出分片估算，并明确显示“约”；完成后的 `usage.completion_tokens` 会修正 token 数，速度仍标注为估算。

MNN 备用模型显示“完成后统计”，回复生成结束后读取原生 token 数和解码计时；语音请求显示最后回答阶段的统计。MNN 正常停止时从计数中扣除未输出的 EOS，达到生成上限时保留完整计数。

`POST /ask-text` 和 `POST /ask-audio` 可传入唯一的 `generation_id`，`GET /generation-stats/<generation_id>` 可在处理期间读取统计，无需等待推理锁。原回复接口仍返回 JSON，新增 `generation_id` 与 `generation` 字段。统计按请求隔离，完成记录保留最多 15 分钟，总记录数限制为 128。

## 独立 Qwen3-TTS 本地语音输出

语音输出使用独立的 `Qwen3-TTS-12Hz-1.7B-Base`，把 Qwen3-Omni 的回答文字合成为 WAV。它需要自己的 GGUF 和 mmproj；现有 Omni GGUF 的 Talker 权重不能直接充当这个 TTS 模型。浏览器接收并播放后端返回的音频。每个分段固定使用同一份官方 Chelsie 样音作参考克隆，避免没有说话人条件时分段音色漂移；这是独立 TTS 对 Chelsie 参考音色的克隆，不是 Qwen3-Omni 原生 Talker 的 `speaker="Chelsie"` 输出。

页面有独立的“声音输出”选择框，提供 `Web TTS` 和 `Qwen3-TTS`，首次打开默认保留 Web TTS，并记住之后的选择。选中 Qwen3-TTS 后才调用本地模型合成。Flask 的 `GET /tts/status` 返回 `engine: "qwen3-tts"`、`ready`、`busy`、`error`、模型名、GPU 层数和固定 Chelsie 参考状态；`POST /tts` 接收 `{"text": "要朗读的文字", "language": "zh"}`，成功时返回 WAV。它通过本机 CLI 合成，无需另启 TTS HTTP 服务或占用新的端口。

先手动运行一次准备脚本：

```powershell
cd D:\download\Qwen\_local
.\setup_qwen3_tts.ps1
```

脚本检查已有 `llama-tts.exe`，从 [ggml-org 官方 GGUF 仓库](https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF)下载模型，并从[阿里云官方音色表](https://help.aliyun.com/zh/model-studio/multimodal-timbre-list)的 Chelsie 行下载[中文样音](https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20251126/jnleoh/Chelsie_ZH.wav)。三个文件均核对完整大小和 SHA256：

```text
QwenModels\Qwen3-TTS-12Hz-1.7B-Base-GGUF\Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf
QwenModels\Qwen3-TTS-12Hz-1.7B-Base-GGUF\mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
QwenModels\Qwen3-TTS-12Hz-1.7B-Base-GGUF\voices\Chelsie.wav
```

两个模型文件合计约 1.48 GB。脚本固定官方仓库 revision `ca27d74bc954b73dadab5b71ca265d87fc861a7c`；Chelsie 样音为 347,564 字节、24 kHz、单声道、16-bit PCM、7.24 秒，SHA256 为 `e2461d0e0fc2bf1e083ea7af40c3b6849e5912b2744162cab6e151c366097b53`。已校验文件不会重复下载，中断下载保留 `.partial` 以便续传。脚本支持已有 aria2c 或 Windows curl，会读取 Windows 系统代理。`QwenModels` 和生成的 WAV 属于本机运行数据，不提交到 Git。正常启动应用不会自动调用准备脚本或下载模型。

只验证已有文件，或执行短中文合成检查：

```powershell
.\setup_qwen3_tts.ps1 -VerifyOnly
.\setup_qwen3_tts.ps1 -VerifyOnly -Test
```

`-VerifyOnly` 同时检查两个模型文件和 Chelsie 样音；缺少样音或校验失败会报错。`-Test` 将“你好，这是本地语音测试。”写入不带 BOM 的 UTF-8 文本文件，使用 CPU、固定 `--tts-speaker-file` 和 `--seed 42` 合成，并在 `QwenTemp` 生成一个唯一命名的 WAV；它不会播放音频、录制麦克风或重启已经运行的 Omni 服务。`-ModelDir` 或环境变量 `QWEN3_TTS_MODEL_DIR` 可以改变模型目录，`-TestText` 可以改变测试文字。Chelsie 样音始终保存在应用目录下上述 `voices\Chelsie.wav`，不随模型目录环境配置改变。

默认使用 CPU，避免和 Omni 争用 GPU 显存。可覆盖的路径参数：

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

修改环境变量后需要重新启动 Flask 应用，已经运行的 Omni 8080 服务可以继续使用。`llama-tts.exe` 是独立可执行文件，本机 b10711 的 `llama.exe` 没有 `tts` 子命令。[上游 TTS 文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md)展示的是 Base 模型和参考音频流程，不能据此承诺 CustomVoice／VoiceDesign 模式也已经接入。

Chelsie 样音是必需资产；应用缺少样音或校验失败时明确报错，不会省略参考条件生成随机音色，也不会自动切回 Web TTS。旧的 `QWEN3_TTS_SPEAKER_FILE` 自定义参考配置被忽略，所有分段都使用固定的官方 Chelsie 样音。`--seed 42` 固定采样随机数，但单独设置 seed 不能指定 Chelsie 音色；音色条件来自参考音频的说话人向量。当前 llama.cpp 路径使用该向量进行克隆，没有使用参考转写文本的完整 ICL 条件；[Qwen 官方说明](https://github.com/QwenLM/Qwen3-TTS#voice-clone)指出这种模式的克隆质量可能降低。语调和韵律仍会随文字变化，不能把参考克隆承诺为与 Omni 原生 Chelsie 完全相同的声音。

应用按标点和空格把长回答分段，含中文的段最多 24 个字符，其他段最多 80 个字符；每段由 CLI 单独合成，然后拼接为一个 WAV 返回。这样避免把整个长回答一次送入波形解码器，但长回答仍需要更长时间。`QWEN3_TTS_MAX_FRAMES` 默认 120，限制的是每段帧数，12 Hz 下约 10 秒；调得过小可能截断语音，调得过大会允许更高内存开销。总合成超时默认 600 秒。

应用的 CPU 参数还关闭 mmproj、算子和 KV cache 的 GPU offload。`QWEN3_TTS_MIN_FREE_RAM_MB` 默认 4096，是内存保护策略的目标，不是硬性最低要求。Windows 上若 Omni 回答后可用内存不足，应用先回收该本地 llama.cpp 进程的工作集，再检查是否能启动 TTS；模型进程继续运行。问答与 TTS 共用推理锁，其他推理请求快速返回忙碌提示，状态接口仍可访问。

“停止播报”或切换语音输出会停止前端播放并放弃本次语音结果；已经在服务器启动的合成会继续到结束或超时，期间新的推理请求会提示忙碌。播放结束后恢复 VAD，避免把合成声音当成用户录音。

2026-10-01 用本机 b10711、固定 Chelsie 参考和 seed 42，经 Flask 测试客户端执行真实两段中文合成，返回 HTTP 200；拼接 WAV 为 24 kHz、单声道、16-bit PCM，长度 8.74 秒，含两次模型加载的墙钟耗时 21.17 秒。已验证两段共用参考路径并生成有效音频，该结果不代表与原生 Chelsie 的听感完全相同或长回答内存上限。应用串行处理 TTS，按上述限制保守分段。准备脚本的 `-Test` 使用 120 帧、固定 Chelsie 参考、seed 42，并关闭 mmproj、算子和 KV cache 的 GPU offload。
