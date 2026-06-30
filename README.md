# Qwen2.5-Omni-3B Voice

This local app uses the 4-bit MNN export of `Qwen2.5-Omni-3B`.

## Paths

- Python environment: `D:\CodexEnvs\qwen25omni`
- Model: `D:\CodexModels\Qwen2.5-Omni-3B-MNN`
- Runtime data and temporary recordings: `D:\CodexData\qwen25-omni-voice`
- Browser VAD assets: `D:\CodexData\qwen25-omni-voice\vad-assets`
- App: this folder

## Run

```powershell
cd C:\Users\mechrevo\Documents\Codex\2026-06-27\q\outputs\qwen25-omni-voice
.\run_voice_chat.ps1
```

Open:

```text
http://127.0.0.1:7860
```

## Notes

- The backend first asks the same Qwen2.5-Omni-3B-MNN model to transcribe recorded audio, then shows that transcript in the user bubble and asks the same model to answer it.
- The browser uses Silero VAD v5 through `@ricky0123/vad-web`; it sends a full detected speech segment to Qwen after about 420 ms of silence. The v5 frame is 512 samples at 16 kHz, about 32 ms. ONNX Runtime Web assets, including `.mjs`, `.wasm`, and `.onnx` files, live under `D:\CodexData\qwen25-omni-voice\vad-assets`.
- The browser speaks replies with local Web Speech TTS because the MNN Python binding does not expose the native `generateWavform` callback.
- On this PC, expect push-to-talk latency rather than low-latency full-duplex conversation.
