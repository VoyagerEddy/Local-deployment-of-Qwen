import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

import MNN.llm as mnn_llm


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("QWEN25_OMNI_DATA_DIR", r"D:\CodexData\qwen25-omni-voice"))
RECORDINGS_DIR = DATA_DIR / "recordings"
VAD_ASSETS_DIR = DATA_DIR / "vad-assets"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
VAD_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIG = Path(
    os.environ.get(
        "QWEN25_OMNI_MNN_CONFIG",
        r"D:\CodexModels\Qwen2.5-Omni-3B-MNN\config.json",
    )
)

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
CHAT_SYSTEM_PROMPT = (
    "You are Qwen, a helpful assistant developed by the Qwen Team, Alibaba Group."
)
ASR_SYSTEM_PROMPT = (
    "You are a speech-to-text transcription engine. "
    "Only output the words actually spoken in the audio. "
    "Never answer questions and never add facts."
)
DEFAULT_AUDIO_PROMPT = (
    "\u8bf7\u6839\u636e\u7528\u6237\u8bed\u97f3\u8f6c\u5199\u5185\u5bb9\u81ea\u7136\u3001\u7b80\u77ed\u5730\u56de\u7b54\u3002"
    "\u9664\u975e\u7528\u6237\u53e6\u6709\u8981\u6c42\uff0c\u5426\u5219\u4f7f\u7528\u7528\u6237\u8bf4\u8bdd\u7684\u8bed\u8a00\u3002"
)
TRANSCRIBE_PROMPT = (
    "\u8bf7\u9010\u5b57\u8f6c\u5199\u8fd9\u6bb5\u97f3\u9891\uff0c\u53ea\u8f93\u51fa\u539f\u8bdd\u3002"
)
TRANSCRIBE_RETRY_PROMPT = (
    "Transcribe the audio verbatim. Output spoken words only."
)
INTERNAL_PROMPT_MARKERS = (
    "\u4f60\u73b0\u5728\u53ea\u80fd\u6267\u884c",
    "ASR \u8bed\u97f3\u8bc6\u522b\u4efb\u52a1",
    "\u8bf7\u9010\u5b57\u8f6c\u5199",
    "\u53ea\u8f93\u51fa\u8f6c\u5199\u6587\u672c",
    "\u4e0d\u8981\u56de\u7b54\u95ee\u9898",
    "\u4e0d\u8981\u81ea\u6211\u4ecb\u7ecd",
    "\u4e0d\u8981\u89e3\u91ca",
    "\u5982\u679c\u97f3\u9891\u91cc\u7684\u4eba\u95ee\u4f60\u662f\u4ec0\u4e48\u6a21\u578b",
    "\u5982\u679c\u6ca1\u6709\u6e05\u695a\u7684\u4eba\u58f0",
    "ASR only",
    "Transcribe the speech",
    "Output only the words",
    "Do not answer",
    "do not say you are Qwen",
    "speech-to-text",
    "transcription engine",
    "Never answer questions",
    "never add facts",
    "\u8bf7\u9010\u5b57\u8f6c\u5199\u8fd9\u6bb5\u97f3\u9891",
    "\u53ea\u8f93\u51fa\u539f\u8bdd",
)

app = Flask(__name__)
model_lock = threading.Lock()
model = None
model_loaded_at = None


def cjk_score(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def repair_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    candidates = [text]
    for encoding in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except UnicodeError:
            pass

    return max(candidates, key=lambda item: (cjk_score(item), -item.count("\ufffd")))


def strip_internal_prompt(text: str) -> str:
    cleaned = text
    lowered = cleaned.lower()
    cut_at = None
    for marker in INTERNAL_PROMPT_MARKERS:
        index = lowered.find(marker.lower())
        if index > 0:
            cut_at = index if cut_at is None else min(cut_at, index)
    if cut_at is not None:
        cleaned = cleaned[:cut_at]
    return cleaned.strip(" \t\r\n\"'\u201c\u201d\u2018\u2019\uff0c,.;\u3002\uff1b;:\uff1a")


def clean_transcript(text: str) -> str:
    transcript = repair_text(text).strip()
    transcript = transcript.replace("```", "").strip()
    for prefix in (
        "\u8f6c\u5199\u6587\u672c\uff1a",
        "\u8f6c\u5199\uff1a",
        "\u6587\u672c\uff1a",
        "\u8bed\u97f3\u8f6c\u5199\uff1a",
        "Transcript:",
        "Transcription:",
    ):
        if transcript.lower().startswith(prefix.lower()):
            transcript = transcript[len(prefix) :].strip()
            break
    transcript = strip_internal_prompt(transcript)
    transcript = transcript.strip(" \t\r\n\"'\u201c\u201d\u2018\u2019")
    transcript = re.sub(r"([\u7684\u5417\u5462\u5427\u554a])\1+$", r"\1", transcript)
    return transcript or "\uff08\u672a\u8bc6\u522b\u5230\u8bed\u97f3\uff09"


def looks_like_assistant_answer(text: str) -> bool:
    normalized = repair_text(text).replace(" ", "").lower()
    assistant_markers = (
        "\u6211\u662fqwen",
        "\u6211\u662fqqwen",
        "\u5927\u6a21\u578b\u7684ai\u52a9\u624b",
        "\u6709\u4ec0\u4e48\u53ef\u4ee5\u5e2e\u52a9",
        "\u6211\u53ef\u4ee5\u5e2e\u52a9",
        "\u6211\u80fd\u5e2e\u52a9",
    )
    return any(marker in normalized for marker in assistant_markers)


def load_model():
    global model, model_loaded_at
    if model is not None:
        return model

    if not MODEL_CONFIG.exists():
        raise FileNotFoundError(f"Model config not found: {MODEL_CONFIG}")

    started = time.time()
    qwen = mnn_llm.create(str(MODEL_CONFIG))
    qwen.load()
    try:
        qwen.set_config(
            {
                "max_new_tokens": 180,
                "temperature": 0.6,
                "topP": 0.9,
                "reuse_kv": False,
            }
        )
    except Exception:
        pass

    model = qwen
    model_loaded_at = time.time() - started
    return model


def convert_to_wav(source: Path, target: Path):
    command = [
        FFMPEG,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(target),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:])


def configure_model(qwen, system_prompt: str, max_new_tokens: int, temperature: float):
    try:
        qwen.set_config(
            {
                "system_prompt": system_prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "topP": 0.9,
                "reuse_kv": False,
            }
        )
    except Exception:
        pass


def ask_model(
    prompt: str,
    reset: bool = True,
    system_prompt: str = CHAT_SYSTEM_PROMPT,
    max_new_tokens: int = 180,
    temperature: float = 0.6,
) -> str:
    qwen = load_model()
    with model_lock:
        if reset:
            qwen.reset()
        configure_model(qwen, system_prompt, max_new_tokens, temperature)
        response = qwen.response(prompt, False)
        if reset:
            qwen.reset()
    return repair_text(response).strip()


def transcribe_audio(audio_path: str) -> str:
    transcript_prompt = f"<audio>{audio_path}</audio>{TRANSCRIBE_PROMPT}"
    transcript = clean_transcript(
        ask_model(
            transcript_prompt,
            system_prompt=ASR_SYSTEM_PROMPT,
            max_new_tokens=48,
            temperature=0.0,
        )
    )
    if looks_like_assistant_answer(transcript):
        retry_prompt = f"<audio>{audio_path}</audio>{TRANSCRIBE_RETRY_PROMPT}"
        transcript = clean_transcript(
            ask_model(
                retry_prompt,
                system_prompt=ASR_SYSTEM_PROMPT,
                max_new_tokens=48,
                temperature=0.0,
            )
        )
    return transcript


def delete_if_exists(path: Path):
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def upload_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".wav", ".webm", ".ogg", ".m4a", ".mp3", ".flac"}:
        return suffix
    return ".audio"


@app.get("/")
def index():
    return render_template("index.html", model_config=str(MODEL_CONFIG))


@app.get("/vad-assets/<path:filename>")
def vad_assets(filename):
    mimetype = None
    if filename.endswith(".wasm"):
        mimetype = "application/wasm"
    elif filename.endswith(".onnx"):
        mimetype = "application/octet-stream"
    elif filename.endswith((".js", ".mjs")):
        mimetype = "application/javascript"
    return send_from_directory(VAD_ASSETS_DIR, filename, mimetype=mimetype)


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "model_config": str(MODEL_CONFIG),
            "data_dir": str(DATA_DIR),
            "vad_assets_dir": str(VAD_ASSETS_DIR),
            "model_loaded": model is not None,
            "load_seconds": model_loaded_at,
        }
    )


@app.post("/ask-text")
def ask_text():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    started = time.time()
    reply = ask_model(text)
    return jsonify({"reply": reply, "elapsed": round(time.time() - started, 2)})


@app.post("/ask-audio")
def ask_audio():
    upload = request.files.get("audio")
    if upload is None:
        return jsonify({"error": "missing audio file"}), 400

    instruction = (request.form.get("instruction") or DEFAULT_AUDIO_PROMPT).strip()
    request_id = uuid.uuid4().hex
    source = RECORDINGS_DIR / f"{request_id}{upload_suffix(upload.filename)}"
    wav = RECORDINGS_DIR / f"{request_id}.16k.wav"
    upload.save(source)

    started = time.time()
    try:
        convert_to_wav(source, wav)
        audio_path = wav.as_posix()
        transcript = transcribe_audio(audio_path)
        reply_prompt = (
            f"{instruction}\n\n"
            f"\u7528\u6237\u8bed\u97f3\u8f6c\u5199\u5982\u4e0b\uff1a{transcript}\n\n"
            "\u8bf7\u76f4\u63a5\u56de\u7b54\u7528\u6237\uff0c\u4e0d\u8981\u590d\u8ff0\u201c\u7528\u6237\u8bed\u97f3\u8f6c\u5199\u5982\u4e0b\u201d\u3002"
        )
        reply = ask_model(reply_prompt)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        delete_if_exists(source)
        delete_if_exists(wav)

    return jsonify(
        {
            "transcript": transcript,
            "reply": reply,
            "elapsed": round(time.time() - started, 2),
        }
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    load_model()
    app.run(host="127.0.0.1", port=7860, threaded=True)
