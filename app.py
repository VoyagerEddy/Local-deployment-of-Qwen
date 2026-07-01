import os
import re
import json
import base64
import ctypes
import http.client
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("QWEN_VOICE_DATA_DIR", r"D:\QwenData\qwen-voice"))
RECORDINGS_DIR = DATA_DIR / "recordings"
VAD_ASSETS_DIR = DATA_DIR / "vad-assets"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
VAD_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIG = Path(
    os.environ.get(
        "QWEN25_OMNI_MNN_CONFIG",
        r"D:\QwenModels\Qwen2.5-Omni-3B-MNN\config.json",
    )
)

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
DEFAULT_BACKEND = os.environ.get("QWEN_VOICE_DEFAULT_BACKEND", "qwen3-gguf")
QWEN3_GGUF_BASE_URL = os.environ.get("QWEN3_GGUF_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
QWEN3_GGUF_MODEL = os.environ.get(
    "QWEN3_GGUF_MODEL",
    "qwen3-omni-30b-a3b-instruct-q4km",
)
QWEN3_GGUF_CONTEXT = int(os.environ.get("QWEN3_GGUF_CONTEXT", "768"))
QWEN3_GGUF_GPU_LAYERS = os.environ.get("QWEN3_GGUF_GPU_LAYERS", "auto")
QWEN3_GGUF_MIN_FREE_VRAM_MB = os.environ.get("QWEN3_GGUF_MIN_FREE_VRAM_MB", "768")
QWEN3_GGUF_CACHE_TYPE_K = os.environ.get("QWEN3_GGUF_CACHE_TYPE_K", "q8_0")
QWEN3_GGUF_CACHE_TYPE_V = os.environ.get("QWEN3_GGUF_CACHE_TYPE_V", "q8_0")
QWEN3_GGUF_FLASH_ATTN = os.environ.get("QWEN3_GGUF_FLASH_ATTN", "on")
QWEN3_GGUF_MLOCK = os.environ.get("QWEN3_GGUF_MLOCK", "off")
QWEN3_GGUF_PARALLEL = os.environ.get("QWEN3_GGUF_PARALLEL", "1")
QWEN3_GGUF_BATCH_SIZE = os.environ.get("QWEN3_GGUF_BATCH_SIZE", "256")
QWEN3_GGUF_UBATCH_SIZE = os.environ.get("QWEN3_GGUF_UBATCH_SIZE", "64")
QWEN3_GGUF_PROMPT_CACHE = os.environ.get("QWEN3_GGUF_PROMPT_CACHE", "off")
QWEN3_GGUF_PROMPT_CACHE_RAM_MB = os.environ.get("QWEN3_GGUF_PROMPT_CACHE_RAM_MB", "0")
QWEN3_GGUF_CTX_CHECKPOINTS = os.environ.get("QWEN3_GGUF_CTX_CHECKPOINTS", "0")
QWEN3_GGUF_MIN_FREE_RAM_MB = os.environ.get("QWEN3_GGUF_MIN_FREE_RAM_MB", "1024")
QWEN3_GGUF_MIN_REQUEST_RAM_MB = int(os.environ.get("QWEN3_GGUF_MIN_REQUEST_RAM_MB", "1024"))
QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB = int(
    os.environ.get("QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB", "512")
)
QWEN3_GGUF_MAX_PAGE_READS_PER_SEC = int(
    os.environ.get("QWEN3_GGUF_MAX_PAGE_READS_PER_SEC", "120")
)
QWEN3_GGUF_MAX_PAGE_WRITES_PER_SEC = int(
    os.environ.get("QWEN3_GGUF_MAX_PAGE_WRITES_PER_SEC", "80")
)
QWEN3_GGUF_TRIM_WORKING_SET = os.environ.get("QWEN3_GGUF_TRIM_WORKING_SET", "on")
QWEN3_GGUF_TRIM_BELOW_RAM_MB = int(os.environ.get("QWEN3_GGUF_TRIM_BELOW_RAM_MB", "1536"))
QWEN3_GGUF_MAX_TEXT_TOKENS = int(os.environ.get("QWEN3_GGUF_MAX_TEXT_TOKENS", "96"))
QWEN3_GGUF_MAX_AUDIO_TOKENS = int(os.environ.get("QWEN3_GGUF_MAX_AUDIO_TOKENS", "96"))
QWEN3_LLAMA_STATE_FILE = Path(
    os.environ.get("QWEN3_LLAMA_STATE_FILE", r"D:\QwenTemp\qwen3-llamacpp-state.json")
)
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("QWEN3_GGUF_TIMEOUT", "600"))
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
mnn_model_lock = threading.Lock()
mnn_model = None
mnn_model_loaded_at = None

BACKENDS = {
    "qwen3-gguf": {
        "id": "qwen3-gguf",
        "name": "Qwen3-Omni-30B-A3B-Instruct GGUF",
        "detail": "llama.cpp/Ollama Q4_K_M, ctx 768, ngl auto, KV q8_0, FA on, RAM guard",
        "audio": True,
    },
    "qwen25-mnn": {
        "id": "qwen25-mnn",
        "name": "Qwen2.5-Omni-3B MNN",
        "detail": "local MNN 4-bit",
        "audio": True,
    },
}


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


def normalize_backend(backend_id: str | None) -> str:
    if backend_id in BACKENDS:
        return backend_id
    return DEFAULT_BACKEND if DEFAULT_BACKEND in BACKENDS else "qwen3-gguf"


def load_mnn_model():
    global mnn_model, mnn_model_loaded_at
    if mnn_model is not None:
        return mnn_model

    if not MODEL_CONFIG.exists():
        raise FileNotFoundError(f"Model config not found: {MODEL_CONFIG}")

    import MNN.llm as mnn_llm

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

    mnn_model = qwen
    mnn_model_loaded_at = time.time() - started
    return mnn_model


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


def ask_mnn_model(
    prompt: str,
    reset: bool = True,
    system_prompt: str = CHAT_SYSTEM_PROMPT,
    max_new_tokens: int = 180,
    temperature: float = 0.6,
) -> str:
    qwen = load_mnn_model()
    with mnn_model_lock:
        if reset:
            qwen.reset()
        configure_model(qwen, system_prompt, max_new_tokens, temperature)
        response = qwen.response(prompt, False)
        if reset:
            qwen.reset()
    return repair_text(response).strip()


def openai_chat_completion(
    messages: list[dict],
    max_new_tokens: int = 180,
    temperature: float = 0.6,
) -> str:
    ready, error = qwen3_backend_status()
    if not ready:
        clear_qwen3_state()
        raise RuntimeError(
            f"Qwen3 llama.cpp 后端当前未在线：{error}。"
            "请等待或重新运行 run_qwen3_llamacpp.ps1，然后再试。"
        )

    payload = {
        "model": QWEN3_GGUF_MODEL,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    request = urllib.request.Request(
        f"{QWEN3_GGUF_BASE_URL}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise RuntimeError("Qwen3 GGUF 响应超时；30B Q4_K_M 首次加载会很慢。") from exc
    except (ConnectionResetError, ConnectionAbortedError, http.client.RemoteDisconnected) as exc:
        clear_qwen3_state()
        raise RuntimeError(qwen3_disconnected_message(exc)) from exc
    except urllib.error.URLError as exc:
        if is_connection_reset(exc):
            clear_qwen3_state()
            raise RuntimeError(qwen3_disconnected_message(exc)) from exc
        raise RuntimeError(
            "Qwen3 GGUF 服务未连接。请先启动 llama.cpp 或 Ollama 后端："
            " llama.cpp 默认 http://127.0.0.1:8080/v1，"
            "Ollama 默认 http://127.0.0.1:11434/v1。"
        ) from exc

    finally:
        trim_qwen3_working_set()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Qwen3 GGUF 返回格式异常: {data}") from exc


def read_llama_state() -> dict | None:
    try:
        return json.loads(QWEN3_LLAMA_STATE_FILE.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def is_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def clear_qwen3_state() -> None:
    try:
        QWEN3_LLAMA_STATE_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def is_connection_reset(exc: BaseException) -> bool:
    text = str(exc)
    reason = getattr(exc, "reason", None)
    return (
        "10054" in text
        or "forcibly closed" in text.lower()
        or isinstance(reason, ConnectionResetError)
        or (reason is not None and "10054" in str(reason))
    )


def qwen3_disconnected_message(exc: BaseException) -> str:
    return (
        "Qwen3 llama.cpp 后端在处理请求时断开连接。"
        "这通常是音频请求触发 llama.cpp/Vulkan 显存或内存压力后进程退出。"
        "我已清除旧运行状态；请等待后端重新启动完成，或重新运行 run_qwen3_llamacpp.ps1 后再试。"
        f"原始错误：{exc}"
    )


def qwen3_backend_status(timeout: float = 1.5) -> tuple[bool, str | None]:
    try:
        with urllib.request.urlopen(f"{QWEN3_GGUF_BASE_URL}/models", timeout=timeout) as response:
            if 200 <= response.status < 300:
                return True, None
            return False, f"HTTP {response.status}"
    except Exception as exc:
        return False, str(exc)


def qwen3_runtime_snapshot() -> tuple[bool, str | None, dict | None, bool]:
    ready, error = qwen3_backend_status()
    state = read_llama_state()
    return ready, error, state if ready else None, (not ready and state is not None)


def get_available_ram_mb() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPhys // (1024 * 1024))


def get_memory_pressure_summary(include_perf: bool = False) -> dict:
    summary = {
        "available_mb": get_available_ram_mb(),
        "pages_per_sec": None,
        "page_reads_per_sec": None,
        "page_writes_per_sec": None,
    }
    if os.name != "nt" or not include_perf:
        return summary

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory;"
            "[pscustomobject]@{"
            "pages_per_sec=[int]$m.PagesPersec;"
            "page_reads_per_sec=[int]$m.PageReadsPersec;"
            "page_writes_per_sec=[int]$m.PageWritesPersec"
            "} | ConvertTo-Json -Compress"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode == 0 and completed.stdout.strip():
            perf = json.loads(completed.stdout)
            summary["pages_per_sec"] = int(perf.get("pages_per_sec") or 0)
            summary["page_reads_per_sec"] = int(perf.get("page_reads_per_sec") or 0)
            summary["page_writes_per_sec"] = int(perf.get("page_writes_per_sec") or 0)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        pass
    return summary


def trim_qwen3_working_set() -> bool:
    if os.name != "nt" or not is_enabled(QWEN3_GGUF_TRIM_WORKING_SET):
        return False

    available_mb = get_available_ram_mb()
    if (
        available_mb is None
        or QWEN3_GGUF_TRIM_BELOW_RAM_MB <= 0
        or available_mb >= QWEN3_GGUF_TRIM_BELOW_RAM_MB
    ):
        return False

    state = read_llama_state() or {}
    process_id = state.get("process_id")
    if not process_id:
        return False

    try:
        pid = int(process_id)
    except (TypeError, ValueError):
        return False

    process_set_quota = 0x0100
    process_query_information = 0x0400
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
    psapi.EmptyWorkingSet.restype = ctypes.c_int

    handle = kernel32.OpenProcess(process_set_quota | process_query_information, False, pid)
    if not handle:
        return False
    try:
        return bool(psapi.EmptyWorkingSet(handle))
    finally:
        kernel32.CloseHandle(handle)


def assert_qwen3_request_memory() -> None:
    memory = get_memory_pressure_summary(include_perf=True)
    available_mb = memory.get("available_mb")
    if available_mb is None:
        return

    available_mb = int(available_mb)
    page_reads = memory.get("page_reads_per_sec")
    page_writes = memory.get("page_writes_per_sec")

    if QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB > 0 and available_mb < QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB:
        raise RuntimeError(
            f"Qwen3 \u5f53\u524d\u53ef\u7528\u5185\u5b58\u53ea\u6709 {available_mb}MB\uff0c"
            f"\u4f4e\u4e8e\u786c\u4e0b\u9650 {QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB}MB\uff0c"
            "\u5df2\u963b\u6b62\u672c\u6b21\u8bf7\u6c42\u4ee5\u907f\u514d Windows \u6362\u9875\u6216\u540e\u7aef\u65ad\u5f00\u3002"
            "\u8bf7\u5173\u95ed\u5176\u4ed6\u7a0b\u5e8f\u6216\u91cd\u542f Qwen3 \u540e\u7aef\u540e\u518d\u8bd5\u3002"
        )

    if available_mb >= QWEN3_GGUF_MIN_REQUEST_RAM_MB:
        return

    page_reads_too_high = (
        page_reads is not None
        and QWEN3_GGUF_MAX_PAGE_READS_PER_SEC > 0
        and int(page_reads) > QWEN3_GGUF_MAX_PAGE_READS_PER_SEC
    )
    page_writes_too_high = (
        page_writes is not None
        and QWEN3_GGUF_MAX_PAGE_WRITES_PER_SEC > 0
        and int(page_writes) > QWEN3_GGUF_MAX_PAGE_WRITES_PER_SEC
    )
    if page_reads_too_high or page_writes_too_high:
        raise RuntimeError(
            f"Qwen3 \u5f53\u524d\u53ef\u7528\u5185\u5b58\u53ea\u6709 {available_mb}MB\uff0c"
            f"\u4e14 Windows \u6362\u9875\u8bfb\u5199\u538b\u529b\u8f83\u9ad8"
            f"\uff08PageReads/s={page_reads}, PageWrites/s={page_writes}\uff09\u3002"
            "\u5df2\u963b\u6b62\u672c\u6b21\u8bf7\u6c42\uff1b\u7b49\u51e0\u79d2\u6216\u91cd\u542f Qwen3 \u540e\u7aef\u540e\u518d\u8bd5\u3002"
        )

    # Low Available RAM is common after the first Qwen3 audio request. If the
    # page file is not being hit hard, allow short follow-up turns to continue.
    return

    available_mb = get_available_ram_mb()
    if available_mb is None or available_mb >= QWEN3_GGUF_MIN_REQUEST_RAM_MB:
        return
    raise RuntimeError(
        f"Qwen3 当前可用内存只有 {available_mb}MB，已阻止本次请求以避免 Windows 换页。"
        " 请关闭其他程序或重启 Qwen3 后端后再试。"
    )


def ask_qwen3_gguf(
    prompt: str,
    max_new_tokens: int = QWEN3_GGUF_MAX_TEXT_TOKENS,
    temperature: float = 0.6,
) -> str:
    assert_qwen3_request_memory()
    return repair_text(
        openai_chat_completion(
            [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    )


def parse_audio_json(text: str) -> tuple[str, str]:
    cleaned = repair_text(text).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(cleaned)
        transcript = str(data.get("transcript") or "").strip()
        reply = str(data.get("reply") or "").strip()
        if transcript or reply:
            return transcript or "（Qwen3 未返回独立转写）", reply or cleaned
    except json.JSONDecodeError:
        pass

    transcript_match = re.search(r"转写[:：]\s*(.+)", cleaned)
    reply_match = re.search(r"回答[:：]\s*(.+)", cleaned)
    if transcript_match or reply_match:
        transcript = transcript_match.group(1).strip() if transcript_match else "（Qwen3 未返回独立转写）"
        reply = reply_match.group(1).strip() if reply_match else cleaned
        return transcript, reply

    return "（Qwen3 未返回独立转写）", cleaned


def ask_qwen3_audio(audio_path: Path, instruction: str) -> tuple[str, str]:
    assert_qwen3_request_memory()
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    prompt = (
        f"{instruction}\n\n"
        "请先逐字转写音频中用户真正说的话，再直接回答用户。"
        "只输出 JSON，不要 Markdown，不要解释。格式："
        '{"transcript":"用户原话","reply":"你的回答"}'
    )
    response = openai_chat_completion(
        [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        max_new_tokens=QWEN3_GGUF_MAX_AUDIO_TOKENS,
        temperature=0.2,
    )
    return parse_audio_json(response)


def ask_backend(prompt: str, backend_id: str) -> str:
    backend = normalize_backend(backend_id)
    if backend == "qwen3-gguf":
        return ask_qwen3_gguf(prompt)
    return ask_mnn_model(prompt)


def transcribe_audio(audio_path: str) -> str:
    transcript_prompt = f"<audio>{audio_path}</audio>{TRANSCRIBE_PROMPT}"
    transcript = clean_transcript(
        ask_mnn_model(
            transcript_prompt,
            system_prompt=ASR_SYSTEM_PROMPT,
            max_new_tokens=48,
            temperature=0.0,
        )
    )
    if looks_like_assistant_answer(transcript):
        retry_prompt = f"<audio>{audio_path}</audio>{TRANSCRIBE_RETRY_PROMPT}"
        transcript = clean_transcript(
            ask_mnn_model(
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


@app.get("/models")
def models():
    qwen3_ready, qwen3_error, qwen3_state, qwen3_state_stale = qwen3_runtime_snapshot()
    return jsonify(
        {
            "default_backend": normalize_backend(DEFAULT_BACKEND),
            "backends": list(BACKENDS.values()),
            "qwen3": {
                "base_url": QWEN3_GGUF_BASE_URL,
                "model": QWEN3_GGUF_MODEL,
                "context": QWEN3_GGUF_CONTEXT,
                "gpu_layers": QWEN3_GGUF_GPU_LAYERS,
                "min_free_vram_mb": QWEN3_GGUF_MIN_FREE_VRAM_MB,
                "cache_type_k": QWEN3_GGUF_CACHE_TYPE_K,
                "cache_type_v": QWEN3_GGUF_CACHE_TYPE_V,
                "flash_attn": QWEN3_GGUF_FLASH_ATTN,
                "mlock": QWEN3_GGUF_MLOCK,
                "parallel": QWEN3_GGUF_PARALLEL,
                "batch_size": QWEN3_GGUF_BATCH_SIZE,
                "ubatch_size": QWEN3_GGUF_UBATCH_SIZE,
                "prompt_cache": QWEN3_GGUF_PROMPT_CACHE,
                "prompt_cache_ram_mb": QWEN3_GGUF_PROMPT_CACHE_RAM_MB,
                "ctx_checkpoints": QWEN3_GGUF_CTX_CHECKPOINTS,
                "min_free_ram_mb": QWEN3_GGUF_MIN_FREE_RAM_MB,
                "min_request_ram_mb": QWEN3_GGUF_MIN_REQUEST_RAM_MB,
                "hard_min_request_ram_mb": QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB,
                "max_page_reads_per_sec": QWEN3_GGUF_MAX_PAGE_READS_PER_SEC,
                "max_page_writes_per_sec": QWEN3_GGUF_MAX_PAGE_WRITES_PER_SEC,
                "trim_working_set": QWEN3_GGUF_TRIM_WORKING_SET,
                "trim_below_ram_mb": QWEN3_GGUF_TRIM_BELOW_RAM_MB,
                "available_ram_mb": get_available_ram_mb(),
                "memory_pressure": get_memory_pressure_summary(),
                "max_text_tokens": QWEN3_GGUF_MAX_TEXT_TOKENS,
                "max_audio_tokens": QWEN3_GGUF_MAX_AUDIO_TOKENS,
                "ready": qwen3_ready,
                "error": qwen3_error,
                "runtime": qwen3_state,
                "runtime_stale": qwen3_state_stale,
            },
        }
    )


@app.get("/health")
def health():
    qwen3_ready, qwen3_error, qwen3_state, qwen3_state_stale = qwen3_runtime_snapshot()
    return jsonify(
        {
            "ok": True,
            "default_backend": normalize_backend(DEFAULT_BACKEND),
            "backends": list(BACKENDS.values()),
            "qwen3_base_url": QWEN3_GGUF_BASE_URL,
            "qwen3_model": QWEN3_GGUF_MODEL,
            "qwen3_context": QWEN3_GGUF_CONTEXT,
            "qwen3_gpu_layers": QWEN3_GGUF_GPU_LAYERS,
            "qwen3_min_free_vram_mb": QWEN3_GGUF_MIN_FREE_VRAM_MB,
            "qwen3_cache_type_k": QWEN3_GGUF_CACHE_TYPE_K,
            "qwen3_cache_type_v": QWEN3_GGUF_CACHE_TYPE_V,
            "qwen3_flash_attn": QWEN3_GGUF_FLASH_ATTN,
            "qwen3_mlock": QWEN3_GGUF_MLOCK,
            "qwen3_parallel": QWEN3_GGUF_PARALLEL,
            "qwen3_batch_size": QWEN3_GGUF_BATCH_SIZE,
            "qwen3_ubatch_size": QWEN3_GGUF_UBATCH_SIZE,
            "qwen3_prompt_cache": QWEN3_GGUF_PROMPT_CACHE,
            "qwen3_prompt_cache_ram_mb": QWEN3_GGUF_PROMPT_CACHE_RAM_MB,
            "qwen3_ctx_checkpoints": QWEN3_GGUF_CTX_CHECKPOINTS,
            "qwen3_min_free_ram_mb": QWEN3_GGUF_MIN_FREE_RAM_MB,
            "qwen3_min_request_ram_mb": QWEN3_GGUF_MIN_REQUEST_RAM_MB,
            "qwen3_hard_min_request_ram_mb": QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB,
            "qwen3_max_page_reads_per_sec": QWEN3_GGUF_MAX_PAGE_READS_PER_SEC,
            "qwen3_max_page_writes_per_sec": QWEN3_GGUF_MAX_PAGE_WRITES_PER_SEC,
            "qwen3_trim_working_set": QWEN3_GGUF_TRIM_WORKING_SET,
            "qwen3_trim_below_ram_mb": QWEN3_GGUF_TRIM_BELOW_RAM_MB,
            "available_ram_mb": get_available_ram_mb(),
            "memory_pressure": get_memory_pressure_summary(),
            "qwen3_max_text_tokens": QWEN3_GGUF_MAX_TEXT_TOKENS,
            "qwen3_max_audio_tokens": QWEN3_GGUF_MAX_AUDIO_TOKENS,
            "qwen3_ready": qwen3_ready,
            "qwen3_error": qwen3_error,
            "qwen3_runtime": qwen3_state,
            "qwen3_runtime_stale": qwen3_state_stale,
            "model_config": str(MODEL_CONFIG),
            "data_dir": str(DATA_DIR),
            "vad_assets_dir": str(VAD_ASSETS_DIR),
            "mnn_model_loaded": mnn_model is not None,
            "mnn_load_seconds": mnn_model_loaded_at,
        }
    )


@app.post("/ask-text")
def ask_text():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    backend = normalize_backend(data.get("backend"))
    if not text:
        return jsonify({"error": "empty text"}), 400

    started = time.time()
    try:
        reply = ask_backend(text, backend)
    except Exception as exc:
        return jsonify({"error": str(exc), "backend": backend}), 500
    return jsonify(
        {
            "reply": reply,
            "backend": backend,
            "elapsed": round(time.time() - started, 2),
        }
    )


@app.post("/ask-audio")
def ask_audio():
    upload = request.files.get("audio")
    if upload is None:
        return jsonify({"error": "missing audio file"}), 400

    backend = normalize_backend(request.form.get("backend"))
    instruction = (request.form.get("instruction") or DEFAULT_AUDIO_PROMPT).strip()
    request_id = uuid.uuid4().hex
    source = RECORDINGS_DIR / f"{request_id}{upload_suffix(upload.filename)}"
    wav = RECORDINGS_DIR / f"{request_id}.16k.wav"
    upload.save(source)

    started = time.time()
    try:
        convert_to_wav(source, wav)
        if backend == "qwen3-gguf":
            transcript, reply = ask_qwen3_audio(wav, instruction)
        else:
            audio_path = wav.as_posix()
            transcript = transcribe_audio(audio_path)
            reply_prompt = (
                f"{instruction}\n\n"
                f"\u7528\u6237\u8bed\u97f3\u8f6c\u5199\u5982\u4e0b\uff1a{transcript}\n\n"
                "\u8bf7\u76f4\u63a5\u56de\u7b54\u7528\u6237\uff0c\u4e0d\u8981\u590d\u8ff0\u201c\u7528\u6237\u8bed\u97f3\u8f6c\u5199\u5982\u4e0b\u201d\u3002"
            )
            reply = ask_mnn_model(reply_prompt)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        delete_if_exists(source)
        delete_if_exists(wav)

    return jsonify(
        {
            "transcript": transcript,
            "reply": reply,
            "backend": backend,
            "elapsed": round(time.time() - started, 2),
        }
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    app.run(host="127.0.0.1", port=7860, threaded=True)
