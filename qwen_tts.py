"""On-demand local Qwen3-TTS inference through llama-tts."""

import atexit
import hashlib
import io
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Callable


class TTSError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class QwenTTS:
    MAX_TEXT_LENGTH = 2000
    LANGUAGES = {"zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"}
    VOICE_NAME = "Chelsie"
    SPEAKER_SHA256 = "e2461d0e0fc2bf1e083ea7af40c3b6849e5912b2744162cab6e151c366097b53"

    def __init__(
        self, app_dir: Path, data_dir: Path,
        available_ram: Callable[[], int | None] | None = None,
        prepare_memory: Callable[[int], object] | None = None,
    ):
        self.app_dir = app_dir
        self.temp_dir = data_dir / "tts"
        self._lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._process = None
        self._stopped = False
        self._available_ram = available_ram
        self._prepare_memory = prepare_memory
        atexit.register(self.shutdown)

    def _path(self, name: str, default: Path) -> Path:
        path = Path(os.environ.get(name) or default).expanduser()
        return path if path.is_absolute() else self.app_dir / path

    def _executable(self) -> str:
        configured = os.environ.get("QWEN3_TTS_LLAMA_TTS", "").strip()
        if configured:
            found = shutil.which(configured)
            if found:
                return found
            raise TTSError("QWEN3_TTS_LLAMA_TTS 指定的 llama-tts 程序不存在。", 503)
        found = shutil.which("llama-tts")
        if found:
            return found
        packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        for path in sorted(packages.glob("ggml.llamacpp_*/llama-tts.exe"), reverse=True):
            if path.is_file():
                return str(path)
        raise TTSError("未找到 llama-tts。请安装包含 llama-tts 的 llama.cpp，或设置 QWEN3_TTS_LLAMA_TTS。", 503)

    def _settings(self) -> dict:
        model_dir = self._path(
            "QWEN3_TTS_MODEL_DIR", self.app_dir / "QwenModels" / "Qwen3-TTS-12Hz-1.7B-Base-GGUF"
        )
        settings = {
            "executable": self._executable(),
            "model": self._path("QWEN3_TTS_MODEL", model_dir / "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"),
            "mmproj": self._path("QWEN3_TTS_MMPROJ", model_dir / "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"),
            # Every segment must use the same verified Chelsie reference. The Base
            # model has no named speaker selector; omitting this changes its voice.
            "speaker": self.app_dir / "QwenModels" / "Qwen3-TTS-12Hz-1.7B-Base-GGUF" / "voices" / "Chelsie.wav",
            "seed": 42,
        }
        try:
            settings["gpu_layers"] = int(os.environ.get("QWEN3_TTS_GPU_LAYERS", "0"))
            settings["threads"] = int(os.environ.get("QWEN3_TTS_THREADS", "4"))
            settings["max_frames"] = int(os.environ.get("QWEN3_TTS_MAX_FRAMES", "120"))
            settings["min_free_ram_mb"] = int(os.environ.get("QWEN3_TTS_MIN_FREE_RAM_MB", "4096"))
            settings["timeout"] = float(os.environ.get("QWEN3_TTS_TIMEOUT_SECONDS", "600"))
            if (
                settings["gpu_layers"] < 0 or settings["threads"] < 1 or settings["max_frames"] < 1
                or settings["min_free_ram_mb"] < 0
                or settings["timeout"] <= 0 or not math.isfinite(settings["timeout"])
            ):
                raise ValueError
        except ValueError as exc:
            raise TTSError("Qwen3-TTS 的 GPU 层数、线程数、帧数或超时配置无效。", 503) from exc
        for key in ("model", "mmproj"):
            path = settings[key]
            try:
                with path.open("rb") as handle:
                    valid = handle.read(4) == b"GGUF"
            except OSError:
                valid = False
            if not valid or Path(str(path) + ".aria2").exists():
                raise TTSError(f"Qwen3-TTS 模型未就绪：{path.name}。请先运行 setup_qwen3_tts.ps1。", 503)
        try:
            speaker_hash = hashlib.sha256(settings["speaker"].read_bytes()).hexdigest()
        except OSError as exc:
            raise TTSError("Chelsie 参考音频缺失或无法读取，请运行 setup_qwen3_tts.ps1。", 503) from exc
        if speaker_hash != self.SPEAKER_SHA256:
            raise TTSError("Chelsie 参考音频校验失败，请运行 setup_qwen3_tts.ps1 检查官方样音。", 503)
        return settings

    def status(self) -> dict:
        result = {
            "engine": "qwen3-tts", "ready": False, "busy": self._lock.locked(),
            "error": None, "model": "Qwen3-TTS-12Hz-1.7B-Base", "max_text_length": self.MAX_TEXT_LENGTH,
            "voice": self.VOICE_NAME, "voice_mode": "reference-clone",
        }
        try:
            if self._stopped:
                raise TTSError("Qwen3-TTS 服务正在关闭。", 503)
            settings = self._settings()
            result.update(ready=True, gpu_layers=settings["gpu_layers"], speaker_reference=True)
        except TTSError as exc:
            result["error"] = str(exc)
        return result

    @staticmethod
    def _terminate(process) -> None:
        if process is not None and process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass

    def shutdown(self) -> None:
        with self._process_lock:
            self._stopped = True
            self._terminate(self._process)

    @staticmethod
    def _chunks(text: str, language: str) -> list[str]:
        # Short segments bound CPU inference time and memory use.
        limit = 24 if language in {"zh", "ja", "ko"} else 80
        chunks = []
        while text:
            end = min(len(text), limit)
            if len(text) > limit:
                breaks = [match.end() for match in re.finditer(r"[。！？；.!?;，,\n\s]", text[:limit])]
                if breaks and breaks[-1] >= limit // 3:
                    end = breaks[-1]
            chunk, text = text[:end].strip(), text[end:].lstrip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _check_memory(self, settings: dict) -> None:
        available_ram = self._available_ram() if self._available_ram is not None else None
        if available_ram is not None and available_ram < settings["min_free_ram_mb"] and self._prepare_memory is not None:
            self._prepare_memory(settings["min_free_ram_mb"])
            available_ram = self._available_ram()
        if available_ram is not None and available_ram < settings["min_free_ram_mb"]:
            raise TTSError("可用内存不足，暂不能启动 Qwen3-TTS。请关闭占用内存的程序后重试，或选择 Web TTS。", 503)

    def _run_segment(self, text: str, language: str, settings: dict, work_dir: Path, timeout: float) -> tuple:
        prompt = work_dir / "input.txt"
        output = work_dir / "output.wav"
        log = work_dir / "inference.log"
        prompt.write_text(text, encoding="utf-8")
        command = [
            settings["executable"], "--offline", "-m", str(settings["model"]),
            "-mm", str(settings["mmproj"]), "-f", str(prompt), "--no-escape",
            "--tts-lang", language, "--output", str(output),
            "--tts-speaker-file", str(settings["speaker"]), "--seed", str(settings["seed"]),
            "-ngl", str(settings["gpu_layers"]), "-t", str(settings["threads"]),
            "-c", "512", "-b", "256", "-ub", "64", "-n", str(settings["max_frames"]),
        ]
        if settings["gpu_layers"] == 0:
            command.extend(["--no-mmproj-offload", "--no-op-offload", "--no-kv-offload"])
        process = None
        try:
            with log.open("wb") as log_file:
                with self._process_lock:
                    if self._stopped:
                        raise TTSError("Qwen3-TTS 服务正在关闭。", 503)
                    process = subprocess.Popen(
                        command, cwd=self.app_dir, stdout=log_file, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    self._process = process
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    self._terminate(process)
                    raise TTSError("Qwen3-TTS 合成超时，已释放推理进程。可缩短回答或调整超时设置。", 504) from exc
            if process.returncode != 0:
                with log.open("rb") as handle:
                    handle.seek(max(0, log.stat().st_size - 3000))
                    detail = handle.read().decode("utf-8", errors="replace").strip()
                raise TTSError(f"Qwen3-TTS 合成失败（退出码 {process.returncode}）：{detail}")
            if not output.is_file() or output.stat().st_size > 64 * 1024 * 1024:
                raise TTSError("Qwen3-TTS 未生成有效的 WAV 文件。")
            try:
                with wave.open(str(output), "rb") as wav:
                    if wav.getnframes() <= 0 or wav.getframerate() <= 0:
                        raise wave.Error("empty audio")
                    params = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
                    frames = wav.readframes(wav.getnframes())
                    if len(frames) != wav.getnframes() * params[0] * params[1]:
                        raise wave.Error("truncated audio")
            except (wave.Error, EOFError) as exc:
                raise TTSError("Qwen3-TTS 输出不是有效的 PCM WAV 音频。") from exc
            return params, frames
        finally:
            self._terminate(process)
            with self._process_lock:
                self._process = None

    def synthesize(self, text: object, language: object = None) -> bytes:
        if not isinstance(text, str) or not text.strip():
            raise TTSError("TTS text 必须是非空字符串。", 400)
        text = text.strip()
        if len(text) > self.MAX_TEXT_LENGTH:
            raise TTSError(f"TTS 文本不能超过 {self.MAX_TEXT_LENGTH} 个字符。", 400)
        if language is None:
            language = "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"
        if not isinstance(language, str) or language not in self.LANGUAGES:
            raise TTSError("不支持此 TTS 语言，请使用 zh、en 等语言代码。", 400)
        settings = self._settings()
        deadline = time.monotonic() + settings["timeout"]
        if not self._lock.acquire(blocking=False):
            raise TTSError("Qwen3-TTS 正在合成其他语音，请稍后重试。", 429)
        try:
            self._check_memory(settings)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="request-", dir=self.temp_dir) as directory:
                work_dir = Path(directory)
                params = None
                segments = []
                for index, chunk in enumerate(self._chunks(text, language)):
                    self._check_memory(settings)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TTSError("Qwen3-TTS 合成超时，请缩短回答或调整超时设置。", 504)
                    segment_dir = work_dir / str(index)
                    segment_dir.mkdir()
                    segment_params, frames = self._run_segment(chunk, language, settings, segment_dir, remaining)
                    if params is None:
                        params = segment_params
                    elif params != segment_params:
                        raise TTSError("Qwen3-TTS 音频分段的格式不一致。")
                    segments.append(frames)
                result = io.BytesIO()
                with wave.open(result, "wb") as merged:
                    merged.setnchannels(params[0])
                    merged.setsampwidth(params[1])
                    merged.setframerate(params[2])
                    for index, frames in enumerate(segments):
                        if index:
                            merged.writeframes(b"\x00" * (params[2] // 10) * params[0] * params[1])
                        merged.writeframes(frames)
                return result.getvalue()
        except OSError as exc:
            raise TTSError(f"无法运行 Qwen3-TTS：{exc}", 503) from exc
        finally:
            self._lock.release()
