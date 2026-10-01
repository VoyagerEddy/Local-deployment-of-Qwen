"""Exercise TTS process ownership and HTTP responses without loading models."""

import importlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_tts import QwenTTS, TTSError


REAL_POPEN = subprocess.Popen
FAKE_CLI = r'''
import json, sys, time, wave
from pathlib import Path
mode, record_path = sys.argv[1:3]
args = sys.argv[3:]
prompt = Path(args[args.index("-f") + 1]).read_text(encoding="utf-8")
output = Path(args[args.index("--output") + 1])
record = Path(record_path)
history = json.loads(record.read_text(encoding="utf-8")).get("history", []) if record.exists() else []
history.append({"text": prompt, "args": args})
record.write_text(json.dumps({"text": prompt, "args": args, "history": history}, ensure_ascii=False), encoding="utf-8")
if mode == "delay":
    time.sleep(0.35)
if mode == "wait":
    time.sleep(30)
elif mode == "failure" or (mode == "second_failure" and len(history) == 2):
    sys.stderr.buffer.write("模拟 CLI 失败".encode("utf-8"))
    sys.exit(7)
elif mode == "missing":
    pass
elif mode == "invalid":
    output.write_bytes(b"not a wav")
else:
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x00" * 240)
    if mode == "truncated":
        output.write_bytes(output.read_bytes()[:-2])
'''


class TTSTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="qwen-tts-test-")
        self.root = Path(self.directory.name)
        self.cli = self.root / "fake_cli.py"
        self.cli.write_text(FAKE_CLI, encoding="utf-8")
        self.record = self.root / "record.json"
        self.model = self.root / "model.gguf"
        self.mmproj = self.root / "mmproj.gguf"
        self.model.write_bytes(b"GGUFtest-model")
        self.mmproj.write_bytes(b"GGUFtest-mmproj")
        self.speaker = self.root / "QwenModels" / "Qwen3-TTS-12Hz-1.7B-Base-GGUF" / "voices" / "Chelsie.wav"
        self.speaker.parent.mkdir(parents=True)
        with wave.open(str(self.speaker), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x01\x00" * 240)
        self.speaker_hash = hashlib.sha256(self.speaker.read_bytes()).hexdigest()
        speaker_hash_patch = mock.patch.object(QwenTTS, "SPEAKER_SHA256", self.speaker_hash)
        speaker_hash_patch.start()
        self.addCleanup(speaker_hash_patch.stop)
        self.service = QwenTTS(self.root, self.root / "data")
        self.processes = []
        self.started = threading.Event()
        self.settings = {
            "executable": sys.executable,
            "model": self.model,
            "mmproj": self.mmproj,
            "speaker": self.speaker,
            "seed": 42,
            "gpu_layers": 0,
            "threads": 1,
            "max_frames": 64,
            "min_free_ram_mb": 4096,
            "timeout": 5.0,
        }

    def tearDown(self):
        self.service.shutdown()
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        self.directory.cleanup()

    def fake_popen(self, mode):
        def launch(command, **kwargs):
            process = REAL_POPEN(
                [sys.executable, str(self.cli), mode, str(self.record), *command[1:]],
                **kwargs,
            )
            self.processes.append(process)
            self.started.set()
            return process

        return mock.patch("qwen_tts.subprocess.Popen", side_effect=launch)

    def assert_clean(self):
        self.assertFalse(self.service._lock.locked())
        self.assertIsNone(self.service._process)
        self.assertFalse(list(self.service.temp_dir.glob("request-*")))
        self.assertTrue(all(process.poll() is not None for process in self.processes))

    def environment(self):
        return mock.patch.dict(os.environ, {
            "QWEN3_TTS_LLAMA_TTS": sys.executable,
            "QWEN3_TTS_MODEL": str(self.model),
            "QWEN3_TTS_MMPROJ": str(self.mmproj),
            "QWEN3_TTS_SPEAKER_FILE": "",
            "QWEN3_TTS_GPU_LAYERS": "0",
            "QWEN3_TTS_THREADS": "1",
            "QWEN3_TTS_MAX_FRAMES": "64",
            "QWEN3_TTS_MIN_FREE_RAM_MB": "4096",
            "QWEN3_TTS_TIMEOUT_SECONDS": "5",
        })


class ProcessTests(TTSTestCase):
    def test_unicode_success_and_cpu_flags(self):
        text = "你好，世界。保留字面反斜线 \\n。"
        with mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("success"):
            audio = self.service.synthesize(text)
        recorded = json.loads(self.record.read_text(encoding="utf-8"))
        self.assertEqual(recorded["text"], text)
        args = recorded["args"]
        self.assertEqual(args[args.index("--tts-lang") + 1], "zh")
        self.assertEqual(args[args.index("--tts-speaker-file") + 1], str(self.speaker))
        self.assertEqual(args[args.index("--seed") + 1], "42")
        for flag in ("--offline", "--no-escape", "--no-mmproj-offload", "--no-op-offload", "--no-kv-offload"):
            self.assertIn(flag, args)
        with wave.open(io.BytesIO(audio), "rb") as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()), (24000, 1, 2, 240))
        self.assert_clean()

    def test_multiple_segments_preserve_text_and_merge_valid_wav(self):
        text = "这是第一段完整的中文文本，用来检查标点和长句分段。\n第二段也不能丢失任何文字和符号。最后还有一句。"
        with mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("success"):
            audio = self.service.synthesize(text, "zh")
        history = json.loads(self.record.read_text(encoding="utf-8"))["history"]
        self.assertGreater(len(history), 1)
        self.assertEqual(re.sub(r"\s", "", "".join(item["text"] for item in history)), re.sub(r"\s", "", text))
        self.assertTrue(all(0 < len(item["text"]) <= 24 for item in history))
        # Text changes between child processes; voice conditioning must not.
        for item in history:
            args = item["args"]
            self.assertEqual(args[args.index("--tts-speaker-file") + 1], str(self.speaker))
            self.assertEqual(args[args.index("--seed") + 1], "42")
        with wave.open(io.BytesIO(audio), "rb") as wav:
            expected_frames = 240 * len(history) + 2400 * (len(history) - 1)
            self.assertEqual(wav.getnframes(), expected_frames)
            self.assertEqual(wav.getframerate(), 24000)
            self.assertEqual(len(wav.readframes(expected_frames)), expected_frames * 2)
        self.assert_clean()

    def test_one_deadline_covers_all_segments(self):
        # Each child takes about 0.35 s. Three children would individually fit
        # a 0.75 s timeout, but their combined request must exceed that budget.
        settings = dict(self.settings, timeout=0.75)
        before = time.monotonic()
        with mock.patch.object(self.service, "_settings", return_value=settings), self.fake_popen("delay"):
            with self.assertRaises(TTSError) as error:
                self.service.synthesize("甲" * 72, "zh")
        self.assertEqual(error.exception.status_code, 504)
        self.assertGreaterEqual(len(self.processes), 2)
        self.assertLess(time.monotonic() - before, 1.5, "Timeout appears to restart for every segment")
        self.assert_clean()

    def test_invalid_text_and_language_do_not_start_a_process(self):
        invalid = [(None, None), (123, None), (" ", None), ("x" * 2001, None), ("hello", "unsupported"), ("hello", 123)]
        with mock.patch("qwen_tts.subprocess.Popen") as launch:
            for text, language in invalid:
                with self.subTest(text_type=type(text).__name__, language=language):
                    with self.assertRaises(TTSError) as error:
                        self.service.synthesize(text, language)
                    self.assertEqual(error.exception.status_code, 400)
            launch.assert_not_called()

    def test_failure_and_invalid_audio_release_resources(self):
        with mock.patch.object(self.service, "_settings", return_value=self.settings):
            for mode in ("failure", "missing", "invalid", "truncated"):
                with self.subTest(mode=mode), self.fake_popen(mode):
                    with self.assertRaises(TTSError) as error:
                        self.service.synthesize("你好")
                    self.assertEqual(error.exception.status_code, 500)
                    if mode == "failure":
                        self.assertIn("退出码 7", str(error.exception))
                        self.assertIn("模拟 CLI 失败", str(error.exception))
                    self.assert_clean()

    def test_timeout_kills_process_and_allows_next_request(self):
        settings = dict(self.settings, timeout=0.1)
        before = time.monotonic()
        with mock.patch.object(self.service, "_settings", return_value=settings), self.fake_popen("wait"):
            with self.assertRaises(TTSError) as error:
                self.service.synthesize("hello")
        self.assertLess(time.monotonic() - before, 3, "Timed-out CLI was not reclaimed promptly")
        self.assertEqual(error.exception.status_code, 504)
        self.assert_clean()
        with mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("success"):
            self.assertTrue(self.service.synthesize("hello").startswith(b"RIFF"))
        self.assert_clean()

    def test_busy_status_and_shutdown_do_not_wait_for_synthesis(self):
        errors = []

        def run():
            try:
                self.service.synthesize("hello")
            except TTSError as error:
                errors.append(error)

        with mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("wait"):
            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(self.started.wait(timeout=3), "CLI did not start")
            try:
                self.assertTrue(self.service.status()["busy"])
                with self.assertRaises(TTSError) as error:
                    self.service.synthesize("second request")
                self.assertEqual(error.exception.status_code, 429)
                before = time.monotonic()
                self.service.shutdown()
                self.assertLess(time.monotonic() - before, 3)
            finally:
                self.service.shutdown()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertFalse(self.service.status()["ready"])
            with self.assertRaises(TTSError) as error:
                self.service.synthesize("after shutdown")
            self.assertEqual(error.exception.status_code, 503)
        self.assertTrue(errors)
        self.assert_clean()


class SettingsTests(TTSTestCase):
    def test_fixed_reference_and_seed_ignore_legacy_environment_overrides(self):
        # A leftover setting from the previous configurable-voice deployment
        # must not switch the requested Chelsie voice or generation seed.
        with self.environment(), mock.patch.dict(os.environ, {
            "QWEN3_TTS_SPEAKER_FILE": str(self.root / "another-speaker.wav"),
            "QWEN3_TTS_SEED": "123",
        }), self.fake_popen("success"):
            settings = self.service._settings()
            self.assertEqual(settings["speaker"], self.speaker)
            self.assertEqual(settings["seed"], 42)
            self.service.synthesize("你好")
        args = json.loads(self.record.read_text(encoding="utf-8"))["args"]
        self.assertEqual(args[args.index("--tts-speaker-file") + 1], str(self.speaker))
        self.assertEqual(args[args.index("--seed") + 1], "42")
        self.assert_clean()

    def test_missing_fixed_reference_refuses_before_starting_a_process(self):
        self.speaker.unlink()
        with self.environment(), mock.patch("qwen_tts.subprocess.Popen") as launch:
            status = self.service.status()
            self.assertFalse(status["ready"])
            self.assertIn("Chelsie", status["error"])
            with self.assertRaises(TTSError) as error:
                self.service.synthesize("你好")
            self.assertEqual(error.exception.status_code, 503)
            launch.assert_not_called()
        self.assert_clean()

    def test_changed_fixed_reference_refuses_before_starting_a_process(self):
        # A WAV file can exist and be decodable yet belong to a different voice.
        with wave.open(str(self.speaker), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x02\x00" * 240)
        with self.environment(), mock.patch("qwen_tts.subprocess.Popen") as launch:
            status = self.service.status()
            self.assertFalse(status["ready"])
            self.assertIn("Chelsie", status["error"])
            with self.assertRaises(TTSError) as error:
                self.service.synthesize("你好")
            self.assertEqual(error.exception.status_code, 503)
            launch.assert_not_called()
        self.assert_clean()

    def test_status_exposes_fixed_voice_and_reference_mode(self):
        with self.environment():
            status = self.service.status()
        self.assertTrue(status["ready"])
        self.assertEqual(status["voice"], "Chelsie")
        self.assertEqual(status["voice_mode"], "reference-clone")
        self.assertTrue(status["speaker_reference"])

    def test_partial_or_missing_models_are_not_ready(self):
        with self.environment(), mock.patch("qwen_tts.subprocess.Popen") as launch:
            self.assertTrue(self.service.status()["ready"])
            partial = Path(str(self.model) + ".aria2")
            partial.write_text("partial", encoding="utf-8")
            self.assertFalse(self.service.status()["ready"])
            partial.unlink()
            self.model.write_bytes(b"bad!")
            self.assertFalse(self.service.status()["ready"])
            self.model.unlink()
            self.assertFalse(self.service.status()["ready"])
            launch.assert_not_called()

    def test_invalid_timeout_is_configuration_error(self):
        with self.environment():
            for value in ("0", "-1", "nan", "inf", "invalid"):
                with self.subTest(timeout=value), mock.patch.dict(os.environ, {"QWEN3_TTS_TIMEOUT_SECONDS": value}):
                    status = self.service.status()
                    self.assertFalse(status["ready"])
                    self.assertIn("配置无效", status["error"])

    def test_memory_guard_configuration_accepts_zero_and_rejects_negative(self):
        with self.environment():
            with mock.patch.dict(os.environ, {"QWEN3_TTS_MIN_FREE_RAM_MB": "0"}):
                self.assertEqual(self.service._settings()["min_free_ram_mb"], 0)
            with mock.patch.dict(os.environ, {"QWEN3_TTS_MIN_FREE_RAM_MB": "-1"}):
                self.assertFalse(self.service.status()["ready"])


class MemoryGuardTests(TTSTestCase):
    def with_ram(self, available_ram):
        self.service.shutdown()
        self.service = QwenTTS(self.root, self.root / "data", available_ram=lambda: available_ram)

    def test_low_ram_refuses_before_starting_a_process(self):
        self.with_ram(4095)
        with mock.patch.object(self.service, "_settings", return_value=self.settings), mock.patch("qwen_tts.subprocess.Popen") as launch:
            with self.assertRaises(TTSError) as error:
                self.service.synthesize("你好")
            self.assertEqual(error.exception.status_code, 503)
            self.assertIn("内存不足", str(error.exception))
            launch.assert_not_called()
        self.assert_clean()

    def test_ram_at_threshold_can_synthesize(self):
        self.with_ram(4096)
        with mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("success"):
            self.assertTrue(self.service.synthesize("hello").startswith(b"RIFF"))
        self.assert_clean()

    def test_zero_threshold_disables_memory_guard(self):
        self.with_ram(0)
        settings = dict(self.settings, min_free_ram_mb=0)
        with mock.patch.object(self.service, "_settings", return_value=settings), self.fake_popen("success"):
            self.assertTrue(self.service.synthesize("hello").startswith(b"RIFF"))
        self.assert_clean()

    def test_unknown_ram_does_not_claim_low_memory(self):
        self.with_ram(None)
        with mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("success"):
            self.assertTrue(self.service.synthesize("hello").startswith(b"RIFF"))
        self.assert_clean()

    def test_ram_is_checked_again_before_the_next_segment(self):
        self.service.shutdown()
        read_ram = mock.Mock(side_effect=[4096, 4096, 4095])
        self.service = QwenTTS(self.root, self.root / "data", available_ram=read_ram)
        with mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("success"):
            with self.assertRaises(TTSError) as error:
                self.service.synthesize("甲" * 48, "zh")
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(len(self.processes), 1, "A low-RAM second segment must not be launched")
        self.assert_clean()

    def test_prepare_memory_then_recheck_allows_synthesis(self):
        self.service.shutdown()
        read_ram = mock.Mock(side_effect=[900, 4096, 4096])
        prepare = mock.Mock(return_value=True)
        self.service = QwenTTS(self.root, self.root / "data", available_ram=read_ram, prepare_memory=prepare)
        with mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("success"):
            self.assertTrue(self.service.synthesize("hello").startswith(b"RIFF"))
        prepare.assert_called_once_with(4096)
        self.assertEqual(read_ram.call_count, 3)
        self.assert_clean()

    def test_prepare_memory_still_rejects_if_actual_ram_stays_low(self):
        # A successful trim call alone does not prove that RAM was recovered.
        for claimed_success in (False, True):
            with self.subTest(claimed_success=claimed_success):
                self.service.shutdown()
                read_ram = mock.Mock(side_effect=[900, 900])
                prepare = mock.Mock(return_value=claimed_success)
                self.service = QwenTTS(self.root, self.root / "data", available_ram=read_ram, prepare_memory=prepare)
                with mock.patch.object(self.service, "_settings", return_value=self.settings), mock.patch("qwen_tts.subprocess.Popen") as launch:
                    with self.assertRaises(TTSError) as error:
                        self.service.synthesize("hello")
                    self.assertEqual(error.exception.status_code, 503)
                    launch.assert_not_called()
                prepare.assert_called_once_with(4096)
                self.assertEqual(read_ram.call_count, 2)
                self.assert_clean()

    def test_sufficient_ram_does_not_prepare_memory(self):
        self.service.shutdown()
        prepare = mock.Mock()
        self.service = QwenTTS(self.root, self.root / "data", available_ram=lambda: 4096, prepare_memory=prepare)
        with mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("success"):
            self.assertTrue(self.service.synthesize("hello").startswith(b"RIFF"))
        prepare.assert_not_called()
        self.assert_clean()

    def test_memory_preparation_uses_the_shared_deadline(self):
        self.service.shutdown()
        now = [0.0]
        read_ram = mock.Mock(side_effect=[4096, 900, 4096])

        def prepare(_minimum):
            now[0] += 6.0

        self.service = QwenTTS(self.root, self.root / "data", available_ram=read_ram, prepare_memory=prepare)
        with mock.patch.object(self.service, "_settings", return_value=self.settings), mock.patch("qwen_tts.time.monotonic", side_effect=lambda: now[0]), mock.patch.object(self.service, "_run_segment", return_value=((1, 2, 24000), b"\x01\x00")) as run_segment:
            with self.assertRaises(TTSError) as error:
                self.service.synthesize("hello")
            self.assertEqual(error.exception.status_code, 504)
            run_segment.assert_not_called()
        self.assert_clean()


class HTTPTests(TTSTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_data = tempfile.TemporaryDirectory(prefix="qwen-tts-app-test-")
        with mock.patch.dict(os.environ, {"QWEN_VOICE_DATA_DIR": cls.app_data.name}):
            cls.web = importlib.import_module("app")

    @classmethod
    def tearDownClass(cls):
        cls.web.tts_service.shutdown()
        cls.app_data.cleanup()

    def test_binary_wav_response_uses_real_process(self):
        with mock.patch.object(self.web, "tts_service", self.service), mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("success"):
            response = self.web.app.test_client().post("/tts", json={"text": "你好"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "audio/wav")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertTrue(response.data.startswith(b"RIFF"))
        self.assert_clean()

    def test_second_segment_failure_returns_json_without_partial_audio(self):
        with mock.patch.object(self.web, "tts_service", self.service), mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("second_failure"):
            response = self.web.app.test_client().post("/tts", json={"text": "甲" * 72, "language": "zh"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.mimetype, "application/json")
        self.assertIn("退出码 7", response.json["error"])
        self.assertFalse(response.data.startswith(b"RIFF"))
        self.assertEqual(len(self.processes), 2, "Generation should stop at the failed segment")
        self.assert_clean()

    def test_invalid_json_input_and_oversized_body_are_rejected(self):
        client = self.web.app.test_client()
        with mock.patch.object(self.web, "tts_service", self.service), mock.patch("qwen_tts.subprocess.Popen") as launch:
            for payload in ([], "hello", {}, {"text": 123}, {"text": ""}, {"text": "x" * 2001}):
                with self.subTest(payload_type=type(payload).__name__):
                    self.assertEqual(client.post("/tts", json=payload).status_code, 400)
            response = client.post("/tts", data=b"x" * 32769, content_type="application/json")
            self.assertEqual(response.status_code, 413)
            self.assertIsNotNone(response.json)
            launch.assert_not_called()

    def test_status_and_not_ready_error_remain_json(self):
        with mock.patch.object(self.web, "tts_service", self.service), mock.patch.object(self.service, "_settings", side_effect=TTSError("模型未就绪", 503)):
            client = self.web.app.test_client()
            status = client.get("/tts/status")
            self.assertEqual(status.status_code, 200)
            self.assertFalse(status.json["ready"])
            response = client.post("/tts", json={"text": "hello"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json, {"engine": "qwen3-tts", "error": "模型未就绪"})

    def test_status_response_exposes_fixed_chelsie_voice(self):
        with mock.patch.object(self.web, "tts_service", self.service), self.environment():
            response = self.web.app.test_client().get("/tts/status")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ready"])
        self.assertEqual(response.json["voice"], "Chelsie")
        self.assertEqual(response.json["voice_mode"], "reference-clone")
        self.assertTrue(response.json["speaker_reference"])

    def test_unknown_content_length_still_enforces_body_limit(self):
        # wsgi.input_terminated tells Werkzeug the stream has a known end even
        # though no Content-Length is provided, as with chunked HTTP input.
        body = json.dumps({"text": "hello", "padding": "x" * 40000}).encode("utf-8")
        with mock.patch.object(self.web, "tts_service", self.service), mock.patch("qwen_tts.subprocess.Popen") as launch:
            response = self.web.app.test_client().post(
                "/tts", data=body, content_type="application/json",
                environ_overrides={"CONTENT_LENGTH": "", "wsgi.input_terminated": True},
            )
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json["engine"], "qwen3-tts")
            launch.assert_not_called()

    def test_unknown_content_length_accepts_body_at_limit(self):
        prefix, suffix = b'{"text":"hello","padding":"', b'"}'
        body = prefix + b"x" * (32768 - len(prefix) - len(suffix)) + suffix
        self.assertEqual(len(body), 32768)
        with mock.patch.object(self.web, "tts_service", self.service), mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("success"):
            response = self.web.app.test_client().post(
                "/tts", data=body, content_type="application/json",
                environ_overrides={"CONTENT_LENGTH": "", "wsgi.input_terminated": True},
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.startswith(b"RIFF"))
        self.assert_clean()

    def test_busy_tts_rejects_chat_but_status_and_health_remain_available(self):
        replies, errors = [], []

        def run_tts():
            try:
                replies.append(self.web.app.test_client().post("/tts", json={"text": "hello"}))
            except Exception as error:
                errors.append(error)

        with mock.patch.object(self.web, "tts_service", self.service), mock.patch.object(self.service, "_settings", return_value=self.settings), self.fake_popen("wait"), mock.patch.object(self.web, "ask_backend", return_value="可以继续") as ask, mock.patch.object(self.web, "qwen3_runtime_snapshot", return_value=(True, None, None, False)), mock.patch.object(self.web, "get_available_ram_mb", return_value=8192), mock.patch.object(self.web, "get_memory_pressure_summary", return_value={"available_mb": 8192}):
            thread = threading.Thread(target=run_tts)
            thread.start()
            try:
                self.assertTrue(self.started.wait(timeout=3), "TTS subprocess did not start")
                client = self.web.app.test_client()
                before = time.monotonic()
                blocked = client.post("/ask-text", json={"text": "你好"})
                self.assertEqual(blocked.status_code, 429)
                self.assertLess(time.monotonic() - before, 1, "Busy chat waited for the TTS request")
                self.assertEqual(client.post("/ask-audio").status_code, 429)
                ask.assert_not_called()
                status = client.get("/tts/status")
                self.assertEqual(status.status_code, 200)
                self.assertTrue(status.json["busy"])
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertTrue(health.json["ok"])
            finally:
                self.service.shutdown()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "TTS worker did not exit after shutdown")
            self.assertFalse(errors)
            self.assertEqual(len(replies), 1)
            self.assertEqual(replies[0].status_code, 500)
            self.assertFalse(self.web.model_request_lock.locked())
            resumed = client.post("/ask-text", json={"text": "你好"})
            self.assertEqual(resumed.status_code, 200)
            self.assertEqual(resumed.json["reply"], "可以继续")
            ask.assert_called_once()
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
