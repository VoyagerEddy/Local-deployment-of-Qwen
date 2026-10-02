"""Regression checks for transcript handling, request guards and API contracts."""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class AppCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="qwen-cleanup-test-")
        with mock.patch.dict(os.environ, {"QWEN_VOICE_DATA_DIR": cls.directory.name}):
            cls.web = importlib.import_module("app")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_transcript_preserves_spoken_instructions_and_technical_terms(self):
        for text in (
            "这件事不要解释，直接给我结果。",
            "请解释 speech-to-text 的原理。",
            "今天学习 ASR only 这个短语。",
            "Do not answer this question yet.",
            f"这句话是什么意思：{self.web.TRANSCRIBE_PROMPT}",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.web.clean_transcript(text), text)

    def test_transcript_discards_only_a_separate_exact_prompt_suffix(self):
        for prompt in (self.web.ASR_SYSTEM_PROMPT, self.web.TRANSCRIBE_PROMPT, self.web.TRANSCRIBE_RETRY_PROMPT):
            with self.subTest(prompt=prompt):
                self.assertEqual(self.web.clean_transcript(f"明天见。\n\n{prompt}"), "明天见。")
        self.assertEqual(self.web.clean_transcript("转写：明天见。"), "明天见。")

    def test_memory_guard_skips_performance_probe_when_ram_is_sufficient_or_unknown(self):
        with mock.patch.object(self.web, "QWEN3_GGUF_MIN_REQUEST_RAM_MB", 1024), mock.patch.object(self.web, "QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB", 512):
            for available in (8192, 1024, None):
                with self.subTest(available=available), mock.patch.object(self.web, "get_available_ram_mb", return_value=available), mock.patch.object(self.web, "get_memory_pressure_summary") as probe:
                    self.web.assert_qwen3_request_memory()
                    probe.assert_not_called()

    def test_hard_memory_limit_blocks_before_performance_probe(self):
        with mock.patch.object(self.web, "QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB", 512), mock.patch.object(self.web, "get_available_ram_mb", return_value=511), mock.patch.object(self.web, "get_memory_pressure_summary") as probe:
            with self.assertRaises(RuntimeError):
                self.web.assert_qwen3_request_memory()
            probe.assert_not_called()

    def test_low_ram_still_checks_paging_and_blocks_high_pressure(self):
        with mock.patch.object(self.web, "QWEN3_GGUF_MIN_REQUEST_RAM_MB", 1024), mock.patch.object(self.web, "QWEN3_GGUF_HARD_MIN_REQUEST_RAM_MB", 512), mock.patch.object(self.web, "QWEN3_GGUF_MAX_PAGE_READS_PER_SEC", 120), mock.patch.object(self.web, "QWEN3_GGUF_MAX_PAGE_WRITES_PER_SEC", 80), mock.patch.object(self.web, "get_available_ram_mb", return_value=768):
            for reads, writes, blocked in ((0, 0, False), (None, None, False), (121, 0, True), (0, 81, True)):
                with self.subTest(reads=reads, writes=writes), mock.patch.object(self.web, "get_memory_pressure_summary", return_value={"page_reads_per_sec": reads, "page_writes_per_sec": writes}):
                    if blocked:
                        with self.assertRaises(RuntimeError):
                            self.web.assert_qwen3_request_memory()
                    else:
                        self.web.assert_qwen3_request_memory()

    def test_mnn_configuration_errors_reach_the_client_without_inference(self):
        qwen = mock.Mock()
        client = self.web.app.test_client()
        with mock.patch.object(self.web, "load_mnn_model", return_value=qwen):
            for failure in (False, RuntimeError("configuration rejected")):
                with self.subTest(failure=failure):
                    qwen.set_config.return_value = failure
                    qwen.set_config.side_effect = failure if isinstance(failure, Exception) else None
                    response = client.post("/ask-text", json={"text": "你好", "backend": "qwen25-mnn"})
                    self.assertEqual(response.status_code, 500)
                    self.assertTrue(response.json["error"])
                    qwen.response.assert_not_called()
                    self.assertFalse(self.web.model_request_lock.locked())

    def test_status_endpoints_preserve_existing_response_fields(self):
        memory = {"available_mb": 8192, "pages_per_sec": None, "page_reads_per_sec": None, "page_writes_per_sec": None}
        with mock.patch.object(self.web, "qwen3_runtime_snapshot", return_value=(True, None, {"gpu_layers": 13}, False)), mock.patch.object(self.web, "get_memory_pressure_summary", return_value=memory):
            client = self.web.app.test_client()
            models = client.get("/models")
            health = client.get("/health")
        self.assertEqual(models.status_code, 200)
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json["ok"])
        self.assertIn("model_config", health.json)
        for name, value in models.json["qwen3"].items():
            health_name = name if name in {"available_ram_mb", "memory_pressure"} else f"qwen3_{name}"
            self.assertIn(health_name, health.json)
            self.assertEqual(health.json[health_name], value)
        self.assertEqual(health.json["available_ram_mb"], health.json["memory_pressure"]["available_mb"])


if __name__ == "__main__":
    unittest.main()
