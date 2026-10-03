"""Generation throughput contracts without loading a local model or server."""

import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation_stats import GenerationStats, GenerationStatsRegistry


class CompletionResponse(io.BytesIO):
    def __init__(self, events=None, data=None):
        if events is not None:
            frames = [
                "data: " + (event if isinstance(event, str) else json.dumps(event, ensure_ascii=False)) + "\n\n"
                for event in events
            ]
            body = ": keepalive\n\n" + "".join(frames)
            self.headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        else:
            body = json.dumps(data, ensure_ascii=False)
            self.headers = {"Content-Type": "application/json"}
        super().__init__(body.encode("utf-8"))


def delta(content=None, reasoning=None, finish_reason=None, **fields):
    values = {}
    if content is not None:
        values["content"] = content
    if reasoning is not None:
        values["reasoning_content"] = reasoning
    return {"choices": [{"delta": values, "finish_reason": finish_reason}], **fields}


class GenerationStatsTests(unittest.TestCase):
    def test_native_decode_timings_exclude_prompt_processing(self):
        stats = GenerationStats()
        stats.record({"timings": {
            "prompt_n": 1000, "prompt_ms": 9000,
            "predicted_n": 40, "predicted_ms": 2000,
        }}, has_token=True)
        stats.finish()
        self.assertEqual(stats.snapshot(), {
            "status": "complete", "generated_tokens": 40,
            "tokens_per_second": 20.0, "generation_seconds": 2.0,
            "estimated": False,
        })

    def test_server_rate_has_priority_over_local_chunk_timing(self):
        stats = GenerationStats()
        with mock.patch("generation_stats.time.monotonic", return_value=100):
            stats.record({"timings": {
                "predicted_n": 50, "predicted_ms": 2000,
                "predicted_per_second": 24.75,
            }}, has_token=True)
        with mock.patch("generation_stats.time.monotonic", return_value=200):
            stats.record({}, has_token=True)
        self.assertEqual(stats.snapshot()["generated_tokens"], 50)
        self.assertEqual(stats.snapshot()["tokens_per_second"], 24.75)
        self.assertFalse(stats.snapshot()["estimated"])

    def test_chunk_estimate_starts_after_first_output_and_usage_corrects_count(self):
        stats = GenerationStats()
        with mock.patch("generation_stats.time.monotonic", return_value=100):
            stats.record({}, has_token=True)
        self.assertIsNone(stats.snapshot()["tokens_per_second"])
        with mock.patch("generation_stats.time.monotonic", return_value=102):
            stats.record({}, has_token=True)
        self.assertEqual(stats.snapshot()["generated_tokens"], 2)
        self.assertEqual(stats.snapshot()["tokens_per_second"], 0.5)
        self.assertTrue(stats.snapshot()["estimated"])
        with mock.patch("generation_stats.time.monotonic", return_value=102):
            stats.record({"usage": {"prompt_tokens": 500, "completion_tokens": 12, "total_tokens": 512}})
        stats.finish()
        self.assertEqual(stats.snapshot()["generated_tokens"], 12)
        self.assertTrue(stats.snapshot()["estimated"])

    def test_begin_discards_previous_audio_transcription_phase(self):
        stats = GenerationStats()
        stats.record({"timings": {"predicted_n": 48, "predicted_ms": 2000}})
        stats.begin()
        self.assertEqual(stats.snapshot(), {
            "status": "waiting", "generated_tokens": None,
            "tokens_per_second": None, "generation_seconds": None,
            "estimated": False,
        })
        stats.record({"timings": {"predicted_n": 12, "predicted_ms": 1000}})
        self.assertEqual(stats.snapshot()["generated_tokens"], 12)

    def test_invalid_numeric_stats_do_not_display_invalid_speed(self):
        for value in (True, -1, float("nan"), float("inf"), "20"):
            with self.subTest(value=value):
                stats = GenerationStats()
                stats.record({"timings": {
                    "predicted_n": value, "predicted_ms": value,
                    "predicted_per_second": value,
                }})
                self.assertIsNone(stats.snapshot()["generated_tokens"])
                self.assertIsNone(stats.snapshot()["tokens_per_second"])

    def test_registry_keeps_active_requests_and_expires_completed_entries(self):
        registry = GenerationStatsRegistry()
        with mock.patch("generation_stats.time.monotonic", return_value=100):
            active = registry.create("active-request")
            completed = registry.create("complete-request")
            completed.finish()
        with self.assertRaises(FileExistsError):
            registry.create("active-request")
        with mock.patch("generation_stats.time.monotonic", return_value=100 + registry.TTL_SECONDS + 1):
            self.assertIsNone(registry.snapshot("complete-request"))
            self.assertEqual(registry.snapshot("active-request"), active.snapshot())
        self.assertIsNone(registry.snapshot("missing-request"))

    def test_registry_bounds_storage_without_evicting_active_generation(self):
        registry = GenerationStatsRegistry()
        registry.MAX_ENTRIES = 2
        registry.create("active-first")
        completed = registry.create("done-second")
        with self.assertRaises(RuntimeError):
            registry.create("new-third")
        completed.finish()
        registry.create("new-third")
        self.assertIsNotNone(registry.snapshot("active-first"))
        self.assertIsNone(registry.snapshot("done-second"))


class GenerationAPIRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="qwen-generation-test-")
        with mock.patch.dict(os.environ, {"QWEN_VOICE_DATA_DIR": cls.directory.name}):
            cls.web = importlib.import_module("app")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.registry = GenerationStatsRegistry()
        self.registry_patch = mock.patch.object(self.web, "generation_registry", self.registry)
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)
        self.client = self.web.app.test_client()

    def backend_patches(self, response):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(self.web, "qwen3_backend_status", return_value=(True, None)))
        stack.enter_context(mock.patch.object(self.web, "assert_qwen3_request_memory"))
        stack.enter_context(mock.patch.object(self.web, "trim_qwen3_working_set"))
        stack.enter_context(mock.patch.object(self.web.urllib.request, "urlopen", return_value=response))
        return stack

    def test_stream_assembles_reply_and_requests_usage_and_per_token_timings(self):
        upstream = CompletionResponse(events=[
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            delta(reasoning="隐藏的推理过程"),
            delta(content="你好，"),
            delta(content="世界。", finish_reason="stop", timings={
                "predicted_n": 10, "predicted_ms": 500,
                "prompt_n": 100, "prompt_ms": 8000,
            }),
            {"choices": [], "usage": {"completion_tokens": 10, "prompt_tokens": 100}},
            "[DONE]",
        ])
        with self.backend_patches(upstream):
            response = self.client.post("/ask-text", json={
                "text": "打个招呼", "backend": "qwen3-gguf", "generation_id": "stream-exact",
            })
            sent_request = self.web.urllib.request.urlopen.call_args.args[0]
            payload = json.loads(sent_request.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["reply"], "你好，世界。")
        self.assertEqual(response.json["generation_id"], "stream-exact")
        self.assertEqual(response.json["generation"]["generated_tokens"], 10)
        self.assertEqual(response.json["generation"]["tokens_per_second"], 20)
        self.assertFalse(response.json["generation"]["estimated"])
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertTrue(payload["timings_per_token"])
        self.assertFalse(self.web.model_request_lock.locked())

    def test_multi_character_chunks_and_reasoning_are_only_estimates(self):
        upstream = CompletionResponse(events=[
            {"choices": [{"delta": {"role": "assistant"}}]},
            delta(reasoning="推理包含很多字符"),
            delta(content="这是一整句文本。"),
            "[DONE]",
        ])
        with self.backend_patches(upstream):
            response = self.client.post("/ask-text", json={
                "text": "你好", "backend": "qwen3-gguf", "generation_id": "chunk-estimate",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["reply"], "这是一整句文本。")
        self.assertEqual(response.json["generation"]["generated_tokens"], 2)
        self.assertTrue(response.json["generation"]["estimated"])

    def test_finish_reason_completes_stream_without_done_sentinel(self):
        response = CompletionResponse(events=[delta(content="完成", finish_reason="length")])
        self.assertEqual(self.web.read_chat_completion(response), "完成")

    def test_interrupted_or_error_stream_discards_partial_speed_and_releases_lock(self):
        for events in (
            [delta(content="不完整的回答", timings={"predicted_n": 4, "predicted_ms": 1000})],
            [delta(content="不完整"), {"error": {"message": "model stopped"}}],
            ["{malformed"],
        ):
            with self.subTest(events=events):
                self.registry = GenerationStatsRegistry()
                self.web.generation_registry = self.registry
                with self.backend_patches(CompletionResponse(events=events)):
                    response = self.client.post("/ask-text", json={
                        "text": "你好", "backend": "qwen3-gguf", "generation_id": "failed-stream",
                    })
                self.assertEqual(response.status_code, 500)
                state = self.client.get("/generation-stats/failed-stream").json
                self.assertEqual(state["status"], "error")
                self.assertIsNone(state["generated_tokens"])
                self.assertIsNone(state["tokens_per_second"])
                self.assertFalse(self.web.model_request_lock.locked())

    def test_non_stream_json_response_remains_compatible(self):
        upstream = CompletionResponse(data={
            "choices": [{"message": {"content": "  兼容回答  "}}],
            "usage": {"completion_tokens": 8, "prompt_tokens": 200},
            "timings": {"predicted_n": 8, "predicted_ms": 400},
        })
        with self.backend_patches(upstream):
            response = self.client.post("/ask-text", json={"text": "你好", "backend": "qwen3-gguf"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["reply"], "兼容回答")
        self.assertEqual(response.json["generation"]["tokens_per_second"], 20)

    def test_stats_polling_during_inference_is_independent_and_ids_are_isolated(self):
        def inference(prompt, backend):
            stats = self.web.current_generation_stats()
            stats.record({"timings": {"predicted_n": 9, "predicted_ms": 300}})
            with self.web.app.test_client() as poller:
                live = poller.get("/generation-stats/request-first")
                self.assertEqual(live.status_code, 200)
                self.assertEqual(live.headers["Cache-Control"], "no-store")
                self.assertEqual(live.json["status"], "generating")
                self.assertEqual(live.json["generated_tokens"], 9)
                busy = poller.post("/ask-text", json={"text": "另一个问题", "generation_id": "request-busy"})
                self.assertEqual(busy.status_code, 429)
            return "回答"

        with mock.patch.object(self.web, "ask_backend", side_effect=inference):
            first = self.client.post("/ask-text", json={"text": "你好", "generation_id": "request-first"})
        with mock.patch.object(self.web, "ask_backend", return_value="第二次回答"):
            second = self.client.post("/ask-text", json={"text": "再见", "generation_id": "request-second"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json["generation"]["generated_tokens"], 9)
        self.assertIsNone(second.json["generation"]["generated_tokens"])
        self.assertEqual(self.client.get("/generation-stats/request-first").json, first.json["generation"])
        self.assertEqual(self.client.get("/generation-stats/request-busy").status_code, 404)

    def test_generation_ids_reject_duplicates_and_invalid_values(self):
        with mock.patch.object(self.web, "ask_backend", return_value="回答") as inference:
            first = self.client.post("/ask-text", json={"text": "你好", "generation_id": "same-request"})
            duplicate = self.client.post("/ask-text", json={"text": "你好", "generation_id": "same-request"})
            invalid = self.client.post("/ask-text", json={"text": "你好", "generation_id": "bad/id"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(inference.call_count, 1)
        self.assertFalse(self.web.model_request_lock.locked())

    def test_audio_returns_final_stats_and_cleans_files_even_on_failure(self):
        with tempfile.TemporaryDirectory(prefix="qwen-audio-generation-") as directory:
            recordings = Path(directory)

            def convert(source, destination):
                self.assertTrue(source.exists())
                destination.write_bytes(b"converted wav")

            def answer(audio_path, instruction):
                self.assertTrue(audio_path.exists())
                self.web.current_generation_stats().record({"timings": {"predicted_n": 20, "predicted_ms": 1000}})
                return "用户原话", "模型回答"

            for failure in (False, True):
                with self.subTest(failure=failure), mock.patch.object(self.web, "RECORDINGS_DIR", recordings), mock.patch.object(self.web, "convert_to_wav", side_effect=convert), mock.patch.object(self.web, "ask_qwen3_audio", side_effect=RuntimeError("audio failed") if failure else answer):
                    identifier = "audio-failure" if failure else "audio-success"
                    response = self.client.post("/ask-audio", data={
                        "audio": (io.BytesIO(b"uploaded audio"), "sample.webm"),
                        "backend": "qwen3-gguf", "generation_id": identifier,
                    })
                    self.assertEqual(response.status_code, 500 if failure else 200)
                    self.assertEqual(list(recordings.iterdir()), [])
                    self.assertFalse(self.web.model_request_lock.locked())
                    state = self.client.get(f"/generation-stats/{identifier}").json
                    self.assertEqual(state["status"], "error" if failure else "complete")
                    if not failure:
                        self.assertEqual(response.json["transcript"], "用户原话")
                        self.assertEqual(response.json["generation_id"], identifier)
                        self.assertEqual(response.json["generation"]["tokens_per_second"], 20)

    def test_mnn_captures_native_decode_counters_before_reset(self):
        class NativeModel:
            def __init__(self, finish_status):
                self.reset_count = 0
                self.context_reads = []
                self._context = None
                self.finish_status = finish_status

            def reset(self):
                self.reset_count += 1
                self._context = SimpleNamespace(gen_seq_len=0, decode_us=0)

            def set_config(self, config):
                return True

            def response(self, prompt, stream):
                self._context = SimpleNamespace(
                    gen_seq_len=25, decode_us=2_000_000,
                    prefill_us=9_000_000, status=self.finish_status,
                )
                return "  MNN 回答  "

            @property
            def context(self):
                self.context_reads.append(self.reset_count)
                return self._context

        # NORMAL_FINISHED counts a sampled EOS that MNN never decodes;
        # MAX_TOKENS_FINISHED consists entirely of emitted tokens.
        for status, expected_tokens in ((1, 24), (2, 25)):
            with self.subTest(status=status):
                qwen = NativeModel(status)
                with mock.patch.object(self.web, "load_mnn_model", return_value=qwen):
                    response = self.client.post("/ask-text", json={
                        "text": "你好", "backend": "qwen25-mnn",
                        "generation_id": f"mnn-request-{status}",
                    })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["reply"], "MNN 回答")
                self.assertEqual(response.json["generation"]["generated_tokens"], expected_tokens)
                self.assertEqual(response.json["generation"]["tokens_per_second"], expected_tokens / 2)
                self.assertFalse(response.json["generation"]["estimated"])
                self.assertEqual(qwen.context_reads, [1])
                self.assertEqual(qwen.reset_count, 2)
                self.assertFalse(self.web.model_request_lock.locked())


if __name__ == "__main__":
    unittest.main()
