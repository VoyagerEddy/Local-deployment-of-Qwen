"""Request-local generation throughput, shared with lightweight status polling."""

import math
import threading
import time


def nonnegative_number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return value
    return None


class GenerationStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.updated_at = time.monotonic()
        self.begin()

    def begin(self):
        # MNN audio transcription and the answer are separate generation phases.
        with self._lock:
            self.status = "waiting"
            self.generated_tokens = None
            self.tokens_per_second = None
            self.generation_seconds = None
            self.estimated = False
            self._first_token_at = None
            self._chunks = 0
            self._exact_count = False
            self._exact_timing = False
            self.updated_at = time.monotonic()

    def record(self, data, has_token=False):
        with self._lock:
            now = time.monotonic()
            timings = data.get("timings") or {}
            usage = data.get("usage") or {}
            count = nonnegative_number(timings.get("predicted_n"))
            if count is None:
                count = nonnegative_number(usage.get("completion_tokens"))
            if has_token or (count is not None and count > 0):
                self.status = "generating"
                if self._first_token_at is None:
                    self._first_token_at = now
            if has_token:
                self._chunks += 1
            if count is not None:
                self.generated_tokens = int(count)
                self._exact_count = True
            elif not self._exact_count and self._chunks:
                # Some compatible servers batch tokens into chunks. Label the
                # temporary chunk count as approximate until usage arrives.
                self.generated_tokens = self._chunks

            milliseconds = nonnegative_number(timings.get("predicted_ms"))
            rate = nonnegative_number(timings.get("predicted_per_second"))
            if milliseconds is not None and milliseconds > 0:
                self.generation_seconds = milliseconds / 1000
                if rate is None and count is not None:
                    rate = count / self.generation_seconds
            if rate is not None and (milliseconds is None or milliseconds > 0):
                self.tokens_per_second = rate
                self._exact_timing = True
            elif not self._exact_timing and self._first_token_at is not None:
                seconds = now - self._first_token_at
                self.generation_seconds = seconds
                # The first chunk's latency includes prefill. Count intervals
                # after that chunk to avoid a spurious first-token speed spike.
                if seconds > 0 and self._chunks > 1:
                    intervals = max(0, (self.generated_tokens or 0) - 1)
                    self.tokens_per_second = intervals / seconds
            self.estimated = not (self._exact_count and self._exact_timing)
            self.updated_at = now

    def finish(self, error=False):
        with self._lock:
            self.status = "error" if error else "complete"
            if error:
                self.generated_tokens = None
                self.tokens_per_second = None
                self.generation_seconds = None
            self.updated_at = time.monotonic()

    def snapshot(self):
        with self._lock:
            return {
                "status": self.status,
                "generated_tokens": self.generated_tokens,
                "tokens_per_second": round(self.tokens_per_second, 2) if self.tokens_per_second is not None else None,
                "generation_seconds": round(self.generation_seconds, 3) if self.generation_seconds is not None else None,
                "estimated": self.estimated,
            }


class GenerationStatsRegistry:
    MAX_ENTRIES = 128
    TTL_SECONDS = 900

    def __init__(self):
        self._lock = threading.Lock()
        self._entries = {}

    def _prune(self):
        now = time.monotonic()
        for key, stats in list(self._entries.items()):
            if stats.status in {"complete", "error"} and now - stats.updated_at > self.TTL_SECONDS:
                del self._entries[key]

    def create(self, generation_id):
        with self._lock:
            self._prune()
            if generation_id in self._entries:
                raise FileExistsError("generation_id already exists")
            if len(self._entries) >= self.MAX_ENTRIES:
                for key, stats in list(self._entries.items()):
                    if stats.status in {"complete", "error"}:
                        del self._entries[key]
                        break
                else:
                    raise RuntimeError("generation statistics are busy")
            stats = GenerationStats()
            self._entries[generation_id] = stats
            return stats

    def snapshot(self, generation_id):
        with self._lock:
            self._prune()
            stats = self._entries.get(generation_id)
            return stats.snapshot() if stats else None
