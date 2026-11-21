"""Real-Time Prometheus Metric Collector for vLLM Inference Engines.

Polls the /metrics endpoint of each vLLM engine instance and extracts:
- vllm:num_requests_running (active requests executing in model batches)
- vllm:num_requests_waiting (pending requests queued for scheduling)
- vllm:kv_cache_usage_perc (fraction of KV cache allocated, 0.0 to 1.0)
- vllm:request_prompt_tokens_count / sum (prefill tokens processed)
"""

from __future__ import annotations

import time

import httpx


class ReplicaMetricsCollector:
    def __init__(self, cache_ttl_sec: float = 2.0, poll_timeout_sec: float = 1.0):
        self.cache_ttl_sec = cache_ttl_sec
        self.poll_timeout_sec = poll_timeout_sec
        # endpoint -> (timestamp, metrics_dict)
        self._cache: dict[str, tuple[float, dict[str, float]]] = {}

    async def get_metrics(self, endpoint: str, force_refresh: bool = False) -> dict[str, float]:
        """Returns metrics for endpoint, using cache if fresh or polling asynchronously."""
        now = time.time()
        cached = self._cache.get(endpoint)
        if cached and not force_refresh:
            cached_time, data = cached
            if now - cached_time < self.cache_ttl_sec:
                return data

        try:
            metrics = await self._poll(endpoint)
            self._cache[endpoint] = (now, metrics)
            return metrics
        except Exception:
            # Fall back to cached if available
            if cached:
                return cached[1]
            return {
                "num_requests_running": 0.0,
                "num_requests_waiting": 0.0,
                "kv_cache_usage_perc": 0.0,
                "is_stale": True,
            }

    async def _poll(self, endpoint: str) -> dict[str, float]:
        clean_base = endpoint.rstrip("/")
        clean_base = clean_base.removesuffix("/v1")
        metrics_url = f"{clean_base}/metrics"

        async with httpx.AsyncClient(timeout=self.poll_timeout_sec) as client:
            resp = await client.get(metrics_url)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code} from {metrics_url}")
            return self.parse_prometheus_text(resp.text)

    @staticmethod
    def parse_prometheus_text(text: str) -> dict[str, float]:
        res = {
            "num_requests_running": 0.0,
            "num_requests_waiting": 0.0,
            "kv_cache_usage_perc": 0.0,
            "is_stale": False,
        }
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Check vllm:num_requests_running
            if line.startswith("vllm:num_requests_running"):
                try:
                    val = float(line.split()[-1])
                    res["num_requests_running"] = val
                except (ValueError, IndexError):
                    pass

            # Check vllm:num_requests_waiting (base gauge, not by_reason)
            elif line.startswith(("vllm:num_requests_waiting{", "vllm:num_requests_waiting ")):
                try:
                    val = float(line.split()[-1])
                    res["num_requests_waiting"] = val
                except (ValueError, IndexError):
                    pass

            # Check vllm:kv_cache_usage_perc or gpu_cache_usage_factor
            elif line.startswith(("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_factor")):
                try:
                    val = float(line.split()[-1])
                    res["kv_cache_usage_perc"] = val
                except (ValueError, IndexError):
                    pass

        return res


default_metrics_collector = ReplicaMetricsCollector()
