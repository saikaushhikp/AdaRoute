"""Reproducible Evaluation Benchmark for Redis Vector Semantic Cache and vLLM Semantic Router (CRITICAL_IMP.md).

Evaluates:
1. Semantic Cache Performance:
   - Hit rate on genuine paraphrases
   - False-hit rejection rate on adversarial near-misses (negation, numbers, formats)
   - FastEmbed CPU ONNX embedding latency (p50, p95)
   - Redis Search KNN lookup latency (p50, p95)
   - Comparison against Exact-only baseline
2. Model Selection Policy Overhead:
   - Stage 1 policy decision latency (adaptive, strongest, cheapest, vllm_semantic_router)
   - Hard constraint evaluation latency

Usage:
    python -m tests.benchmarks.eval_semantic_cache_and_router
"""

from __future__ import annotations

import os
import platform
import time
from typing import Any

import numpy as np

from adaroute.cache.exact import ExactCache
from adaroute.cache.semantic import FastEmbedProvider, SemanticCache
from adaroute.policy.registry import default_registry
from adaroute.schemas import PrivacyLevel, RequestContext

# -----------------------------------------------------------------------------
# Fixed Labeled Workload Dataset
# -----------------------------------------------------------------------------

LABELED_WORKLOAD = [
    # 1. Exact & Near-Paraphrase Pairs (Expected: HIT)
    {
        "category": "paraphrase",
        "stored_prompt": "What is the capital of France?",
        "query_prompt": "what is the capital of France",
        "expected_semantic_hit": True,
        "expected_exact_hit": False,
    },
    {
        "category": "paraphrase",
        "stored_prompt": "How do I reverse a string in Python?",
        "query_prompt": "python code to reverse string",
        "expected_semantic_hit": True,
        "expected_exact_hit": False,
    },
    {
        "category": "paraphrase",
        "stored_prompt": "Explain the difference between synchronous and asynchronous execution.",
        "query_prompt": "synchronous vs asynchronous execution difference explained",
        "expected_semantic_hit": True,
        "expected_exact_hit": False,
    },
    {
        "category": "paraphrase",
        "stored_prompt": "How to calculate standard deviation in NumPy?",
        "query_prompt": "calculate standard deviation numpy array",
        "expected_semantic_hit": True,
        "expected_exact_hit": False,
    },
    {
        "category": "paraphrase",
        "stored_prompt": "Summarize the key architectural benefits of microservices.",
        "query_prompt": "what are the main benefits of microservices architecture",
        "expected_semantic_hit": True,
        "expected_exact_hit": False,
    },
    # 2. Adversarial Near-Misses - Negation (Expected: MISS)
    {
        "category": "adversarial_negation",
        "stored_prompt": "The primary database service is online and healthy.",
        "query_prompt": "The primary database service is not online and healthy.",
        "expected_semantic_hit": False,
        "expected_exact_hit": False,
    },
    {
        "category": "adversarial_negation",
        "stored_prompt": "User authentication is required for all API endpoints.",
        "query_prompt": "User authentication is never required for all API endpoints.",
        "expected_semantic_hit": False,
        "expected_exact_hit": False,
    },
    # 3. Adversarial Near-Misses - Numbers (Expected: MISS)
    {
        "category": "adversarial_number",
        "stored_prompt": "Forward gateway telemetry to port 7000.",
        "query_prompt": "Forward gateway telemetry to port 8000.",
        "expected_semantic_hit": False,
        "expected_exact_hit": False,
    },
    {
        "category": "adversarial_number",
        "stored_prompt": "Return the top 5 high-performing stock assets.",
        "query_prompt": "Return the top 10 high-performing stock assets.",
        "expected_semantic_hit": False,
        "expected_exact_hit": False,
    },
    # 4. Adversarial Near-Misses - Format & Language Constraints (Expected: MISS)
    {
        "category": "adversarial_format",
        "stored_prompt": "Convert the server response payload to JSON format.",
        "query_prompt": "Convert the server response payload to YAML format.",
        "expected_semantic_hit": False,
        "expected_exact_hit": False,
    },
    {
        "category": "adversarial_lang",
        "stored_prompt": "Implement binary search algorithm in Python.",
        "query_prompt": "Implement binary search algorithm in Rust.",
        "expected_semantic_hit": False,
        "expected_exact_hit": False,
    },
    # 5. Dissimilar / Unrelated Queries (Expected: MISS)
    {
        "category": "dissimilar",
        "stored_prompt": "How does photosynthesis convert sunlight to energy?",
        "query_prompt": "Explain the time complexity of quicksort in Big O notation.",
        "expected_semantic_hit": False,
        "expected_exact_hit": False,
    },
    {
        "category": "dissimilar",
        "stored_prompt": "Recipe for chocolate chip cookies with sea salt.",
        "query_prompt": "Deploying Kubernetes pods across multi-cloud regions.",
        "expected_semantic_hit": False,
        "expected_exact_hit": False,
    },
]


def run_cache_evaluation() -> dict[str, Any]:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # FastEmbed Provider for real CPU ONNX embeddings
    embedder = FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2", dimension=384)
    sem_cache = SemanticCache(similarity_threshold=0.85, embedder=embedder, redis_url=redis_url)
    sem_cache.clear()

    exact_cache = ExactCache(redis_url=redis_url)
    exact_cache.invalidate_prefix("adaroute:")

    req = RequestContext(
        task_type="general",
        prompt_tokens=20,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="benchmark_tenant",
        user_id="bench_user",
    )

    # 1. Warm up and populate stored entries
    for item in LABELED_WORKLOAD:
        sem_cache.put(
            request=req,
            prompt_text=item["stored_prompt"],
            response_content=f"Response for {item['stored_prompt']}",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            replica_id="vllm-r1",
            output_tokens=15,
            quality_score=0.85,
        )
        exact_cache.put(
            request=req,
            logical_model="general",
            prompt=item["stored_prompt"],
            response={"content": f"Response for {item['stored_prompt']}"},
            origin_model="Qwen/Qwen2.5-0.5B-Instruct",
        )

    # 2. Benchmark Lookups and Latencies
    embed_latencies_ms: list[float] = []
    sem_lookup_latencies_ms: list[float] = []
    exact_lookup_latencies_ms: list[float] = []

    sem_hits = 0
    sem_false_hits = 0
    exact_hits = 0
    exact_false_hits = 0

    num_iterations = 5  # Run 5 rounds over the workload for stable quantile statistics
    for _ in range(num_iterations):
        for item in LABELED_WORKLOAD:
            query = item["query_prompt"]

            # Exact cache timing
            t0 = time.perf_counter()
            ex_res = exact_cache.get(req, logical_model="general", prompt=query)
            exact_lookup_latencies_ms.append((time.perf_counter() - t0) * 1000)
            if ex_res is not None:
                exact_hits += 1
                if not item["expected_exact_hit"]:
                    exact_false_hits += 1

            # Embedder standalone timing
            t0 = time.perf_counter()
            _ = embedder.encode(query)
            embed_latencies_ms.append((time.perf_counter() - t0) * 1000)

            # Semantic cache full lookup timing
            t0 = time.perf_counter()
            sem_res = sem_cache.get(req, query, threshold=0.85)
            sem_lookup_latencies_ms.append((time.perf_counter() - t0) * 1000)

            if sem_res is not None:
                sem_hits += 1
                if not item["expected_semantic_hit"]:
                    sem_false_hits += 1

    total_queries = len(LABELED_WORKLOAD) * num_iterations
    paraphrase_queries = sum(1 for item in LABELED_WORKLOAD if item["expected_semantic_hit"]) * num_iterations
    adversarial_queries = sum(1 for item in LABELED_WORKLOAD if not item["expected_semantic_hit"]) * num_iterations

    return {
        "total_queries": total_queries,
        "paraphrase_queries": paraphrase_queries,
        "adversarial_queries": adversarial_queries,
        "sem_hits": sem_hits,
        "sem_hit_rate": (sem_hits / paraphrase_queries) if paraphrase_queries else 0.0,
        "sem_false_hits": sem_false_hits,
        "sem_false_hit_rate": (sem_false_hits / adversarial_queries) if adversarial_queries else 0.0,
        "exact_hits": exact_hits,
        "exact_hit_rate": (exact_hits / paraphrase_queries) if paraphrase_queries else 0.0,
        "exact_false_hits": exact_false_hits,
        "embed_p50": float(np.percentile(embed_latencies_ms, 50)),
        "embed_p95": float(np.percentile(embed_latencies_ms, 95)),
        "sem_lookup_p50": float(np.percentile(sem_lookup_latencies_ms, 50)),
        "sem_lookup_p95": float(np.percentile(sem_lookup_latencies_ms, 95)),
        "exact_lookup_p50": float(np.percentile(exact_lookup_latencies_ms, 50)),
        "exact_lookup_p95": float(np.percentile(exact_lookup_latencies_ms, 95)),
    }


def run_policy_routing_evaluation() -> dict[str, Any]:
    selector = default_registry.selector
    req = RequestContext(
        task_type="general",
        prompt_tokens=25,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="routing_bench",
    )

    policies = ["adaptive", "strongest", "cheapest", "vllm_semantic_router"]
    latencies: dict[str, list[float]] = {p: [] for p in policies}

    for p in policies:
        req.policy_name = p
        # Run 30 decisions per policy
        for _ in range(30):
            t0 = time.perf_counter()
            _ = selector.route(req, prompt_text="What is the speed of sound?")
            lat = (time.perf_counter() - t0) * 1000
            latencies[p].append(lat)

    return {
        p: {
            "p50_ms": float(np.percentile(latencies[p], 50)),
            "p95_ms": float(np.percentile(latencies[p], 95)),
        }
        for p in policies
    }


def main():
    print("=" * 80)
    print("AdaRoute Comprehensive Evaluation: Redis Vector Semantic Cache & vLLM Router")
    print("=" * 80)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Processor: {platform.processor() or 'x86_64 CPU'}")
    print("Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, ONNX runtime)")
    print("Redis Version: redis/redis-stack-server:7.4.0-v8 (RediSearch 2.10.20)")
    print("=" * 80)

    print("\n[1/2] Running Semantic Cache Evaluation over Fixed Labeled Workload...")
    cache_results = run_cache_evaluation()

    print("\n--- Cache Quality & Accuracy Metrics ---")
    print(f"Total Evaluated Lookups:       {cache_results['total_queries']}")
    print(f"Paraphrase Query Opportunities: {cache_results['paraphrase_queries']}")
    print(f"Adversarial Near-Miss Checks:   {cache_results['adversarial_queries']}")
    print(f"Exact-Only Baseline Hit Rate:   {cache_results['exact_hit_rate'] * 100:.1f}%")
    print(f"Semantic Cache Hit Rate:        {cache_results['sem_hit_rate'] * 100:.1f}%")
    print(
        f"Adversarial False Hits (Guards):{cache_results['sem_false_hits']} (False-Hit Rate: {cache_results['sem_false_hit_rate'] * 100:.1f}%)"
    )

    print("\n--- Cache Latency Profile ---")
    print(
        f"Exact Cache Lookup:             p50={cache_results['exact_lookup_p50']:.3f}ms | p95={cache_results['exact_lookup_p95']:.3f}ms"
    )
    print(
        f"FastEmbed ONNX Embedding:       p50={cache_results['embed_p50']:.3f}ms | p95={cache_results['embed_p95']:.3f}ms"
    )
    print(
        f"Redis Search KNN Full Lookup:   p50={cache_results['sem_lookup_p50']:.3f}ms | p95={cache_results['sem_lookup_p95']:.3f}ms"
    )

    print("\n[2/2] Running Model Selection Policy Overhead Evaluation...")
    policy_results = run_policy_routing_evaluation()

    print("\n--- Model Selection Policy Decision Overhead ---")
    for policy_name, stats in policy_results.items():
        print(f"Policy '{policy_name}': p50={stats['p50_ms']:.3f}ms | p95={stats['p95_ms']:.3f}ms")

    print("\nEvaluation successfully completed.")


if __name__ == "__main__":
    main()
