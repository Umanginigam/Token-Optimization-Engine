"""Phase 4a entry point — semantic caching on a realistic traffic stream.

Usage:
    # Offline plumbing (mock + lexical similarity):
    python run_cache.py

    # Real model + semantic cache:
    LLM_PROVIDER=groq LLM_API_KEY=gsk_xxx RELEVANCE=embedding python run_cache.py

Env:
    REQUESTS    number of requests in the simulated stream (default 200)
    PARAPHRASE  fraction of requests that are paraphrased (default 0.5)
    THRESHOLDS  comma-separated cache similarity thresholds to sweep
                (default 0.99,0.95,0.90,0.85,0.80)

The harness scores every cached answer against the NEW request's gold answer,
so a too-loose threshold shows up as falling F1 even as the hit rate climbs.
"""

import os

from harness.client import build_client_from_env
from harness.dataset import load_from_env
from harness.trim import build_relevance_from_env
from harness.cache import CachingClient, make_traffic
from harness.pipeline import Pipeline, identity_transform
from harness.runner import evaluate, print_summary

DEFAULT_THRESHOLDS = [0.99, 0.95, 0.90, 0.85, 0.80]


def main():
    client = build_client_from_env()
    dataset, ds_label = load_from_env()
    relevance = build_relevance_from_env()

    n_requests = int(os.environ.get("REQUESTS", "200"))
    paraphrase = float(os.environ.get("PARAPHRASE", "0.5"))
    raw = os.environ.get("THRESHOLDS")
    thresholds = [float(x) for x in raw.split(",")] if raw else DEFAULT_THRESHOLDS

    traffic = make_traffic(dataset, n_requests=n_requests, paraphrase_rate=paraphrase)

    print(f"client    : {client.name}")
    print(f"dataset   : {ds_label}  ({len(dataset)} unique questions)")
    print(f"traffic   : {len(traffic)} requests  (paraphrase_rate={paraphrase})")
    print(f"relevance : {relevance.name}")

    # No cache: pay for every request.
    base, _ = evaluate(Pipeline(client, identity_transform), traffic, verbose=False)
    print_summary("NO CACHE  (every request hits the model)", base)

    # Sweep cache thresholds. Each needs a fresh cache + fresh client state.
    print("\n" + "=" * 56)
    print(" CACHE THRESHOLD SWEEP")
    print("=" * 56)
    print(f"{'thresh':>7} | {'hit %':>6} | {'tok saved %':>11} | {'F1':>6} | {'F1 d':>6}")
    print("-" * 56)
    for th in thresholds:
        cached_client = CachingClient(build_client_from_env(), relevance, threshold=th)
        s, _ = evaluate(Pipeline(cached_client, identity_transform), traffic, verbose=False)
        saved = 100.0 * (1 - s["avg_total_tokens"] / max(base["avg_total_tokens"], 1e-9))
        f1_d = s["f1"] - base["f1"]
        print(f"{th:>7.2f} | {s['cache_hit_rate']:>6.1f} | {saved:>11.1f} | {s['f1']:>6.2f} | {f1_d:>+6.2f}")

    print("-" * 56)
    print("Read it as: higher hit% and token savings are good, but watch F1 d.")
    print("The right threshold is the loosest one where F1 d stays ~0.")

    if client.name == "mock":
        print(
            "\nNote: 'mock' is offline and not a real model — hit rates and token\n"
            "savings are real; F1 becomes meaningful with LLM_PROVIDER=ollama/groq."
        )


if __name__ == "__main__":
    main()