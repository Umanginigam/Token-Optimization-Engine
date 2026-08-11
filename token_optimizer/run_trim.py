"""Trim runner for token optimization."""
"""Phase 2 entry point — BASELINE vs TRIMMED, with the savings-vs-quality row.

Usage:
    # Offline plumbing (mock model + lexical trimmer, no network/keys):
    python run_trim.py

    # Real model + lexical trimmer (no extra installs):
    LLM_PROVIDER=groq LLM_API_KEY=gsk_xxx python run_trim.py

    # Real model + semantic trimmer (pip install sentence-transformers):
    LLM_PROVIDER=groq LLM_API_KEY=gsk_xxx RELEVANCE=embedding python run_trim.py

    # Tune aggressiveness and dataset:
    KEEP_RATIO=0.4 DATASET=squad N=100 LLM_PROVIDER=groq LLM_API_KEY=gsk_xxx python run_trim.py

    # Peek at what got trimmed for the first example:
    SHOW=1 python run_trim.py
"""

import os

from harness.client import build_client_from_env
from harness.dataset import load_from_env
from harness.pipeline import Pipeline, identity_transform
from harness.runner import evaluate, print_summary, compare
from harness.trim import build_relevance_from_env, make_trimmer


def maybe_show_example(dataset, relevance, keep_ratio):
    if os.environ.get("SHOW", "0") != "1" or not dataset:
        return
    trim = make_trimmer(relevance, keep_ratio=keep_ratio)
    ex = dataset[0]
    trimmed = trim(ex["question"], ex["context"])
    print("\n--- trim preview (example 1) " + "-" * 28)
    print(f"Q: {ex['question']}")
    print(f"\nFULL context ({len(ex['context'])} chars):\n  {ex['context']}")
    print(f"\nTRIMMED context ({len(trimmed)} chars):\n  {trimmed}")
    print("-" * 56)


def main():
    client = build_client_from_env()
    dataset, ds_label = load_from_env()
    relevance = build_relevance_from_env()
    keep_ratio = float(os.environ.get("KEEP_RATIO", "0.5"))

    print(f"client    : {client.name}")
    print(f"dataset   : {ds_label}  ({len(dataset)} examples)")
    print(f"relevance : {relevance.name}")
    print(f"keep_ratio: {keep_ratio}")

    maybe_show_example(dataset, relevance, keep_ratio)

    # 1) Baseline: full context.
    base_summary, _ = evaluate(Pipeline(client, identity_transform), dataset, verbose=False)
    print_summary("BASELINE  (full context)", base_summary)

    # 2) Candidate: trimmed context (same harness, swapped transform).
    trim = make_trimmer(relevance, keep_ratio=keep_ratio)
    trim_summary, _ = evaluate(Pipeline(client, trim), dataset, verbose=False)
    print_summary(f"TRIMMED  ({relevance.name}, keep_ratio={keep_ratio})", trim_summary)

    # 3) The row that is the whole product.
    compare(base_summary, trim_summary, candidate_label=f"trim:{relevance.name} keep={keep_ratio}")

    if client.name == "mock":
        print(
            "\nNote: 'mock' is an offline lexical baseline, not a real model. The\n"
            "savings number is real; the quality numbers only become meaningful\n"
            "with a real model (LLM_PROVIDER=ollama or =groq)."
        )


if __name__ == "__main__":
    main()