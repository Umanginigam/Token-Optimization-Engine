"""Baseline runner for token optimization."""
"""Phase 1 entry point — run the full-context BASELINE and print the numbers.

Usage:
    # Offline plumbing test (no network, no key):
    python run_baseline.py

    # Real model, free local (after `ollama pull llama3.2 && ollama serve`):
    LLM_PROVIDER=ollama LLM_MODEL=llama3.2 python run_baseline.py

    # Real model, free Groq API tier:
    LLM_PROVIDER=groq LLM_API_KEY=gsk_xxx LLM_MODEL=llama-3.1-8b-instant python run_baseline.py

    # Bigger, real dataset (pip install datasets):
    DATASET=squad N=100 python run_baseline.py
"""

import os

from harness.client import build_client_from_env
from harness.dataset import load_sample, load_squad, load_hotpotqa
from harness.pipeline import Pipeline, identity_transform
from harness.runner import evaluate, print_summary


def load_dataset_from_env():
    name = os.environ.get("DATASET", "sample").lower()
    n = int(os.environ.get("N", "100"))
    if name == "sample":
        return load_sample(), "built-in sample"
    if name == "squad":
        return load_squad(n=n), f"SQuAD (first {n})"
    if name in ("hotpot", "hotpotqa"):
        return load_hotpotqa(n=n), f"HotpotQA (first {n})"
    raise ValueError(f"Unknown DATASET: {name!r}")


def main():
    client = build_client_from_env()
    dataset, ds_label = load_dataset_from_env()

    print(f"client  : {client.name}")
    print(f"dataset : {ds_label}  ({len(dataset)} examples)")
    print("running baseline (full context)...\n")

    pipeline = Pipeline(client, context_transform=identity_transform)
    summary, _rows = evaluate(pipeline, dataset, verbose=True)
    print_summary("BASELINE  (full context)", summary)

    if client.name == "mock":
        print(
            "\nNote: 'mock' is an offline lexical-overlap baseline, not a real\n"
            "model — low accuracy here is expected. It only proves the harness\n"
            "runs end to end. Set LLM_PROVIDER=ollama or =groq for real numbers."
        )


if __name__ == "__main__":
    main()