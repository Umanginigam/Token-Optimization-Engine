"""Routing runner for token optimization."""
"""Phase 4b entry point — model routing on cost vs quality.

This demo runs OFFLINE with two stand-in "models" of different quality and price
so you can see the mechanics without two API keys:
  - cheap  : returns the first context sentence  (weaker, cheap)
  - strong : returns the best-overlap sentence    (better, expensive)

Swap these for real PricedClient(build-your-own real client, price_in, price_out)
when you wire it to Groq/OpenAI/etc. The PRICES BELOW ARE ILLUSTRATIVE — edit
them to your providers' real rates.

Usage:
    python run_routing.py
    MAX_CONTEXT_CHARS=600 python run_routing.py
"""

import os
import re
from typing import Optional, Dict

from harness.client import LLMClient, GenerationResult
from harness.tokens import count_tokens
from harness.dataset import load_from_env
from harness.scoring import normalize_answer
from harness.pipeline import Pipeline, identity_transform
from harness.runner import evaluate, print_summary
from harness.routing import (
    PricedClient, RoutingClient, CascadeClient,
    length_router, accept_if_nonempty_short,
)

# ----- ILLUSTRATIVE prices (USD per 1,000,000 tokens). EDIT THESE. -----
CHEAP_IN, CHEAP_OUT = 0.50, 1.50
STRONG_IN, STRONG_OUT = 5.00, 15.00


class _FirstSentenceModel(LLMClient):
    """Stand-in cheap model: always returns the first context sentence."""
    name = "cheap-model"

    def generate(self, prompt: str, meta: Optional[Dict] = None) -> GenerationResult:
        ctx = (meta or {}).get("context", "")
        first = re.split(r"(?<=[.!?])\s+", ctx.strip())[0] if ctx.strip() else ""
        return GenerationResult(first, count_tokens(prompt), count_tokens(first))


class _BestSentenceModel(LLMClient):
    """Stand-in strong model: returns the best-overlap context sentence."""
    name = "strong-model"

    def generate(self, prompt: str, meta: Optional[Dict] = None) -> GenerationResult:
        meta = meta or {}
        q = set(normalize_answer(meta.get("question", "")).split())
        ctx = meta.get("context", "")
        sents = [s for s in re.split(r"(?<=[.!?])\s+", ctx.strip()) if s] or [ctx]
        best = max(sents, key=lambda s: len(q & set(normalize_answer(s).split())))
        return GenerationResult(best, count_tokens(prompt), count_tokens(best))


def main():
    dataset, ds_label = load_from_env()
    max_ctx = int(os.environ.get("MAX_CONTEXT_CHARS", "600"))

    cheap = PricedClient(_FirstSentenceModel(), CHEAP_IN, CHEAP_OUT, label="cheap")
    strong = PricedClient(_BestSentenceModel(), STRONG_IN, STRONG_OUT, label="strong")

    print(f"dataset : {ds_label}  ({len(dataset)} examples)")
    print(f"prices  : cheap ${CHEAP_IN}/{CHEAP_OUT}  strong ${STRONG_IN}/{STRONG_OUT} per 1M (in/out)")

    # Four strategies, all scored by the same harness.
    all_strong, _ = evaluate(Pipeline(strong, identity_transform), dataset, verbose=False)
    print_summary("ALL STRONG  (expensive everywhere)", all_strong)

    all_cheap, _ = evaluate(Pipeline(cheap, identity_transform), dataset, verbose=False)
    print_summary("ALL CHEAP  (cheap everywhere)", all_cheap)

    routed_client = RoutingClient(cheap, strong, length_router(max_context_chars=max_ctx))
    routed, _ = evaluate(Pipeline(routed_client, identity_transform), dataset, verbose=False)
    print_summary(f"ROUTED  (cheap if context<= {max_ctx} chars)", routed)

    cascade_client = CascadeClient(cheap, strong, accept_if_nonempty_short(max_words=8))
    cascade, _ = evaluate(Pipeline(cascade_client, identity_transform), dataset, verbose=False)
    print_summary("CASCADE  (cheap first, escalate if weak)", cascade)

    # The comparison that matters: cost vs quality, all four side by side.
    line = "=" * 64
    print("\n" + line)
    print(" COST vs QUALITY  (vs ALL STRONG baseline)")
    print(line)
    print(f"{'strategy':>12} | {'$ total':>9} | {'$ saved %':>9} | {'F1':>6} | {'F1 d':>6}")
    print("-" * 64)
    base_cost = all_strong["total_cost"] or 1e-9
    for label, s in [("all strong", all_strong), ("all cheap", all_cheap),
                     ("routed", routed), ("cascade", cascade)]:
        saved = 100.0 * (1 - s["total_cost"] / base_cost)
        f1_d = s["f1"] - all_strong["f1"]
        print(f"{label:>12} | {s['total_cost']:>9.4f} | {saved:>9.1f} | {s['f1']:>6.2f} | {f1_d:>+6.2f}")
    print("-" * 64)
    print("The win is a strategy that saves $ with F1 d near 0. 'All cheap'")
    print("usually saves the most $ but drops quality — routing/cascade aim to")
    print("keep quality while still cutting most of the cost.")


if __name__ == "__main__":
    main()