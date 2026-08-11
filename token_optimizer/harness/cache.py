"""Caching mechanisms for token optimization."""
"""Phase 4a — semantic caching.

A `CachingClient` wraps any inner client. Before calling the model it checks
whether a semantically similar request was already answered; if so it returns
the cached answer for ~0 tokens and ~0 cost. This is usually the single biggest
saver on real traffic, because real traffic repeats (FAQs, common questions).

The catch the harness exists to catch: a cache HIT returns the answer that was
produced for a *different but similar* question. If the threshold is too loose,
you serve subtly wrong answers and save tokens by lying. So the cached answer is
scored against the NEW request's gold answer — quality degradation shows up
immediately as an F1 drop. Tune the threshold against that, not against hit rate.

Reuses the Phase 2 relevance scorers (lexical / embedding) for similarity, so
the question-to-question matching is the same machinery, just applied to the
cache keys instead of context sentences.
"""

import random
import re
from typing import Dict, List, Optional

from .client import LLMClient, GenerationResult


class CachingClient(LLMClient):
    name = "cache"

    def __init__(self, inner: LLMClient, relevance, threshold: float = 0.9, key: str = "question"):
        self.inner = inner
        self.relevance = relevance
        self.threshold = threshold
        self.key = key
        self._keys: List[str] = []
        self._answers: List[str] = []

    def generate(self, prompt: str, meta: Optional[Dict] = None) -> GenerationResult:
        meta = meta or {}
        key_text = meta.get(self.key) or prompt

        if self._keys:
            sims = self.relevance.score(key_text, self._keys)
            best_i = max(range(len(sims)), key=lambda i: sims[i])
            if sims[best_i] >= self.threshold:
                # HIT: return cached answer, no model call, no tokens, no cost.
                return GenerationResult(
                    self._answers[best_i],
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost=0.0,
                    info={"cache_hit": True, "sim": round(float(sims[best_i]), 3)},
                )

        # MISS: call the real model, then store.
        res = self.inner.generate(prompt, meta)
        self._keys.append(key_text)
        self._answers.append(res.text)
        res.info = {**(res.info or {}), "cache_hit": False}
        return res


# ---------------------------------------------------------------------------
# Traffic generator — so caching has something to hit.
# ---------------------------------------------------------------------------

_PARAPHRASE_RULES = [
    (r"^What is ", "What's "),
    (r"^What are ", "What're "),
    (r"^Who wrote ", "Who is the author of "),
    (r"^How many ", "What number of "),
    (r"^What year ", "In what year "),
]


def _paraphrase(question: str, rng: random.Random) -> str:
    """Light surface rewrite — same meaning, different words. Exact-match caches
    miss these; a semantic cache should still hit them."""
    q = question
    for pat, repl in _PARAPHRASE_RULES:
        if re.match(pat, q):
            q = re.sub(pat, repl, q)
            break
    if rng.random() < 0.5:
        q = rng.choice(["Can you tell me ", "I want to know ", "Please tell me "]) + q[0].lower() + q[1:]
    return q


def make_traffic(
    dataset: List[Dict],
    n_requests: int = 200,
    zipf_s: float = 1.3,
    paraphrase_rate: float = 0.5,
    seed: int = 0,
) -> List[Dict]:
    """Build a realistic request stream from a small set of unique questions.

    A few questions are popular (Zipf-like), the rest are rare — like real
    traffic. A fraction of requests are paraphrased so you're testing SEMANTIC
    matching, not exact-string matching. Each request keeps its original context
    and gold answers so it can still be scored.
    """
    rng = random.Random(seed)
    base = list(dataset)
    m = len(base)

    # Zipf-like popularity weights over the unique questions.
    weights = [1.0 / ((i + 1) ** zipf_s) for i in range(m)]
    total = sum(weights)
    weights = [w / total for w in weights]

    traffic: List[Dict] = []
    for _ in range(n_requests):
        ex = base[_weighted_index(weights, rng)]
        q = ex["question"]
        if rng.random() < paraphrase_rate:
            q = _paraphrase(q, rng)
        traffic.append({"question": q, "context": ex["context"], "answers": ex["answers"]})
    return traffic


def _weighted_index(weights: List[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1