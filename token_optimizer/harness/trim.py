"""Token trimming functionality."""
"""Phase 2 — relevance-based context trimming.

This is the first real optimizer. It plugs into the Phase 1 seam:

    Pipeline(client, context_transform=make_trimmer(relevance, keep_ratio=0.5))

Nothing else in the harness changes. The trimmer:
  1. splits the context into sentences,
  2. scores each sentence's relevance to the question,
  3. keeps the top fraction (keep_ratio), always at least `min_sentences`,
  4. reassembles them in ORIGINAL order (so reading flow is preserved).

The relevance scorer is itself pluggable, mirroring the client design:
  - LexicalRelevance:  pure Python, no deps, no downloads. Runs anywhere.
  - EmbeddingRelevance: semantic, via sentence-transformers (free, but pulls a
                        small model on first use). Better quality.

Both expose:  score(question: str, sentences: list[str]) -> list[float]
"""

import math
import os
import re
from collections import Counter
from typing import Callable, List

from .scoring import normalize_answer


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Rough sentence split. Good enough for ranking; not linguistically perfect."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> List[str]:
    return normalize_answer(text).split()


# ---------------------------------------------------------------------------
# Relevance scorers (pluggable)
# ---------------------------------------------------------------------------

class LexicalRelevance:
    """TF-IDF cosine similarity between the question and each sentence.

    IDF is computed over the sentences of THIS context (treating each sentence
    as a document), so rare, distinctive words count more than common ones.
    Zero dependencies, zero downloads — runs on any machine instantly.
    """

    name = "lexical"

    def score(self, question: str, sentences: List[str]) -> List[float]:
        docs = [_tokenize(s) for s in sentences]
        q = _tokenize(question)

        n = len(docs)
        df = Counter()
        for d in docs:
            for w in set(d):
                df[w] += 1

        def idf(w: str) -> float:
            return math.log((n + 1) / (df.get(w, 0) + 1)) + 1.0

        q_tf = Counter(q)
        q_vec = {w: q_tf[w] * idf(w) for w in q_tf}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scores: List[float] = []
        for d in docs:
            d_tf = Counter(d)
            d_vec = {w: d_tf[w] * idf(w) for w in d_tf}
            dot = sum(q_vec.get(w, 0.0) * d_vec.get(w, 0.0) for w in q_vec)
            d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
            scores.append(dot / (q_norm * d_norm))
        return scores


class EmbeddingRelevance:
    """Semantic similarity via sentence-transformers.

    Requires: pip install sentence-transformers
    Downloads a small model (e.g. all-MiniLM-L6-v2, ~90MB) once, then runs
    locally for free. Catches paraphrases that lexical overlap misses
    (question says "absorb", context says "take in").
    """

    name = "embedding"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def score(self, question: str, sentences: List[str]) -> List[float]:
        from sentence_transformers import util
        q_emb = self.model.encode(question, convert_to_tensor=True)
        s_emb = self.model.encode(sentences, convert_to_tensor=True)
        return util.cos_sim(q_emb, s_emb)[0].tolist()


def build_relevance_from_env():
    method = os.environ.get("RELEVANCE", "lexical").lower()
    if method == "lexical":
        return LexicalRelevance()
    if method in ("embedding", "embeddings", "semantic"):
        return EmbeddingRelevance(os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2"))
    raise ValueError(f"Unknown RELEVANCE: {method!r}")


# ---------------------------------------------------------------------------
# The trimmer (a context_transform factory)
# ---------------------------------------------------------------------------

def make_trimmer(
    relevance,
    keep_ratio: float = 0.5,
    min_sentences: int = 1,
) -> Callable[[str, str], str]:
    """Return a context_transform(question, context) -> trimmed_context.

    keep_ratio: fraction of sentences to keep (0.5 keeps the top half).
    min_sentences: never drop below this many (so context is never empty).
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1]")

    def trim(question: str, context: str) -> str:
        sentences = split_sentences(context)
        if len(sentences) <= min_sentences:
            return context

        scores = relevance.score(question, sentences)

        k = max(min_sentences, math.ceil(len(sentences) * keep_ratio))
        k = min(k, len(sentences))

        # indices of the top-k sentences by score...
        top = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:k]
        keep = set(top)

        # ...reassembled in original reading order.
        return " ".join(sentences[i] for i in range(len(sentences)) if i in keep)

    return trim