"""Request routing functionality."""
"""Phase 4b — model routing (and cost).

Routing's value is COST, not token count: a cheap model and an expensive model
use about the same tokens for the same prompt, but cost very different amounts
per token. So this module introduces a price wrapper and routes between a cheap
and an expensive model. On real workloads, routing usually saves more money than
prompt compression does.

  PricedClient   — wraps any client, attaches a $/1M-token price, fills in cost.
  RoutingClient  — picks cheap vs expensive per request via a router() function.
  CascadeClient  — calls cheap first, escalates to expensive only if the cheap
                   answer fails an accept() test. Honestly charges for BOTH calls
                   on escalation, because you really do pay for the failed attempt.

The harness scores quality the same way, so a router that sends too much to the
cheap model shows up as an F1 drop, and the cost saving shows up next to it.
NOTE: the example prices in the entry script are ILLUSTRATIVE — set your own.
"""

from typing import Callable, Dict, Optional

from .client import LLMClient, GenerationResult


class PricedClient(LLMClient):
    """Attach a price to any client. price_* are USD per 1,000,000 tokens."""

    def __init__(self, inner: LLMClient, price_in: float, price_out: float, label: str = ""):
        self.inner = inner
        self.price_in = price_in
        self.price_out = price_out
        self.name = label or f"priced:{inner.name}"

    def _cost(self, pt: int, ct: int) -> float:
        return pt / 1e6 * self.price_in + ct / 1e6 * self.price_out

    def generate(self, prompt: str, meta: Optional[Dict] = None) -> GenerationResult:
        res = self.inner.generate(prompt, meta)
        res.cost = self._cost(res.prompt_tokens, res.completion_tokens)
        return res


class RoutingClient(LLMClient):
    name = "router"

    def __init__(self, cheap: LLMClient, expensive: LLMClient,
                 router: Callable[[str, str], str]):
        self.cheap = cheap
        self.expensive = expensive
        self.router = router

    def generate(self, prompt: str, meta: Optional[Dict] = None) -> GenerationResult:
        meta = meta or {}
        choice = self.router(meta.get("question", ""), meta.get("context", ""))
        client = self.cheap if choice == "cheap" else self.expensive
        res = client.generate(prompt, meta)
        res.info = {**(res.info or {}), "route": choice}
        return res


class CascadeClient(LLMClient):
    """Try cheap; escalate to expensive only if the cheap answer fails accept()."""

    name = "cascade"

    def __init__(self, cheap: LLMClient, expensive: LLMClient,
                 accept: Callable[[str], bool]):
        self.cheap = cheap
        self.expensive = expensive
        self.accept = accept

    def generate(self, prompt: str, meta: Optional[Dict] = None) -> GenerationResult:
        r = self.cheap.generate(prompt, meta)
        if self.accept(r.text):
            r.info = {**(r.info or {}), "route": "cheap"}
            return r
        # Escalate. You already paid for the cheap call — count both.
        r2 = self.expensive.generate(prompt, meta)
        return GenerationResult(
            r2.text,
            prompt_tokens=r.prompt_tokens + r2.prompt_tokens,
            completion_tokens=r.completion_tokens + r2.completion_tokens,
            cost=r.cost + r2.cost,
            info={"route": "escalated"},
        )


# ---------------------------------------------------------------------------
# Routers — (question, context) -> "cheap" | "expensive"
# ---------------------------------------------------------------------------

def length_router(max_context_chars: int = 600) -> Callable[[str, str], str]:
    """Heuristic: short context is probably easy -> cheap; long/complex -> expensive."""
    def route(question: str, context: str) -> str:
        return "cheap" if len(context) <= max_context_chars else "expensive"
    return route


def always(choice: str) -> Callable[[str, str], str]:
    def route(question: str, context: str) -> str:
        return choice
    return route


# ---------------------------------------------------------------------------
# Accept predicates for CascadeClient
# ---------------------------------------------------------------------------

def accept_if_nonempty_short(max_words: int = 8) -> Callable[[str], bool]:
    """Keep the cheap answer if it's a confident-looking short span, else escalate.

    A real system would use a verifier or a confidence signal; this is a simple,
    measurable stand-in so the cascade mechanics can be evaluated.
    """
    hedges = ("i don't know", "not sure", "cannot", "unclear", "as an ai")

    def accept(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        low = t.lower()
        if any(h in low for h in hedges):
            return False
        return len(t.split()) <= max_words
    return accept