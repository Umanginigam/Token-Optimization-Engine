"""Token handling and analysis."""
"""Token counting.

Rule: always prefer the token counts the API actually reports (in each client).
This module is only the *fallback* for when an endpoint doesn't return usage
(e.g. the offline MockClient), so your savings numbers are never guesses when
they don't have to be.
"""

_ENC = None


def count_tokens(text: str) -> int:
    """Best-effort local token count.

    Uses tiktoken's cl100k_base if installed (close enough across modern models
    for *relative* comparisons, which is all the harness needs). Otherwise falls
    back to a rough words*1.3 estimate.
    """
    global _ENC
    if not text:
        return 0
    try:
        if _ENC is None:
            import tiktoken  # optional dependency
            _ENC = tiktoken.get_encoding("cl100k_base")
        return len(_ENC.encode(text))
    except Exception:
        # No tiktoken available — rough estimate. Fine for plumbing tests.
        return max(1, int(len(text.split()) * 1.3))