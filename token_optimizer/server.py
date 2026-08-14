#!/usr/bin/env python3
"""server.py — the merge point between the web console and the harness.

One process serves BOTH the static frontend and the engine, so they share an
origin (no CORS) and the page calls the same Python code the CLI uses:

    GET  /                served from web/ (index.html, css, js)
    GET  /api/health      is the engine up? are embeddings available?
    POST /api/trim        run the REAL trimmer (real tokenizer + chosen relevance)
    GET  /api/results     latest results/curve.json + results/bench.json

Zero dependencies — standard library only. Run:

    python server.py            # http://127.0.0.1:8000
    PORT=9000 python server.py

For production you'd put this behind a real WSGI/ASGI server or port these three
handlers to FastAPI; the harness calls inside them don't change.
"""

import importlib.util
import json
import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
RESULTS_DIR = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)

from harness.trim import LexicalRelevance, annotate  # noqa: E402
from harness.tokens import count_tokens              # noqa: E402

# Relevance scorers are cached: lexical is cheap; embedding loads a model lazily
# on first use and is reused after that.
_LEXICAL = LexicalRelevance()
_EMBEDDING = None
_EMBEDDING_TRIED = False


def embeddings_available() -> bool:
    """True if sentence-transformers is importable (model loads on first use)."""
    return importlib.util.find_spec("sentence_transformers") is not None


def get_relevance(name: str):
    """Return (scorer, name_actually_used). Falls back to lexical if embedding
    isn't available, so the endpoint never fails just because a model is missing."""
    global _EMBEDDING, _EMBEDDING_TRIED
    if name == "embedding":
        if _EMBEDDING is None and not _EMBEDDING_TRIED:
            _EMBEDDING_TRIED = True
            try:
                from harness.trim import EmbeddingRelevance
                _EMBEDDING = EmbeddingRelevance()
            except Exception:
                _EMBEDDING = None
        if _EMBEDDING is not None:
            return _EMBEDDING, "embedding"
        return _LEXICAL, "lexical"
    return _LEXICAL, "lexical"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=WEB_DIR, **k)

    # ---- helpers ----
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    # ---- routing ----
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json({"status": "ok", "embeddings": embeddings_available()})
        if path == "/api/results":
            return self._json(self._load_results())
        return super().do_GET()  # static files from web/

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/trim":
            return self._handle_trim()
        self.send_error(404, "unknown endpoint")

    # ---- endpoints ----
    def _handle_trim(self):
        try:
            req = self._read_json()
        except Exception:
            return self._json({"error": "invalid JSON"}, 400)

        question = req.get("question", "") or ""
        context = req.get("context", "") or ""
        try:
            keep_ratio = float(req.get("keep_ratio", 0.5))
        except (TypeError, ValueError):
            return self._json({"error": "keep_ratio must be a number"}, 400)
        keep_ratio = min(1.0, max(0.05, keep_ratio))

        scorer, used = get_relevance(req.get("relevance", "lexical"))
        sentences = annotate(question, context, keep_ratio, scorer)
        kept = " ".join(s["text"] for s in sentences if s["keep"])

        before = count_tokens(context)
        after = count_tokens(kept)
        return self._json({
            "sentences": sentences,
            "tokens_before": before,
            "tokens_after": after,
            "tokens_saved": max(0, before - after),
            "saved_pct": (100.0 * (1 - after / before)) if before else 0.0,
            "relevance_used": used,
        })

    def _load_results(self):
        out = {"curve": None, "bench": None}
        for fname, key in (("curve.json", "curve"), ("bench.json", "bench")):
            p = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        out[key] = json.load(f)
                except Exception:
                    pass
        return out

    def log_message(self, fmt, *args):
        # Quiet by default; uncomment to debug.
        # sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))
        pass


def main():
    port = int(os.environ.get("PORT", "8000"))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Token Meter engine → http://127.0.0.1:{port}")
    print(f"  web     : {WEB_DIR}")
    print(f"  results : {RESULTS_DIR}")
    print(f"  embeddings available: {embeddings_available()}")
    print("  Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()