# Token Optimizer

> **Same answers. Fewer tokens. Measured, not claimed.**

A measurement harness + web console that reduces LLM token usage while tracking
quality at every step. Built from scratch in Python (backend) and vanilla JS
(frontend), zero heavy dependencies.

---

## Why this exists

Every AI company's token bill is the product of three things: how big the prompt
is, how often the model is called, and how expensive the model is per token.
Most "token optimization" tools address only the first — and without measuring
the quality they trade away in the process.

This project attacks all three and refuses to report a savings number without
its quality cost right next to it. The measurement layer **is** the product.

---

## Real numbers (HotpotQA, llama-3.1-8b-instant via Groq)

| Strategy | Tokens saved | F1 change | Verdict |
|---|---|---|---|
| Semantic cache, threshold 0.95 | **76.6 %** | −0.23 pts | `safe · ns` |
| Semantic cache, threshold 0.99 | 68.5 % | +0.11 pts | `safe · ns` |
| Context trim, keep\_ratio 0.40 | 51.3 % | −1.33 pts | `ns` |
| Context trim, keep\_ratio 0.90 | 6.6 % | −2.67 pts | `ns` |

`ns` = quality change is inside the noise band (bootstrap CI spans zero —
not statistically distinguishable from no change). Reproduce any row with the
commands below.

---

## What's inside

```
token-optimizer/
  server.py           ← start here: serves web/ + /api endpoints in one process
  harness/            ← the measurement engine (pure Python)
    trim.py           relevance-based context trimming (TF-IDF or embedding)
    cache.py          semantic caching + realistic traffic generator
    routing.py        model routing + cost model (PricedClient, CascadeClient)
    sweep.py          checkpointed keep_ratio sweep → savings-vs-quality curve
    stats.py          bootstrap confidence intervals + significance verdicts
    bench.py          combined benchmark: baseline / trim / cache / cache+trim
    scoring.py        canonical SQuAD exact-match + token-F1 (validated)
    client.py         LLM clients: mock / Ollama / Groq / OpenAI-compatible
    dataset.py        built-in sample + SQuAD / HotpotQA loaders
  web/                ← the visual console (static HTML/CSS/JS)
    index.html
    css/              tokens → base → layout → components
    js/               trim · format · sample-data · parse · chart · api · demo · main
  run_baseline.py     Phase 1: full-context baseline
  run_trim.py         Phase 2: baseline vs trimmed comparison
  run_sweep.py        Phase 3: sweep keep_ratio, write curve + recommendation
  run_cache.py        Phase 4a: cache threshold sweep on simulated traffic
  run_routing.py      Phase 4b: cheap vs strong vs routed vs cascade on cost+quality
  run_bench.py        Phase 5: combined benchmark with confidence intervals
```

---

## Quick start

### 1 — install

```bash
git clone https://github.com/your-username/token-optimizer
cd token-optimizer
pip install requests                      # only hard requirement
pip install tiktoken datasets             # recommended
pip install sentence-transformers         # optional: embedding relevance
pip install matplotlib                    # optional: curve PNG
```

### 2 — run the web console (no API key needed)

```bash
python server.py
# open http://127.0.0.1:8000
```

Drag the retention dial, watch the token meter drop, and see which sentences
get kept vs dropped in real time. Works fully offline — the trimmer runs in the
browser when the engine isn't connected, and switches to the real Python engine
(real tokenizer, real relevance scoring) when the server is up.

### 3 — run a real benchmark (needs a free Groq key)

```bash
# Phase 1: establish the baseline
DATASET=hotpot N=25 REQUEST_DELAY=3 \
  LLM_PROVIDER=groq LLM_API_KEY=gsk_xxx python run_baseline.py

# Phase 3: sweep keep_ratio → savings-vs-quality curve (checkpointed)
DATASET=hotpot N=25 REQUEST_DELAY=3 \
  LLM_PROVIDER=groq LLM_API_KEY=gsk_xxx python run_sweep.py

# Phase 4: semantic cache threshold sweep
DATASET=hotpot N=100 REQUEST_DELAY=3 RELEVANCE=embedding \
  LLM_PROVIDER=groq LLM_API_KEY=gsk_xxx python run_cache.py

# Phase 5: combined benchmark with confidence intervals
DATASET=hotpot N=40 REQUESTS=160 REQUEST_DELAY=3 RELEVANCE=embedding \
  THRESHOLD=0.95 KEEP_RATIO=0.5 \
  LLM_PROVIDER=groq LLM_API_KEY=gsk_xxx python run_bench.py
```

Results land in `results/`. Refresh the page — the chart auto-loads them.

---

## How it works

### Three optimizers, one measurement loop

Every optimizer is a plugin into the same evaluation harness. The loop is always:

```
dataset → prompt → model → score (EM + F1) → aggregate
```

**Context trimming** (`harness/trim.py`) splits the context into sentences,
scores each one's relevance to the question (TF-IDF cosine or semantic
embeddings), and keeps the top fraction. The `keep_ratio` knob controls
aggressiveness. `annotate()` is the single source of truth shared by the CLI and
the web server — they can never disagree about which sentences survive.

**Semantic caching** (`harness/cache.py`) reuses answers for semantically similar
requests. On real repetitive traffic this is the biggest saver — a cache hit
costs ~0 tokens. The similarity threshold is swept and each threshold's quality
cost is measured with the same loop, so "hit rate went up but so did wrong
answers" shows up immediately as falling F1.

**Model routing** (`harness/routing.py`) sends easy requests to a cheap model and
hard ones to an expensive model. This is really about **cost**, not tokens, so
`PricedClient` attaches a $/1M-token rate and every summary reports total USD
alongside token counts. `CascadeClient` honestly charges for both calls when it
escalates — because you really do pay for the failed cheap attempt.

### The confidence guard (why it matters)

A naive benchmark reported that `keep_ratio=0.40` saved 51% of tokens with only
−1.3 F1 — the best operating point. It was noise: at n=25 examples, each
question is 4 F1 points, and the CI was roughly ±19. The guard in
`harness/stats.py` catches this:

```
trim keep=0.40   saved=51.3%   F1 delta=-1.33 [-9.2,+6.5]   → ns
```

`ns` = not significant. The interval spans zero. Don't ship it.

A real effect looks like this:

```
cache th=0.95    saved=76.6%   F1 delta=-0.23 [-0.9,+0.4]   → ns  ✓ safe
trim  keep=0.80  saved=13.4%   F1 delta=-6.67 [-9.1,-4.2]   → real cost  ✗
```

Every number in the Phase 5 benchmark table carries a 95% paired bootstrap CI
and a plain-language verdict. That's the thing that separates this from a
compressor with a marketing claim.

### The server (frontend + backend in one)

```
browser                   server.py                 harness/
  POST /api/trim   ──────► annotate() + count_tokens()
  GET  /api/results ─────► results/curve.json, bench.json
  GET  /api/health  ─────► embeddings_available()
  (static)         ◄─────  web/  (SimpleHTTPRequestHandler)
```

One origin, no CORS. Open `web/index.html` directly (no server) and it still
works — it falls back to the in-browser trimmer and manual file upload.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` / `ollama` / `groq` / `openai` / `openai_compatible` |
| `LLM_API_KEY` | — | API key for groq / openai |
| `LLM_MODEL` | provider default | model name override |
| `REQUEST_DELAY` | `0` | seconds between calls (set to `3` on free Groq tier) |
| `MAX_RETRIES` | `6` | retry attempts on 429 / 5xx |
| `DATASET` | `sample` | `sample` / `squad` / `hotpot` |
| `N` | `100` | number of dataset examples |
| `RELEVANCE` | `lexical` | `lexical` / `embedding` |
| `KEEP_RATIO` | `0.5` | context fraction to keep (trimmer) |
| `THRESHOLD` | `0.95` | cache similarity threshold |
| `OUTDIR` | `results` | where to write curve.csv / curve.json / bench.json |
| `PORT` | `8000` | server port |

---

## Adding your own optimizer

The harness is built around two seams:

**`context_transform(question, context) -> context`** — for anything that shrinks
the prompt before it's sent. Pass it to `Pipeline(client, context_transform=...)`.

**`LLMClient.generate(prompt, meta) -> GenerationResult`** — for anything that
changes how the model is called (caching, routing, pricing). Wrap an existing
client.

Either way, plug it into `evaluate()` and it's measured identically to everything
else.

---

## Roadmap

- [ ] LLMLingua-2 adapter (`context_transform` drop-in, compares against lexical/embedding on the same sweep)
- [ ] GitHub Pages deploy of the static console with bundled results
- [ ] FastAPI port of `server.py` for production concurrency
- [ ] Cross-encoder reranker as a third relevance scorer option
- [ ] Agent-loop optimizer (the use case where trim + cache compound most)

---

## License

MIT — do whatever you want with it.
