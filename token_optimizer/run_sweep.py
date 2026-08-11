"""Phase 3 entry point — sweep the trimmer and produce the curve + recommendation.

RATE LIMIT SAFE: checkpoints after every ratio. If you hit a 429 crash, just
re-run the exact same command — completed ratios load from the checkpoint and
are skipped. Only the remaining ones call the API.

Usage:
    # Offline plumbing (mock + lexical):
    python run_sweep.py

    # Real model — recommended settings for a free Groq tier:
    DATASET=hotpot N=25 REQUEST_DELAY=3 \\
      LLM_PROVIDER=groq LLM_API_KEY=gsk_xxx python run_sweep.py

    # If you still hit 429 after a crash, just re-run the same command.
    # The checkpoint file (results/checkpoint.jsonl) skips done ratios.
    # To start fresh, delete results/checkpoint.jsonl.

    # Semantic trimmer + custom knobs:
    RELEVANCE=embedding RATIOS=0.9,0.7,0.5,0.3 MAX_F1_DROP=1.5 \\
      REQUEST_DELAY=3 python run_sweep.py

Env vars:
    REQUEST_DELAY   seconds to wait between API calls (default 0; set to 3 for
                    Groq free tier — keeps you under 20 req/min safely)
    MAX_RETRIES     retry attempts on 429/5xx before giving up (default 6)
    RATIOS          comma-separated keep_ratios (default 0.9..0.2)
    MAX_F1_DROP     F1 pts you'll accept losing for the recommendation (default 2.0)
    OUTDIR          where to write results + checkpoint (default results/)
    N               number of dataset examples to use (25 is enough for a real curve)
"""

import os

from harness.client import build_client_from_env
from harness.dataset import load_from_env
from harness.trim import build_relevance_from_env
from harness.sweep import run_sweep, recommend
from harness.report import print_table, ascii_curve, write_csv, write_json, save_plot

DEFAULT_RATIOS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]


def parse_ratios() -> list:
    raw = os.environ.get("RATIOS")
    if not raw:
        return DEFAULT_RATIOS
    return [float(x) for x in raw.split(",") if x.strip()]


def main():
    client     = build_client_from_env()
    dataset, ds_label = load_from_env()
    relevance  = build_relevance_from_env()
    ratios     = parse_ratios()
    max_f1_drop = float(os.environ.get("MAX_F1_DROP", "2.0"))
    outdir     = os.environ.get("OUTDIR", "results")
    delay      = float(os.environ.get("REQUEST_DELAY", "0.0"))

    n_total = (len(ratios) + 1) * len(dataset)  # +1 for baseline
    eta_min = n_total * delay / 60

    print(f"client      : {client.name}")
    print(f"dataset     : {ds_label}  ({len(dataset)} examples)")
    print(f"relevance   : {relevance.name}")
    print(f"ratios      : {ratios}")
    print(f"request_delay: {delay}s   (~{n_total} total API calls, "
          f"ETA ≥ {eta_min:.0f} min)")
    print(f"outdir      : {outdir}/  (checkpoint saves after every ratio)")
    print()

    base, points = run_sweep(client, dataset, relevance, ratios, outdir=outdir)
    rec = recommend(points, max_f1_drop=max_f1_drop)

    print()
    print_table(points)
    print("\n" + ascii_curve(points))

    csv_path  = os.path.join(outdir, "curve.csv")
    json_path = os.path.join(outdir, "curve.json")
    png_path  = os.path.join(outdir, "curve.png")
    write_csv(points, csv_path)
    write_json(base, points, rec, json_path)
    has_png = save_plot(base, points, png_path, recommended=rec,
                        title=f"Savings vs Quality ({relevance.name}, {ds_label})")

    print("\nwrote:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    if has_png:
        print(f"  {png_path}")
    else:
        print("  (curve.png skipped — pip install matplotlib for the chart)")

    line = "=" * 56
    print("\n" + line)
    if rec:
        print(" RECOMMENDED OPERATING POINT")
        print(line)
        print(f"  keep_ratio   : {rec['keep_ratio']:.2f}")
        print(f"  tokens saved : {rec['tokens_saved_pct']:.2f} %")
        print(f"  F1 change    : {rec['f1_delta']:+.2f} pts  (budget: -{max_f1_drop:.1f})")
        print(f"  EM change    : {rec['em_delta']:+.2f} pts")
        print(line)
        print(f"\n  => ship keep_ratio={rec['keep_ratio']:.2f}: "
              f"cut ~{rec['tokens_saved_pct']:.0f}% of tokens "
              f"for {rec['f1_delta']:+.1f} F1.")
    else:
        print(" NO POINT MET THE QUALITY BUDGET")
        print(line)
        print(f"  Even the gentlest trim dropped F1 > {max_f1_drop:.1f} pts.")
        print("  Loosen MAX_F1_DROP, raise keep_ratio, or try RELEVANCE=embedding.")

    if client.name == "mock":
        print(
            "\nNote: 'mock' is offline — savings are real, but the quality axis\n"
            "only becomes meaningful with LLM_PROVIDER=ollama or =groq."
        )


if __name__ == "__main__":
    main()