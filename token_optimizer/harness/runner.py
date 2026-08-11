"""Runner for executing optimization experiments."""
"""The runner — loop, score, aggregate, report.

Always reports quality (EM, F1) and cost (tokens) side by side. A savings number
without its quality cost next to it is a number you're not allowed to trust.
"""

from typing import Dict, List, Tuple

from .pipeline import Pipeline
from .scoring import exact_match, f1_score, metric_max_over_ground_truths


def evaluate(pipeline: Pipeline, dataset: List[Dict], verbose: bool = True) -> Tuple[Dict, List[Dict]]:
    rows: List[Dict] = []

    for i, ex in enumerate(dataset):
        out = pipeline.run_one(ex)
        golds = ex["answers"]
        em = metric_max_over_ground_truths(exact_match, out["prediction"], golds)
        f1 = metric_max_over_ground_truths(f1_score, out["prediction"], golds)

        row = {
            **out,
            "question": ex["question"],
            "gold": golds,
            "em": em,
            "f1": f1,
        }
        rows.append(row)

        if verbose:
            mark = "OK " if em == 1.0 else ("~  " if f1 > 0 else "X  ")
            pred = row["prediction"].replace("\n", " ")
            if len(pred) > 60:
                pred = pred[:57] + "..."
            print(
                f"  [{i+1:>2}] {mark} f1={f1:4.2f}  "
                f"tok={out['total_tokens']:>5}  "
                f"pred={pred!r}  gold={golds}"
            )

    n = len(rows) or 1
    summary = {
        "n": len(rows),
        "exact_match": 100.0 * sum(r["em"] for r in rows) / n,
        "f1": 100.0 * sum(r["f1"] for r in rows) / n,
        "avg_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / n,
        "avg_completion_tokens": sum(r["completion_tokens"] for r in rows) / n,
        "avg_total_tokens": sum(r["total_tokens"] for r in rows) / n,
        "total_tokens": sum(r["total_tokens"] for r in rows),
        # Phase 4 extras (0 / empty when not used):
        "total_cost": sum(r.get("cost", 0.0) for r in rows),
        "avg_cost": sum(r.get("cost", 0.0) for r in rows) / n,
        "cache_hit_rate": 100.0 * sum(1 for r in rows if r.get("info", {}).get("cache_hit")) / n,
        "route_mix": _route_mix(rows),
    }
    return summary, rows


def _route_mix(rows: List[Dict]) -> Dict[str, int]:
    mix: Dict[str, int] = {}
    for r in rows:
        route = r.get("info", {}).get("route")
        if route:
            mix[route] = mix.get(route, 0) + 1
    return mix


def print_summary(label: str, summary: Dict) -> None:
    line = "=" * 56
    print("\n" + line)
    print(f" {label}")
    print(line)
    print(f"  examples evaluated   : {summary['n']}")
    print(f"  Exact Match          : {summary['exact_match']:6.2f} %")
    print(f"  F1                   : {summary['f1']:6.2f} %")
    print(f"  avg prompt tokens    : {summary['avg_prompt_tokens']:8.1f}")
    print(f"  avg completion tokens: {summary['avg_completion_tokens']:8.1f}")
    print(f"  avg total tokens     : {summary['avg_total_tokens']:8.1f}")
    print(f"  total tokens (all)   : {summary['total_tokens']}")
    if summary.get("total_cost", 0.0) > 0:
        print(f"  total cost (USD)     : ${summary['total_cost']:.4f}")
        print(f"  avg cost / request   : ${summary['avg_cost']:.6f}")
    if summary.get("cache_hit_rate", 0.0) > 0:
        print(f"  cache hit rate       : {summary['cache_hit_rate']:6.2f} %")
    if summary.get("route_mix"):
        mix = ", ".join(f"{k}={v}" for k, v in summary["route_mix"].items())
        print(f"  route mix            : {mix}")
    print(line)


def compare(baseline: Dict, candidate: Dict, candidate_label: str = "candidate") -> None:
    """Print the row that matters: savings vs quality change. Used from Phase 2 on."""
    tok_saving = 100.0 * (1 - candidate["avg_total_tokens"] / max(baseline["avg_total_tokens"], 1e-9))
    em_delta = candidate["exact_match"] - baseline["exact_match"]
    f1_delta = candidate["f1"] - baseline["f1"]
    line = "-" * 56
    print("\n" + line)
    print(f" SAVINGS vs QUALITY  ({candidate_label} vs baseline)")
    print(line)
    print(f"  tokens saved         : {tok_saving:6.2f} %")
    print(f"  Exact Match change   : {em_delta:+6.2f} pts")
    print(f"  F1 change            : {f1_delta:+6.2f} pts")
    print(line)