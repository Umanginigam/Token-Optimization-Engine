"""Phase 3 — sweep the knob, build the curve, recommend an operating point.

Checkpointing: each completed ratio is written to OUTDIR/checkpoint.jsonl
immediately after it finishes. On re-run, already-completed ratios are loaded
from the checkpoint and skipped — a crash at ratio 0.4 doesn't re-run 0.9/0.8/0.7.
"""

import json
import os
from typing import Dict, List, Optional

from .pipeline import Pipeline, identity_transform
from .runner import evaluate
from .trim import make_trimmer


def _point(keep_ratio: float, summary: Dict, base: Dict) -> Dict:
    base_tok = base["avg_total_tokens"] or 1e-9
    saved = 100.0 * (1 - summary["avg_total_tokens"] / base_tok)
    return {
        "keep_ratio":        keep_ratio,
        "avg_total_tokens":  summary["avg_total_tokens"],
        "tokens_saved_pct":  saved,
        "exact_match":       summary["exact_match"],
        "f1":                summary["f1"],
        "em_delta":          summary["exact_match"] - base["exact_match"],
        "f1_delta":          summary["f1"] - base["f1"],
    }


def _ckpt_path(outdir: str) -> str:
    return os.path.join(outdir, "checkpoint.jsonl")


def _load_checkpoint(outdir: str) -> Dict[str, Dict]:
    """Return {label -> point} for everything already finished."""
    path = _ckpt_path(outdir)
    done: Dict[str, Dict] = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                done[obj["label"]] = obj
    return done


def _save_checkpoint(outdir: str, label: str, point: Dict) -> None:
    os.makedirs(outdir, exist_ok=True)
    with open(_ckpt_path(outdir), "a") as f:
        f.write(json.dumps({"label": label, **point}) + "\n")


def run_sweep(
    client, dataset, relevance, keep_ratios: List[float],
    outdir: str = "results",
) -> (Dict, List[Dict]):
    """Run baseline + trimmer at each keep_ratio. Checkpoints after every ratio.

    Re-running resumes from where it left off — completed ratios are loaded
    from the checkpoint file and not re-evaluated.
    """
    os.makedirs(outdir, exist_ok=True)
    done = _load_checkpoint(outdir)

    # ---- baseline ----
    if "baseline" in done:
        base_point = done["baseline"]
        # Reconstruct the summary dict the rest of the code needs.
        base = {
            "exact_match":       base_point["exact_match"],
            "f1":                base_point["f1"],
            "avg_total_tokens":  base_point["avg_total_tokens"],
        }
        print(f"  [resume] baseline loaded from checkpoint  "
              f"(EM={base['exact_match']:.2f}  F1={base['f1']:.2f}  "
              f"tok={base['avg_total_tokens']:.1f})")
    else:
        print("  running baseline...")
        base_summary, _ = evaluate(
            Pipeline(client, identity_transform), dataset, verbose=False
        )
        base = {
            "exact_match":      base_summary["exact_match"],
            "f1":               base_summary["f1"],
            "avg_total_tokens": base_summary["avg_total_tokens"],
        }
        base_pt = _point(1.0, base_summary, base_summary)
        _save_checkpoint(outdir, "baseline", base_pt)
        print(f"  baseline done  "
              f"(EM={base['exact_match']:.2f}  F1={base['f1']:.2f}  "
              f"tok={base['avg_total_tokens']:.1f})")

    # ---- one ratio at a time ----
    points: List[Dict] = [_point(1.0, base, base)]   # anchor
    ratios_sorted = sorted({r for r in keep_ratios if 0.0 < r < 1.0}, reverse=True)
    total = len(ratios_sorted)

    for i, kr in enumerate(ratios_sorted, 1):
        label = f"kr_{kr:.4f}"
        if label in done:
            pt = {k: v for k, v in done[label].items() if k != "label"}
            points.append(pt)
            print(f"  [{i}/{total}] keep_ratio={kr}  [resumed from checkpoint]  "
                  f"saved={pt['tokens_saved_pct']:.1f}%  F1={pt['f1']:.2f}  "
                  f"F1d={pt['f1_delta']:+.2f}")
            continue

        print(f"  [{i}/{total}] keep_ratio={kr}  running {len(dataset)} examples...")
        trim = make_trimmer(relevance, keep_ratio=kr)
        summary, _ = evaluate(Pipeline(client, trim), dataset, verbose=False)
        pt = _point(kr, summary, base)
        _save_checkpoint(outdir, label, pt)
        points.append(pt)
        print(f"           done  saved={pt['tokens_saved_pct']:.1f}%  "
              f"F1={pt['f1']:.2f}  F1d={pt['f1_delta']:+.2f}")

    points.sort(key=lambda p: p["tokens_saved_pct"])
    return base, points


def recommend(points: List[Dict], max_f1_drop: float = 2.0) -> Optional[Dict]:
    """Most tokens saved while F1 drop <= budget."""
    eligible = [
        p for p in points
        if p["keep_ratio"] < 1.0 and p["f1_delta"] >= -abs(max_f1_drop)
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda p: p["tokens_saved_pct"])