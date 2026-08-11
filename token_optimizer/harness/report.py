"""Report generation for optimization results."""
"""Phase 3 reporting — the curve in every form you might need.

write_csv / write_json : portable data you can use anywhere.
ascii_curve            : always works, no dependencies.
save_plot              : the PNG you put in a README / pitch (needs matplotlib).
print_table            : the console summary.
"""

import csv
import json
from typing import Dict, List, Optional

COLUMNS = [
    "keep_ratio",
    "avg_total_tokens",
    "tokens_saved_pct",
    "exact_match",
    "f1",
    "em_delta",
    "f1_delta",
]


def print_table(points: List[Dict]) -> None:
    print(f"\n{'keep':>6} | {'avg_tok':>8} | {'saved %':>8} | {'EM':>6} | {'F1':>6} | {'F1 d':>6}")
    print("-" * 56)
    for p in points:
        tag = "  (base)" if p["keep_ratio"] >= 1.0 else ""
        print(
            f"{p['keep_ratio']:>6.2f} | {p['avg_total_tokens']:>8.1f} | "
            f"{p['tokens_saved_pct']:>8.1f} | {p['exact_match']:>6.2f} | "
            f"{p['f1']:>6.2f} | {p['f1_delta']:>+6.2f}{tag}"
        )


def write_csv(points: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for p in points:
            w.writerow({k: round(p[k], 4) for k in COLUMNS})


def write_json(base: Dict, points: List[Dict], recommended: Optional[Dict], path: str) -> None:
    payload = {
        "baseline": {k: round(base[k], 4) for k in ("exact_match", "f1", "avg_total_tokens")},
        "points": [{k: round(p[k], 4) for k in COLUMNS} for p in points],
        "recommended": ({k: round(recommended[k], 4) for k in COLUMNS} if recommended else None),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def ascii_curve(points: List[Dict], width: int = 46, height: int = 12) -> str:
    """A no-dependency scatter: x = tokens saved %, y = F1. Marks each point."""
    xs = [p["tokens_saved_pct"] for p in points]
    ys = [p["f1"] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1

    grid = [[" "] * width for _ in range(height)]
    for p in points:
        gx = int((p["tokens_saved_pct"] - xmin) / (xmax - xmin) * (width - 1))
        gy = int((p["f1"] - ymin) / (ymax - ymin) * (height - 1))
        grid[height - 1 - gy][gx] = "o"

    lines = [f"F1={ymax:6.2f} +" + "".join(row) for row in grid[:1]]
    lines += ["          |" + "".join(row) for row in grid[1:]]
    lines.append(f"F1={ymin:6.2f} +" + "-" * width)
    lines.append(f"           {xmin:.0f}%" + " " * (width - 6) + f"{xmax:.0f}%")
    lines.append("           tokens saved  -->")
    return "\n".join(lines)


def save_plot(base: Dict, points: List[Dict], path: str,
              recommended: Optional[Dict] = None, title: str = "Savings vs Quality") -> bool:
    """Write the curve PNG. Returns False (without raising) if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    xs = [p["tokens_saved_pct"] for p in points]
    ys = [p["f1"] for p in points]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "-o", color="#2563eb", label="trim curve")
    for p in points:
        ax.annotate(f"{p['keep_ratio']:.1f}",
                    (p["tokens_saved_pct"], p["f1"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(base["f1"], ls="--", color="#94a3b8", label="baseline F1")
    if recommended:
        ax.scatter([recommended["tokens_saved_pct"]], [recommended["f1"]],
                   s=240, facecolors="none", edgecolors="#dc2626", linewidths=2,
                   label="recommended", zorder=5)

    ax.set_xlabel("tokens saved (%)")
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True