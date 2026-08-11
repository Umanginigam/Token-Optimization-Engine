"""Dataset management for token optimization."""
"""Datasets for retrieval-QA.

`load_sample()` returns a tiny built-in set so the harness runs with no download
and no network. For real numbers (and a real savings story), use `load_squad`
or `load_hotpotqa`, which pull from Hugging Face — HotpotQA especially has long
contexts, i.e. lots of tokens to cut, which is exactly what you want to show off.

Each example is a dict: {question: str, context: str, answers: [str, ...]}.
"""

from typing import List, Dict

SAMPLE: List[Dict] = [
    {
        "question": "What year did the Eiffel Tower officially open?",
        "context": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
            "in Paris, France. It is named after the engineer Gustave Eiffel, whose "
            "company designed and built the tower. Construction began in 1887. The "
            "tower officially opened to the public in 1889 for the World's Fair. It "
            "was the tallest man-made structure in the world for 41 years."
        ),
        "answers": ["1889"],
    },
    {
        "question": "Who wrote the play Romeo and Juliet?",
        "context": (
            "Romeo and Juliet is a tragedy about two young lovers whose deaths "
            "ultimately reconcile their feuding families. It was written early in "
            "the career of William Shakespeare. The play has been highly popular "
            "since its first performance and is among the most frequently staged "
            "of all his works."
        ),
        "answers": ["William Shakespeare", "Shakespeare"],
    },
    {
        "question": "What is the capital of Australia?",
        "context": (
            "Australia is a country comprising the mainland of the Australian "
            "continent. Sydney is its largest city and a major financial center, "
            "and Melbourne is renowned for its culture. However, the capital of "
            "Australia is Canberra, a planned city chosen as a compromise between "
            "the rivals Sydney and Melbourne."
        ),
        "answers": ["Canberra"],
    },
    {
        "question": "What gas do plants absorb from the air during photosynthesis?",
        "context": (
            "Photosynthesis is the process used by plants to convert light energy "
            "into chemical energy. During photosynthesis, plants take in carbon "
            "dioxide from the air and water from the soil. Using sunlight, they "
            "produce glucose and release oxygen as a byproduct."
        ),
        "answers": ["carbon dioxide"],
    },
    {
        "question": "How many continents are there on Earth?",
        "context": (
            "A continent is one of several large landmasses. Geographers and "
            "scientists generally recognize seven continents: Asia, Africa, North "
            "America, South America, Antarctica, Europe, and Australia. Some models "
            "combine Europe and Asia into a single continent called Eurasia."
        ),
        "answers": ["seven", "7"],
    },
    {
        "question": "What is the largest planet in the solar system?",
        "context": (
            "The solar system contains eight planets. The four inner planets are "
            "rocky, while the outer planets are gas and ice giants. Saturn is famous "
            "for its rings, but Jupiter is the largest planet in the solar system, "
            "with a mass more than two and a half times that of all the other "
            "planets combined."
        ),
        "answers": ["Jupiter"],
    },
]


def load_sample() -> List[Dict]:
    """Built-in offline dataset. No network needed."""
    return [dict(ex) for ex in SAMPLE]


def load_squad(n: int = 100, split: str = "validation") -> List[Dict]:
    """Load n examples from SQuAD via Hugging Face `datasets`.

    Requires: pip install datasets
    """
    from datasets import load_dataset
    ds = load_dataset("squad", split=split)
    n = min(n, len(ds))
    out = []
    for ex in ds.select(range(n)):
        out.append(
            {
                "question": ex["question"],
                "context": ex["context"],
                "answers": list(ex["answers"]["text"]) or [""],
            }
        )
    return out


def load_from_env():
    """Pick a dataset via env vars: DATASET=sample|squad|hotpot, N=<count>.

    Returns (dataset, human_label).
    """
    import os
    name = os.environ.get("DATASET", "sample").lower()
    n = int(os.environ.get("N", "100"))
    if name == "sample":
        return load_sample(), "built-in sample"
    if name == "squad":
        return load_squad(n=n), f"SQuAD (first {n})"
    if name in ("hotpot", "hotpotqa"):
        return load_hotpotqa(n=n), f"HotpotQA (first {n})"
    raise ValueError(f"Unknown DATASET: {name!r}")


def load_hotpotqa(n: int = 100, split: str = "validation") -> List[Dict]:
    """Load n examples from HotpotQA (distractor) — long multi-paragraph context.

    Requires: pip install datasets
    """
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    n = min(n, len(ds))
    out = []
    for ex in ds.select(range(n)):
        paragraphs = ex["context"]["sentences"]  # list[list[str]]
        context = "\n".join(" ".join(sents) for sents in paragraphs)
        out.append(
            {
                "question": ex["question"],
                "context": context,
                "answers": [ex["answer"]],
            }
        )
    return out