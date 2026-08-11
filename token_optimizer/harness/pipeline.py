"""Pipeline orchestration for token optimization."""
"""The pipeline — the one place optimizers plug in.

A Pipeline = a client + a `context_transform(question, context) -> context`.
For the baseline the transform is the identity (full context). In Phase 2 you
write a relevance-trimming transform and pass it here; NOTHING else in the
harness changes. That is the whole architectural point: optimizers are swappable
functions, measured by the same loop.
"""

from typing import Callable, Dict

from .client import LLMClient

# Asking for the shortest exact answer keeps outputs extractive, which makes
# EM/F1 meaningful and also keeps completion tokens (and cost) down.
PROMPT_TEMPLATE = (
    "Answer the question using ONLY the context below. "
    "Reply with the shortest exact answer (a single word or short phrase). "
    "Do not explain.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


def identity_transform(question: str, context: str) -> str:
    """Baseline: pass the full context through unchanged."""
    return context


class Pipeline:
    def __init__(
        self,
        client: LLMClient,
        context_transform: Callable[[str, str], str] = identity_transform,
        prompt_template: str = PROMPT_TEMPLATE,
    ):
        self.client = client
        self.context_transform = context_transform
        self.prompt_template = prompt_template

    def run_one(self, example: Dict) -> Dict:
        question = example["question"]
        full_context = example["context"]

        context = self.context_transform(question, full_context)
        prompt = self.prompt_template.format(context=context, question=question)

        result = self.client.generate(
            prompt, meta={"question": question, "context": context}
        )

        return {
            "prediction": (result.text or "").strip(),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "cost": result.cost,
            "info": result.info or {},
            "context_chars": len(context),
            "full_context_chars": len(full_context),
        }