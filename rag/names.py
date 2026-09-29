"""Names that appear all over the corpus: the company, the product, the contract.

A question such as "Who is the CEO of Larkspur Home Robotics?" looks relevant to
half the chunks just because the company name is everywhere, so the relevance
gate lets it through even when no chunk says who the CEO is. A name that shows
up in a quarter of the chunks tells us nothing about whether the documents hold
the fact being asked for, so the gate scores the question a second time with
those names replaced by "it" (see RAGPipeline.ask).
"""
from __future__ import annotations

import re
from collections import Counter

from .bm25 import tokenize

# Capitalised multi-word names ("Larkspur Home Robotics") and product codes ("LR-200").
NAME_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{1,4}-?\d{2,4}[A-Za-z]?)\b")

# Words that only point at the name itself ("What is the LR-200?", "Tell me about X").
# If nothing else is left after masking, the question is about the name and is not masked.
ABOUT = set("tell me about describe explain overview information info know it its".split())


def frequent_names(texts: list[str], min_share: float = 0.25, min_count: int = 3) -> list[str]:
    """Names found in at least `min_share` of the texts (and in `min_count` of them), longest first."""
    if not texts:
        return []
    df = Counter()
    for t in texts:
        df.update(set(NAME_RE.findall(t)))
    n = len(texts)
    names = [name for name, f in df.items() if f >= min_count and f / n >= min_share]
    return sorted(names, key=len, reverse=True)


def mask_names(question: str, names: list[str]) -> str:
    """Replace frequent names with "it" (or "its" for possessives); unchanged if nothing else would be left."""
    masked = question
    for name in names:
        pattern = re.compile(r"(?:\bthe\s+)?" + re.escape(name) + r"(?P<pos>'s|’s)?(?![\w-])", re.I)
        masked = pattern.sub(lambda m: "its" if m.group("pos") else "it", masked)
    if masked == question:
        return question
    if not [t for t in tokenize(masked) if t not in ABOUT]:
        return question  # the question is about the name itself
    return masked
