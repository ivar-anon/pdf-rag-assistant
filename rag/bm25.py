"""Okapi BM25 over the chunk texts.

Keyword search catches what embeddings miss: exact codes ("E07"), numbers,
names and clause titles. It is small enough to keep in memory and rebuild
whenever the document set changes.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOP = set("a an and are as at be by for from has have how i if in is it its of on or that the this to was what when "
           "which who why will with do does can my our your their there any".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.tokens]
        self.len = [len(t) for t in self.tokens]
        self.avgdl = sum(self.len) / max(len(self.len), 1)
        df = Counter(term for toks in self.tokens for term in set(toks))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for term in q:
                if term in tf:
                    f = tf[term]
                    s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out
