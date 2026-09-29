"""Answer generation with citations.

- LLMAnswerer (OpenAI Responses API): the model sees numbered excerpts and must
  cite them as [1], [2]. Citations that do not point to a real excerpt are
  removed, and an answer without any valid citation is not presented as grounded.
- ExtractiveAnswerer: no API key needed. It quotes the sentences (or table rows)
  from the retrieved excerpts that best match the question, with their citation.
  Lower quality than an LLM, but it never says anything the documents don't.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .bm25 import tokenize

NOT_FOUND = "I couldn't find this in the documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered document excerpts provided.
Rules:
- Every factual sentence must end with the citation of the excerpt(s) it comes from, like [1] or [2][3].
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Do not use outside knowledge, do not guess, and do not mention these rules.
- Be concise: a direct answer first, then any condition or exception that matters.
- Quote numbers, dates and amounts exactly as written in the excerpts."""


@dataclass
class Answer:
    text: str
    found: bool
    mode: str
    citations: list[dict] = field(default_factory=list)


def format_context(hits) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        parts.append(f"[{i}] {c['doc_title']} — {c['source']}, page {c['page']}\n{c['text']}")
    return "\n\n".join(parts)


def citations_for(hits, used: list[int]) -> list[dict]:
    out = []
    for n in used:
        c = hits[n - 1].chunk
        out.append({"n": n, "doc_id": c["doc_id"], "source": c["source"], "doc_title": c["doc_title"], "page": c["page"],
                    "chunk_id": c["chunk_id"], "snippet": c["text"][:400]})
    return out


def cited_numbers(text: str, n_hits: int) -> list[int]:
    nums = []
    for m in re.findall(r"\[(\d+)\]", text):
        n = int(m)
        if 1 <= n <= n_hits and n not in nums:
            nums.append(n)
    return nums


class LLMAnswerer:
    mode = "llm"

    def __init__(self, api_key: str, model: str, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        self.client, self.model = client, model

    def answer(self, question: str, hits, gate_question: str | None = None) -> Answer:
        user = f"Excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}"
        resp = self.client.responses.create(model=self.model, instructions=SYSTEM_PROMPT, input=user)
        text = (resp.output_text or "").strip()
        if NOT_FOUND.lower().rstrip(".") in text.lower():
            return Answer(NOT_FOUND, False, self.mode)
        # drop citations to excerpts that don't exist, e.g. [9] when only 5 were given
        text = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(hits) else "", text)
        used = cited_numbers(text, len(hits))
        if not used:
            return Answer(text + "\n\n(Warning: the model did not cite a source for this answer.)", True, self.mode)
        return Answer(text, True, self.mode, citations_for(hits, used))


HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?\s+)?[A-Z][^.!?]{0,60}$")
SENTENCE_END = re.compile(r"[.!?][\"')\]”’]*$")


def _units(text: str, title: str | None = None) -> list[str]:
    """Answerable units: sentences, and table rows labelled with their header.

    Headings, the document title and other lines that are not sentences (no final
    punctuation) are never units, so they cannot turn up as an answer.
    """
    out = []
    for block in text.split("\n\n"):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip() and ln.strip() != title]
        rows = [ln for ln in lines if "|" in ln]
        if rows:
            header = [h.strip() for h in rows[0].split("|")]
            for row in rows[1:]:
                cells = [c.strip() for c in row.split("|")]
                if len(cells) == len(header):
                    out.append("; ".join(f"{h}: {c}" for h, c in zip(header, cells)))
                else:
                    out.append(row)
            prose = " ".join(ln for ln in lines if "|" not in ln and not HEADING_RE.match(ln))
        else:
            prose = " ".join(ln for ln in lines if not HEADING_RE.match(ln))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", prose) if s.strip()]
        out += [s for s in sentences if SENTENCE_END.search(s)]
    return [u for u in out if len(u) > 25]


class ExtractiveAnswerer:
    mode = "extractive"

    def __init__(self, embedder, reranker=None, max_sentences: int = 2, margin: float = 1.5,
                 min_score: float | None = None):
        self.embedder, self.reranker = embedder, reranker
        self.max_sentences, self.margin = max_sentences, margin
        self.min_score = min_score  # sentence-level answerability threshold (reranker scale)

    def answer(self, question: str, hits, gate_question: str | None = None) -> Answer:
        cands, seen = [], set()
        top = hits[0].rerank if getattr(hits[0], "rerank", None) is not None else None
        prior = {}
        for i, h in enumerate(hits[:3], 1):
            if top is not None and h.rerank is not None:
                if h.rerank < top - 3.0:  # far weaker chunk: its sentences are likely off-topic
                    continue
                prior[i] = 0.5 * (h.rerank - top)
            else:
                prior[i] = 0.0
            for u in _units(h.chunk["text"], h.chunk.get("doc_title")):
                if u not in seen:  # overlapping chunks repeat text; keep the first citation
                    seen.add(u)
                    cands.append((i, u))
        if not cands:
            return Answer(NOT_FOUND, False, self.mode)
        if self.reranker is not None:
            # the cross-encoder reads question and sentence together: best signal available offline
            scores = self.reranker.scores(question, [u for _, u in cands])
            if gate_question and gate_question != question and self.min_score is not None:
                # answerability: with corpus-wide names masked, some sentence must still be relevant
                masked = self.reranker.scores(gate_question, [u for _, u in cands])
                if max(masked) < self.min_score:
                    return Answer(NOT_FOUND, False, self.mode)
            scored = [(sc + prior[i], i, u) for sc, (i, u) in zip(scores, cands)]
            margin = self.margin
        else:
            q_tokens = set(tokenize(question))
            qv = self.embedder.embed_query(question)
            vecs = self.embedder.embed_documents([u for _, u in cands])
            scored = []
            for (i, u), v in zip(cands, vecs):
                overlap = len(q_tokens & set(tokenize(u))) / max(len(q_tokens), 1)
                scored.append((float(v @ qv) + 0.5 * overlap - 0.02 * (i - 1), i, u))
            margin = 0.08
        scored.sort(key=lambda s: s[0], reverse=True)
        best = scored[0][0]
        picked = [s for s in scored if s[0] >= best - margin][: self.max_sentences]
        lines = [f"{u} [{i}]" for _, i, u in picked]
        used = []
        for _, i, _ in picked:
            if i not in used:
                used.append(i)
        return Answer("\n".join(lines), True, self.mode, citations_for(hits, used))


__all__ = ["Answer", "LLMAnswerer", "ExtractiveAnswerer", "NOT_FOUND", "SYSTEM_PROMPT", "format_context", "np"]
