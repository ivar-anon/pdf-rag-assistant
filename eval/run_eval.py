"""Measure retrieval and answers on eval/questions.jsonl.

Retrieval: for each method (BM25, dense, hybrid) — hit@1, hit@3, MRR@5, where a
hit means the right document AND the right page.
Answers: the expected facts appear in the answer, and questions the documents
don't cover are refused instead of answered.
Writes eval/results.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import Settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

QUESTIONS = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if line.strip()]


def rank_of(hits, q) -> int | None:
    for i, h in enumerate(hits, 1):
        if h.chunk["source"] == q["source"] and h.chunk["page"] == q["page"]:
            return i
    return None


def main() -> None:
    rag = RAGPipeline(Settings(openai_api_key=None), in_memory=True)
    for pdf in sorted((ROOT / "sample_docs" / "pdfs").glob("*.pdf")):
        rag.add_pdf(pdf)
    answerable = [q for q in QUESTIONS if q["source"]]
    unanswerable = [q for q in QUESTIONS if not q["source"]]

    lines = ["# Evaluation", "",
             f"Corpus: 3 sample PDFs (9 pages). {len(answerable)} answerable questions, phrased differently from the "
             f"documents, and {len(unanswerable)} questions the documents do not cover.", "",
             "## Retrieval (right document and right page)", "",
             "| method | hit@1 | hit@3 | MRR@5 |", "|---|---|---|---|"]
    misses = {}

    def reranked(q, k=5):
        hits = rag.retriever.search(q, rag.s.candidates, "hybrid")
        for h, sc in zip(hits, rag.reranker.scores(q, [h.chunk["text"] for h in hits])):
            h.rerank = sc
        return sorted(hits, key=lambda h: h.rerank, reverse=True)[:k]

    methods = {"bm25": lambda q: rag.retriever.search(q, 5, "bm25"),
               "dense": lambda q: rag.retriever.search(q, 5, "dense"),
               "hybrid (BM25 + dense, RRF)": lambda q: rag.retriever.search(q, 5, "hybrid"),
               "hybrid + reranker (default)": reranked}
    for method, fn in methods.items():
        h1 = h3 = 0
        mrr = 0.0
        for q in answerable:
            r = rank_of(fn(q["q"]), q)
            h1 += r == 1
            h3 += bool(r and r <= 3)
            mrr += 1 / r if r else 0
            if method.endswith("(default)") and r != 1:
                misses[q["q"]] = r
        n = len(answerable)
        lines.append(f"| {method} | {h1/n:.0%} ({h1}/{n}) | {h3/n:.0%} ({h3}/{n}) | {mrr/n:.3f} |")

    # relevance gate: the reranker score of the best chunk, and of the best chunk again with
    # corpus-wide names masked (rag/names.py); the gate uses the lower of the two
    best, masked_best = {}, {}
    for q in QUESTIONS:
        top = reranked(q["q"], rag.s.top_k)
        best[q["q"]] = top[0].rerank
        gq = rag.gate_question(q["q"])
        if gq != q["q"]:
            masked_best[q["q"]] = max(rag.reranker.scores(gq, [h.chunk["text"] for h in top]))
    gate = {qq: min(sc, masked_best.get(qq, sc)) for qq, sc in best.items()}
    lines += ["", "## Relevance gate", "",
              "If the best chunk's reranker score is below the threshold, the answer is \"I couldn't find this in the "
              "documents\" and no model is called. When the question names something that appears all over the "
              "documents (the company, the product), the best chunk is scored again with that name masked, and the "
              "lower score counts: a name found everywhere is not evidence that the fact is there.", "",
              "| threshold | answerable kept | out-of-scope refused |", "|---|---|---|"]
    for t in (-7.0, -6.0, -5.0, -4.0, -3.0, -2.0):
        kept = sum(gate[q["q"]] >= t for q in answerable)
        refused = sum(gate[q["q"]] < t for q in unanswerable)
        mark = " ← default" if abs(t - rag.s.min_rerank) < 1e-9 else ""
        lines.append(f"| {t:.1f}{mark} | {kept}/{len(answerable)} | {refused}/{len(unanswerable)} |")
    lines += ["", "Out-of-scope questions and their gate score (with the name masked, when there is one):", ""]
    lines += [f"- {q['q']} → {best[q['q']]:.2f}" + (f", masked {masked_best[q['q']]:.2f}" if q["q"] in masked_best else "")
              for q in unanswerable]
    lines += ["", f"Lowest gate score of an answerable question: {min(gate[q['q']] for q in answerable):.2f}.", ""]

    # end-to-end answers (extractive mode: no API key)
    correct, cited_right, wrong = 0, 0, []
    for q in answerable:
        a = rag.ask(q["q"])
        ok = a["found"] and all(e.lower() in a["text"].lower() for e in q["expect"])
        correct += ok
        cited_right += any(c["source"] == q["source"] and c["page"] == q["page"] for c in a["citations"])
        if not ok:
            wrong.append((q["q"], a["text"].replace("\n", " ")[:160]))
    refused = sum(not rag.ask(q["q"])["found"] for q in unanswerable)
    lines += ["## Answers (extractive mode, no LLM)", "",
              f"- Expected facts present in the answer: **{correct}/{len(answerable)}**",
              f"- Answer cites the right document and page: **{cited_right}/{len(answerable)}**",
              f"- Out-of-scope questions refused: **{refused}/{len(unanswerable)}**", ""]
    if wrong:
        lines += ["Answers that missed an expected fact:", ""] + [f"- *{q}* → {a}" for q, a in wrong] + [""]
    if misses:
        lines += ["Questions where hybrid retrieval did not put the right page first (rank shown):", ""]
        lines += [f"- {q}: {r if r else 'not in top 5'}" for q, r in misses.items()] + [""]
    lines += ["With an OpenAI key the answer step is generated by the model instead; retrieval, the relevance gate "
              "and citation checking stay the same.", ""]
    (ROOT / "eval" / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))
    rag.close()


if __name__ == "__main__":
    main()
