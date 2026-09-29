import pytest

from rag.answer import NOT_FOUND, LLMAnswerer, SYSTEM_PROMPT
from rag.bm25 import BM25

from conftest import FakeLLM, make_rag


def test_bm25_prefers_exact_codes():
    bm = BM25(["E04 cliff sensor blocked", "E07 battery over temperature", "general care of the robot"])
    s = bm.scores("what is E07")
    assert s.index(max(s)) == 1


def test_documents_listing(rag):
    docs = rag.documents()
    assert [d["source"] for d in docs] == ["employee-handbook.pdf", "lr200-technical-manual.pdf", "services-agreement.pdf"]
    assert all(d["pages"] == 3 for d in docs)


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_retrieval_finds_the_error_code_page(rag, method):
    hits = rag.retriever.search("error code E07 battery over temperature", 3, method)
    assert (hits[0].chunk["source"], hits[0].chunk["page"]) == ("lr200-technical-manual.pdf", 2)


def test_filter_by_document(rag):
    manual = next(d["doc_id"] for d in rag.documents() if "manual" in d["source"])
    hits = rag.retriever.search("days", 5, "hybrid", doc_ids=[manual])
    assert hits and all(h.chunk["doc_id"] == manual for h in hits)


def test_extractive_answer_is_quoted_and_cited(rag):
    a = rag.ask("How many unused vacation days can be carried over to the next year?")
    assert a["found"] and a["mode"] == "extractive"
    assert "five unused vacation days" in a["text"]
    assert (a["citations"][0]["source"], a["citations"][0]["page"]) == ("employee-handbook.pdf", 2)


def test_gate_refuses_unrelated_question(rag):
    a = rag.ask("What is the capital of France?")
    assert not a["found"] and a["text"] == NOT_FOUND and a["citations"] == []


def test_removing_a_document(rag):
    manual = next(d["doc_id"] for d in rag.documents() if "manual" in d["source"])
    rag.remove(manual)
    assert "lr200-technical-manual.pdf" not in [d["source"] for d in rag.documents()]
    hits = rag.retriever.search("E07", 5, "hybrid")
    assert all(h.chunk["doc_id"] != manual for h in hits)


def test_reindexing_same_file_does_not_duplicate(rag, sample_pdfs):
    before = sum(d["chunks"] for d in rag.documents())
    rag.add_pdf(sample_pdfs[0])
    assert sum(d["chunks"] for d in rag.documents()) == before


# --- LLM mode, with a fake OpenAI client ------------------------------------------

def llm_rag(sample_pdfs, text):
    fake = FakeLLM(text)
    r = make_rag(answerer=LLMAnswerer("sk-test", "test-model", client=fake))
    for p in sample_pdfs:
        r.add_pdf(p)
    return r, fake


def test_llm_receives_numbered_excerpts_and_rules(sample_pdfs):
    r, fake = llm_rag(sample_pdfs, "Five days can be carried over [1].")
    a = r.ask("How many vacation days carry over?")
    call = fake.calls[0]
    assert call["model"] == "test-model" and call["instructions"] == SYSTEM_PROMPT
    assert "[1] " in call["input"] and "page" in call["input"] and "Question: How many vacation days" in call["input"]
    assert a["found"] and a["citations"][0]["n"] == 1


def test_llm_invalid_citations_are_removed(sample_pdfs):
    r, _ = llm_rag(sample_pdfs, "The answer is 22 days [1][9].")
    a = r.ask("How many vacation days per year?")
    assert "[9]" not in a["text"] and [c["n"] for c in a["citations"]] == [1]


def test_llm_uncited_answer_is_flagged(sample_pdfs):
    r, _ = llm_rag(sample_pdfs, "It is probably 22 days.")
    a = r.ask("How many vacation days per year?")
    assert "did not cite" in a["text"] and a["citations"] == []


def test_llm_not_found_is_normalised(sample_pdfs):
    r, _ = llm_rag(sample_pdfs, "I couldn't find this in the documents.")
    a = r.ask("Who is the CEO of Larkspur Home Robotics?")
    assert not a["found"] and a["text"] == NOT_FOUND


def test_llm_not_called_when_gate_refuses(sample_pdfs):
    r, fake = llm_rag(sample_pdfs, "should not be used [1]")
    r.ask("What is the capital of France?")
    assert fake.calls == []
