from rag.ingest import _blocks, chunk_pages, ingest_pdf, read_pdf


def test_tables_are_kept_row_by_row(sample_pdfs):
    manual = next(p for p in sample_pdfs if "manual" in p.name)
    _, pages = read_pdf(manual)
    assert "E07 | Battery over temperature | Move the robot to a cooler area and wait 30 minutes." in pages[1]


def test_page_footers_are_removed(sample_pdfs):
    _, pages = read_pdf(sample_pdfs[0])
    assert all("— page" not in p for p in pages)


def test_title_comes_from_metadata(sample_pdfs):
    handbook = next(p for p in sample_pdfs if "handbook" in p.name)
    assert read_pdf(handbook)[0].startswith("Larkspur Home Robotics")


def test_wrapped_lines_are_rejoined_into_paragraphs():
    raw = "  Employees may work remotely up to\n  three days per week.\n\n  Next paragraph here.\n"
    assert _blocks(raw) == ["Employees may work remotely up to three days per week.", "Next paragraph here."]


def test_chunks_stay_within_a_page_and_respect_size():
    pages = ["\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6)), "Short second page with one paragraph."]
    chunks = chunk_pages(pages, doc_id="d", doc_title="T", source="t.pdf", max_chars=500)
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 500 + 260 for c in chunks)  # one block of overlap at most
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_overlap_repeats_last_block():
    pages = ["\n\n".join(f"Block number {i}. " + "x " * 120 for i in range(3))]
    c = chunk_pages(pages, doc_id="d", doc_title="T", source="t.pdf", max_chars=300)
    assert c[1].text.startswith(c[0].text.split("\n\n")[-1])


def test_every_sample_pdf_yields_chunks(sample_pdfs):
    for p in sample_pdfs:
        chunks = ingest_pdf(p)
        assert chunks and all(ch.text.strip() for ch in chunks)
        assert max(ch.page for ch in chunks) == 3


def test_headings_stay_on_their_own_line():
    pages = ["Acme Handbook\n\n1. About this handbook\n\nThis handbook applies to every employee of Acme Corp and explains the rules."]
    c = chunk_pages(pages, doc_id="d", doc_title="Acme Handbook", source="a.pdf", max_chars=600)
    assert c[0].text.splitlines() == ["Acme Handbook", "1. About this handbook",
                                      "This handbook applies to every employee of Acme Corp and explains the rules."]


def test_heading_without_blank_line_is_split_from_its_paragraph():
    raw = "  2. Term and renewal\n  The Agreement starts on the effective date and runs for twelve months.\n"
    assert _blocks(raw) == ["2. Term and renewal", "The Agreement starts on the effective date and runs for twelve months."]


def test_wrapped_line_before_a_name_is_not_taken_for_a_heading():
    raw = "  Questions about any policy should go to the\n  People Operations team at the help desk.\n"
    assert _blocks(raw) == ["Questions about any policy should go to the People Operations team at the help desk."]
