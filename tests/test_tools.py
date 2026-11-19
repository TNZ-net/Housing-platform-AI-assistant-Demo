import pytest
from hp import tools
def make_pdf(path, text):
    """Write a tiny one-page PDF by hand, so PDF reading can be tested with no extra library."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf, offsets = b"%PDF-1.4\n", []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))  # a PDF's index (xref) lists where each object starts
        pdf += f"{number} 0 obj\n{obj}\nendobj\n".encode()
    index = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    index += "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    trailer = f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\nstartxref\n{len(pdf)}\n%%EOF"
    path.write_bytes(pdf + (index + trailer).encode())


@pytest.fixture
def docs(tmp_path):
    (tmp_path / "cooking.md").write_text("Boil the pasta for ten minutes in salted water.")
    (tmp_path / "houses.txt").write_text("Coastal houses cost more than inland houses.")
    make_pdf(tmp_path / "report.pdf", "Drift means the incoming data has changed.")
    (tmp_path / "ignore.csv").write_text("a,b\n1,2")  # only .md .txt .pdf are read
    return tmp_path


def test_search_finds_the_right_document(docs):
    assert "[cooking.md p.1]" in tools.search_docs("how long to boil pasta", docs)
    assert "[houses.txt p.1]" in tools.search_docs("which houses cost more", docs)


def test_search_reads_pdfs_with_page_numbers(docs):
    assert "[report.pdf p.1]" in tools.search_docs("what is drift", docs)


def test_search_says_so_when_nothing_matches(docs):
    assert tools.search_docs("quantum entanglement", docs) == "No relevant passages found."


def test_search_with_no_documents_says_so(tmp_path):
    assert tools.search_docs("anything", tmp_path).startswith("No documents found")


@pytest.mark.parametrize("question, fact", [
    ("how many rows does the dataset have", "20,640"),
    ("which census year is the data from", "1990"),
    ("what unit is median income in", "10,000"),
    ("price cap highest value", "500,000"),
    ("what PSI means a large drift", "0.25"),
])
def test_the_project_doc_answers_basic_questions(question, fact):
    assert fact in tools.search_docs(question)


def test_unknown_tool_and_unknown_query_are_errors_not_crashes():
    assert tools.run_tool("delete_everything", {}).startswith("ERROR: unknown tool")
    assert tools.get_stats("DROP TABLE listings").startswith("ERROR: unknown query")


def test_wrong_arguments_come_back_as_an_error_message():
    assert tools.run_tool("search_docs", {"nonsense": 1}).startswith("ERROR: TypeError")


def test_predict_price_formats_the_api_reply(fake_server, monkeypatch):
    url, seen, reply = fake_server
    monkeypatch.setenv("API_URL", url)
    reply["body"] = {"price": 2.5, "price_usd": 250000, "model_version": 3}
    assert tools.predict_price(med_inc=5) == "Predicted price: $250,000 (model v3)"
    assert seen[0]["path"] == "/predict" and seen[0]["body"] == {"med_inc": 5}


def test_predict_price_passes_api_errors_to_the_model(fake_server, monkeypatch):
    url, _, reply = fake_server
    monkeypatch.setenv("API_URL", url)
    reply.update(status=422, body={"detail": "med_inc must be greater than 0"})
    assert tools.predict_price(med_inc=-1) == "ERROR 422: med_inc must be greater than 0"