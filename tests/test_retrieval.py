from app.retrieval.service import HybridRetrievalService


def test_index_and_search():
    svc = HybridRetrievalService()
    ids = ["doc1__0000", "doc2__0000", "doc3__0000"]
    texts = [
        "FastAPI is a modern web framework for building APIs with Python.",
        "Retrieval augmented generation improves question answering.",
        "Tesseract OCR extracts text from images.",
    ]
    svc.index(ids, texts)
    results = svc.search("What is RAG?", top_k=2)
    assert len(results) == 2
    assert results[0][1] >= results[-1][1]


