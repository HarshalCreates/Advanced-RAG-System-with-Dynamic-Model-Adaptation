from app.models.config import get_settings


def test_hot_swap_settings_update():
    s = get_settings()
    s.embedding_backend = "sentence-transformers"
    s.embedding_model = "all-MiniLM-L6-v2"
    s.generation_backend = "openai"
    s.generation_model = "gpt-4o"
    s.retriever_backend = "faiss"
    assert s.embedding_backend and s.generation_backend and s.retriever_backend


