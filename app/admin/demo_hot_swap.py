import time

import httpx


def main() -> None:
    base = "http://localhost:8000/api/admin"
    with httpx.Client() as c:
        r = c.post(f"{base}/hot-swap/embeddings", params={"backend": "sentence-transformers", "model": "all-MiniLM-L6-v2"})
        print("Embeddings swap:", r.json())
        r = c.post(f"{base}/hot-swap/generation", params={"backend": "openai", "model": "gpt-4o"})
        print("Generation swap:", r.json())
        r = c.post(f"{base}/hot-swap/retriever", params={"backend": "faiss"})
        print("Retriever swap:", r.json())


if __name__ == "__main__":
    main()


