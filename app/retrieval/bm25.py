from __future__ import annotations

from typing import List, Tuple

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self) -> None:
        self.tokenized_corpus: List[List[str]] = []
        self.ids: List[str] = []
        self.bm25: BM25Okapi | None = None

    def fit(self, ids: List[str], texts: List[str]) -> None:
        self.ids = ids or []
        self.tokenized_corpus = [t.lower().split() for t in (texts or [])]
        if not self.tokenized_corpus:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, queries: List[str], top_k: int) -> List[List[Tuple[str, float]]]:
        if self.bm25 is None or not self.ids:
            return [[] for _ in queries]
        results: List[List[Tuple[str, float]]] = []
        for q in queries:
            toks = q.lower().split()
            scores = self.bm25.get_scores(toks)
            idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results.append([(self.ids[i], float(scores[i])) for i in idxs])
        return results


