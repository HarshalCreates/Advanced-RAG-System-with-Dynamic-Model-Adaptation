from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SparseRetriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.ids: List[str] = []
        self.matrix = None

    def fit(self, ids: List[str], texts: List[str]) -> None:
        self.ids = ids
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, queries: List[str], top_k: int) -> List[List[Tuple[str, float]]]:
        if self.matrix is None:
            return [[] for _ in queries]
        q = self.vectorizer.transform(queries)
        sims = cosine_similarity(q, self.matrix)
        results: List[List[Tuple[str, float]]] = []
        for row in sims:
            idxs = row.argsort()[::-1][:top_k]
            results.append([(self.ids[i], float(row[i])) for i in idxs])
        return results


