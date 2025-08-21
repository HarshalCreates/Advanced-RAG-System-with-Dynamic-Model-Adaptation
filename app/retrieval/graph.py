from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


class GraphRetriever:
    def __init__(self) -> None:
        self.links: Dict[str, List[str]] = defaultdict(list)

    def add_link(self, src: str, dst: str) -> None:
        self.links[src].append(dst)

    def neighbors(self, node: str) -> List[str]:
        return self.links.get(node, [])

    def rerank(self, base: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        # Simple heuristic boost for nodes with more neighbors
        reranked: List[Tuple[str, float]] = []
        for doc_id, score in base:
            boost = 1.0 + 0.05 * len(self.neighbors(doc_id))
            reranked.append((doc_id, score * boost))
        return sorted(reranked, key=lambda x: x[1], reverse=True)


