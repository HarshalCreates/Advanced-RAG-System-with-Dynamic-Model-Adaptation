from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def fuse_scores(
    dense: List[Tuple[str, float]],
    sparse: List[Tuple[str, float]],
    graph: List[Tuple[str, float]] | None = None,
    w_dense: float = 0.6,
    w_sparse: float = 0.3,
    w_graph: float = 0.1,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for doc_id, s in dense:
        scores[doc_id] = scores.get(doc_id, 0.0) + w_dense * s
    for doc_id, s in sparse:
        scores[doc_id] = scores.get(doc_id, 0.0) + w_sparse * s
    if graph:
        for doc_id, s in graph:
            scores[doc_id] = scores.get(doc_id, 0.0) + w_graph * s
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@dataclass
class FusionRanker:
    w_dense: float = 0.6
    w_sparse: float = 0.3
    w_graph: float = 0.1

    def combine(
        self,
        dense: List[Tuple[str, float]],
        sparse: List[Tuple[str, float]],
        graph: List[Tuple[str, float]] | None = None,
    ) -> List[Tuple[str, float]]:
        return fuse_scores(dense, sparse, graph, self.w_dense, self.w_sparse, self.w_graph)


