from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class DocumentStore:
    texts: Dict[str, str]
    metas: Dict[str, dict]

    def __init__(self) -> None:
        self.texts = {}
        self.metas = {}

    def upsert(self, ids: List[str], texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        metadatas = metadatas or [{} for _ in ids]
        for i, t, m in zip(ids, texts, metadatas):
            self.texts[i] = t
            self.metas[i] = m or {}

    def get_texts(self, ids: List[str]) -> List[str]:
        return [self.texts[i] for i in ids if i in self.texts]

    def filter_ids(self, filters: dict) -> List[str]:
        if not filters:
            return list(self.texts.keys())
        matched: List[str] = []
        for i, meta in self.metas.items():
            ok = True
            for k, v in filters.items():
                if meta.get(k) != v:
                    ok = False
                    break
            if ok:
                matched.append(i)
        return matched

    def save(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metas": self.metas}, f)

    def load(self, source: Path) -> None:
        if not source.exists():
            return
        data = json.loads(source.read_text(encoding="utf-8"))
        self.texts = data.get("texts", {})
        self.metas = data.get("metas", {})


