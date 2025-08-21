from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Chunk:
    id: str
    text: str
    page: int | None = None
    section_path: List[str] | None = None


class StructureAwareChunker:
    def split(self, text: str) -> List[Chunk]:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [Chunk(id=f"chunk_{i:04d}", text=part) for i, part in enumerate(parts)]

    def sectioned(self, sections: Iterable[tuple[list[str], str]]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for i, (path, text) in enumerate(sections):
            chunks.append(Chunk(id=f"chunk_{i:04d}", text=text, section_path=path))
        return chunks


