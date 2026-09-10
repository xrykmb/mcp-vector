from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embed import hash_embed
from .hnsw import HNSW


@dataclass(frozen=True)
class Record:
    doc_id: str
    text: str
    metadata: dict[str, str]
    node: int


class VectorStore:
    def __init__(self, dim: int = 64) -> None:
        self.hnsw = HNSW(dim=dim, rng=__import__("random").Random(0))
        self.records: dict[str, Record] = {}
        self._node_to_id: dict[int, str] = {}

    def upsert(self, doc_id: str, text: str, metadata: dict[str, str] | None = None) -> None:
        meta = dict(metadata or {})
        vector = hash_embed(text, self.hnsw.dim)
        node = self.hnsw.insert(vector)
        rec = Record(doc_id=doc_id, text=text, metadata=meta, node=node)
        self.records[doc_id] = rec
        self._node_to_id[node] = doc_id

    def search(
        self, query: str, k: int = 5, where: dict[str, str] | None = None
    ) -> list[tuple[Record, float]]:
        vector = hash_embed(query, self.hnsw.dim)
        # Over-fetch so metadata filters can drop neighbors.
        raw = self.hnsw.search(vector, k=max(k * 4, k))
        hits: list[tuple[Record, float]] = []
        for node, score in raw:
            doc_id = self._node_to_id.get(node)
            if not doc_id:
                continue
            rec = self.records[doc_id]
            if where and any(rec.metadata.get(key) != value for key, value in where.items()):
                continue
            hits.append((rec, score))
            if len(hits) >= k:
                break
        return hits

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": [
                {"doc_id": r.doc_id, "text": r.text, "metadata": r.metadata}
                for r in self.records.values()
            ]
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        data = json.loads(path.read_text(encoding="utf-8"))
        store = cls()
        for row in data.get("records") or []:
            store.upsert(str(row["doc_id"]), str(row["text"]), dict(row.get("metadata") or {}))
        return store
