from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

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
        self.hnsw = HNSW(dim=dim, rng=random.Random(0))
        self.records: dict[str, Record] = {}
        self._node_to_id: dict[int, str] = {}

    def upsert(self, doc_id: str, text: str, metadata: dict[str, str] | None = None) -> None:
        meta = dict(metadata or {})
        if doc_id in self.records:
            kept = [(r.doc_id, r.text, r.metadata) for r in self.records.values() if r.doc_id != doc_id]
            self._reset()
            for item_id, item_text, item_meta in kept:
                self._insert(item_id, item_text, item_meta)
        self._insert(doc_id, text, meta)

    def search(
        self, query: str, k: int = 5, where: dict[str, str] | None = None
    ) -> list[tuple[Record, float]]:
        vector = hash_embed(query, self.hnsw.dim)
        raw = self.hnsw.search(vector, k=max(k * 4, k))
        hits: list[tuple[Record, float]] = []
        for node, score in raw:
            mapped = self._node_to_id.get(node)
            if not mapped:
                continue
            rec = self.records[mapped]
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

    def _reset(self) -> None:
        dim = self.hnsw.dim
        self.hnsw = HNSW(dim=dim, rng=random.Random(0))
        self.records = {}
        self._node_to_id = {}

    def _insert(self, doc_id: str, text: str, metadata: dict[str, str]) -> None:
        node = self.hnsw.insert(hash_embed(text, self.hnsw.dim))
        rec = Record(doc_id=doc_id, text=text, metadata=metadata, node=node)
        self.records[doc_id] = rec
        self._node_to_id[node] = doc_id
