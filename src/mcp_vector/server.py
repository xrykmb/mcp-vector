from __future__ import annotations

import json
import os
from pathlib import Path

from .store import VectorStore

INDEX = Path(os.environ.get("MCP_VECTOR_INDEX", ".mcp-vector/index.json"))


def _load() -> VectorStore:
    if INDEX.exists():
        return VectorStore.load(INDEX)
    return VectorStore()


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit('Install MCP extra: python -m pip install -e ".[mcp]"') from exc

    mcp = FastMCP("mcp-vector")

    @mcp.tool()
    def upsert_document(doc_id: str, text: str, metadata_json: str = "{}") -> str:
        """Insert a text chunk into the HNSW index."""
        meta = json.loads(metadata_json) if metadata_json else {}
        store = _load()
        store.upsert(doc_id, text, {str(k): str(v) for k, v in meta.items()})
        store.save(INDEX)
        return f"upserted {doc_id}"

    @mcp.tool()
    def search(query: str, k: int = 5, where_json: str = "{}") -> str:
        """ANN search; where_json is an exact-match metadata filter."""
        where = json.loads(where_json) if where_json else {}
        where_s = {str(k): str(v) for k, v in where.items()} if where else None
        hits = _load().search(query, k=k, where=where_s)
        return json.dumps(
            [
                {"id": rec.doc_id, "score": round(score, 4), "text": rec.text, "metadata": rec.metadata}
                for rec, score in hits
            ],
            ensure_ascii=False,
        )

    @mcp.tool()
    def index_stats() -> str:
        """Return document count and index path."""
        store = _load()
        return json.dumps({"documents": len(store.records), "index": str(INDEX)})

    mcp.run()


if __name__ == "__main__":
    main()
