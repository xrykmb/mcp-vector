from __future__ import annotations

import argparse
from pathlib import Path

from .store import VectorStore

DEFAULT_INDEX = Path(".mcp-vector/index.json")


def _store(path: Path) -> VectorStore:
    if path.exists():
        return VectorStore.load(path)
    return VectorStore()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mcp-vector")
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upsert", help="Add or replace a document")
    up.add_argument("--id", required=True)
    up.add_argument("--text", required=True)
    up.add_argument("--meta", action="append", default=[], help="key=value")
    up.add_argument("--index", default=str(DEFAULT_INDEX))
    se = sub.add_parser("search", help="ANN search with optional metadata filter")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=5)
    se.add_argument("--where", action="append", default=[], help="key=value filter")
    se.add_argument("--index", default=str(DEFAULT_INDEX))
    args = parser.parse_args(argv)
    path = Path(args.index)
    store = _store(path)
    if args.command == "upsert":
        meta = _pairs(args.meta)
        store.upsert(args.id, args.text, meta)
        store.save(path)
        print(f"upserted {args.id}")
        return 0
    where = _pairs(args.where) or None
    hits = store.search(args.query, k=args.k, where=where)
    if not hits:
        print("no hits")
        return 1
    for rec, score in hits:
        print(f"{score:.3f}  {rec.doc_id}  {rec.metadata}  {rec.text[:80]}")
    return 0


def _pairs(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        out[key] = value
    return out


if __name__ == "__main__":
    raise SystemExit(main())
