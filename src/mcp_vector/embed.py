from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
DIM = 64  # small dim keeps the demo index tiny and tests fast


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = _TOKEN.findall(lowered)
    compact = re.sub(r"\s+", "", lowered)
    grams = [compact[i : i + 2] for i in range(max(0, len(compact) - 1))]
    return words + grams


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def hash_embed(text: str, dim: int = DIM) -> list[float]:
    """Deterministic signed hashing so demos run without an embedding API."""
    vec = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[index] += sign
    return l2_normalize(vec)
