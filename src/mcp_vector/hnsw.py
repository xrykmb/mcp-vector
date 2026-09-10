from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .embed import cosine


def _distance(a: list[float], b: list[float]) -> float:
    return 1.0 - cosine(a, b)


@dataclass
class HNSW:
    """Simplified Hierarchical NSW (Malkov & Yashunin): greedy descend + ef search.

    Neighbor pick is "M nearest" (not the paper's extended heuristic).
    """

    dim: int
    m: int = 8
    ef: int = 16
    ml: float = 1 / math.log(2)
    rng: random.Random = field(default_factory=random.Random)
    vectors: dict[int, list[float]] = field(default_factory=dict)
    # neighbors[node][layer] = list of node ids
    neighbors: dict[int, list[list[int]]] = field(default_factory=dict)
    enter: int | None = None
    max_layer: int = 0
    _next_id: int = 0

    def insert(self, vector: list[float]) -> int:
        node = self._next_id
        self._next_id += 1
        level = min(int(-math.log(self.rng.random()) * self.ml), 8)
        self.vectors[node] = vector
        self.neighbors[node] = [[] for _ in range(level + 1)]
        if self.enter is None:
            self.enter = node
            self.max_layer = level
            return node
        curr = self.enter
        for layer in range(self.max_layer, level, -1):
            curr = self._greedy(curr, vector, layer)
        for layer in range(min(level, self.max_layer), -1, -1):
            candidates = self._search_layer(curr, vector, layer, self.ef)
            chosen = [n for n, _ in candidates[: self.m]]
            self.neighbors[node][layer] = chosen
            for other in chosen:
                while len(self.neighbors[other]) <= layer:
                    self.neighbors[other].append([])
                links = self.neighbors[other][layer]
                if node not in links:
                    links.append(node)
                    if len(links) > self.m:
                        links.sort(key=lambda n: _distance(self.vectors[n], self.vectors[other]))
                        del links[self.m :]
            if candidates:
                curr = candidates[0][0]
        if level > self.max_layer:
            self.enter = node
            self.max_layer = level
        return node

    def search(self, vector: list[float], k: int) -> list[tuple[int, float]]:
        if self.enter is None:
            return []
        curr = self.enter
        for layer in range(self.max_layer, 0, -1):
            curr = self._greedy(curr, vector, layer)
        ranked = self._search_layer(curr, vector, 0, max(self.ef, k))
        return [(n, cosine(vector, self.vectors[n])) for n, _ in ranked[:k]]

    def _greedy(self, start: int, query: list[float], layer: int) -> int:
        curr = start
        while True:
            best = curr
            best_d = _distance(query, self.vectors[curr])
            for nb in self._layer(curr, layer):
                d = _distance(query, self.vectors[nb])
                if d < best_d:
                    best_d = d
                    best = nb
            if best == curr:
                return curr
            curr = best

    def _search_layer(
        self, start: int, query: list[float], layer: int, ef: int
    ) -> list[tuple[int, float]]:
        visited = {start}
        d0 = _distance(query, self.vectors[start])
        candidates = [(d0, start)]
        w = [(d0, start)]
        while candidates:
            candidates.sort()
            dist, node = candidates.pop(0)
            if w and dist > w[-1][0]:
                break
            for nb in self._layer(node, layer):
                if nb in visited:
                    continue
                visited.add(nb)
                d = _distance(query, self.vectors[nb])
                if len(w) < ef or d < w[-1][0]:
                    candidates.append((d, nb))
                    w.append((d, nb))
                    w.sort()
                    if len(w) > ef:
                        w.pop()
        w.sort()
        return [(n, d) for d, n in w]

    def _layer(self, node: int, layer: int) -> list[int]:
        layers = self.neighbors.get(node) or []
        if layer >= len(layers):
            return []
        return layers[layer]
