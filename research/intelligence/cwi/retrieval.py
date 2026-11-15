"""Small dependency-free BM25 index over evidence items.

Retrieval is a fallback; most questions are answered through structured tools
(port -> notices / measurements / C-HARM), which are exact rather than fuzzy.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .evidence import Evidence

_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("a an and are as at be by for from has have in is it of on or that the this to was were with what".split())


def tokenize(s: str) -> list[str]:
    toks = [t for t in _TOK.findall(s.lower()) if t not in _STOP]
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks]


class BM25:
    def __init__(self, items: list[Evidence], k1: float = 1.4, b: float = 0.75):
        self.items, self.k1, self.b = items, k1, b
        self.docs = [tokenize(f"{e.title} {e.title} {e.text} {' '.join(e.species)}") for e in items]
        self.tf = [Counter(d) for d in self.docs]
        self.avg = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5, tiers: set[int] | None = None) -> list[tuple[Evidence, float]]:
        q = tokenize(query)
        scored = []
        for e, tf, d in zip(self.items, self.tf, self.docs):
            if tiers and int(e.tier) not in tiers:
                continue
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            if s > 0:
                scored.append((e, s))
        scored.sort(key=lambda x: (-x[1], x[0].id))
        return scored[:k]
