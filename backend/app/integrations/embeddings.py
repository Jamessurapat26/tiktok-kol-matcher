import re
from typing import Protocol

_TOKEN_RE = re.compile(r"[\w\u0E00-\u0E7F]+")


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN_RE.finditer(text)}


class Embedder(Protocol):
    def similarity(self, brand_text: str, kol_texts: list[str]) -> list[float]:
        """Return one similarity score per KOL text, ordered like the input."""
        ...


class MockEmbedder:
    """MOCK: replace with real implementation (embedding model + cosine sim)."""

    def similarity(self, brand_text: str, kol_texts: list[str]) -> list[float]:
        brand_tokens = _tokens(brand_text)
        scores: list[float] = []
        for kol_text in kol_texts:
            kol_tokens = _tokens(kol_text)
            union = brand_tokens | kol_tokens
            score = len(brand_tokens & kol_tokens) / len(union) if union else 0.0
            scores.append(round(score, 6))
        return scores
