"""Hybrid semantic and lexical verse ranking."""

import re

from quran_assistant.models import SearchResult, Verse
from quran_assistant.store import VerseStore

TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_PATTERN.findall(text) if len(token) > 2}


def lexical_similarity(query: str, verse: Verse) -> float:
    """Measure query token coverage in a verse's searchable text."""

    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    document_tokens = _tokens(verse.search_document)
    return len(query_tokens & document_tokens) / len(query_tokens)


class HybridRetriever:
    """Retrieve semantically, then rerank with direct lexical evidence."""

    def __init__(self, store: VerseStore, semantic_weight: float = 0.75) -> None:
        self.store = store
        self.semantic_weight = semantic_weight

    def search(self, query: str, top_k: int, candidate_k: int) -> list[SearchResult]:
        """Return the strongest unique verse matches."""

        if not query.strip():
            raise ValueError("Question cannot be empty")
        matches = self.store.semantic_search(query, max(top_k, candidate_k))
        results = []
        for verse, semantic_score in matches:
            lexical_score = lexical_similarity(query, verse)
            score = (
                self.semantic_weight * semantic_score + (1 - self.semantic_weight) * lexical_score
            )
            results.append(
                SearchResult(
                    verse=verse,
                    score=max(0.0, min(1.0, score)),
                    semantic_score=semantic_score,
                    lexical_score=lexical_score,
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]
