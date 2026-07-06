"""Persistent ChromaDB storage for verse-level retrieval."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api.models.Collection import Collection

from quran_assistant.embeddings import OllamaEmbedder
from quran_assistant.models import Verse


class VerseStore:
    """Own verse indexing and vector queries."""

    def __init__(self, path: Path, collection_name: str, embedder: OllamaEmbedder) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Number of indexed verses."""

        return self.collection.count()

    def upsert(self, verses: Sequence[Verse], batch_size: int = 128) -> int:
        """Embed and upsert verses in bounded batches."""

        for start in range(0, len(verses), batch_size):
            batch = verses[start : start + batch_size]
            documents = [verse.search_document for verse in batch]
            self.collection.upsert(
                ids=[verse.reference for verse in batch],
                documents=documents,
                metadatas=[verse.metadata() for verse in batch],
                embeddings=cast(Any, self.embedder.embed(documents)),
            )
        return len(verses)

    def semantic_search(self, query: str, limit: int) -> list[tuple[Verse, float]]:
        """Return verses and cosine similarity scores."""

        if self.count == 0:
            return []
        result = self.collection.query(
            query_embeddings=cast(Any, self.embedder.embed([query])),
            n_results=min(limit, self.count),
            include=["metadatas", "distances"],
        )
        metadata_groups = result["metadatas"]
        distance_groups = result["distances"]
        if not metadata_groups or not distance_groups:
            return []
        metadatas = metadata_groups[0]
        distances = distance_groups[0]
        matches: list[tuple[Verse, float]] = []
        for metadata, distance in zip(metadatas, distances, strict=True):
            if metadata is None:
                continue
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            matches.append((Verse.from_metadata(dict(metadata)), similarity))
        return matches
