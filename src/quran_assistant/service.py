"""Application service joining retrieval and grounded generation."""

from quran_assistant.config import Settings
from quran_assistant.embeddings import OllamaEmbedder
from quran_assistant.generation import INSUFFICIENT_EVIDENCE, GroundedGenerator
from quran_assistant.models import SearchResult
from quran_assistant.retrieval import HybridRetriever
from quran_assistant.store import VerseStore


class QuranAssistant:
    """High-level API used by CLI and future interfaces."""

    def __init__(self, settings: Settings) -> None:
        embedder = OllamaEmbedder(settings.embedding_model, settings.ollama_host)
        self.settings = settings
        self.store = VerseStore(settings.chroma_path, settings.collection_name, embedder)
        self.retriever = HybridRetriever(self.store, settings.semantic_weight)
        self.generator = GroundedGenerator(settings.llm_model, settings.ollama_host)

    def search(self, question: str) -> list[SearchResult]:
        """Retrieve evidence using configured limits."""

        return self.retriever.search(
            question,
            top_k=self.settings.top_k,
            candidate_k=self.settings.candidate_k,
        )

    def answer(self, question: str) -> str:
        """Return a grounded answer or the explicit insufficient-evidence message."""

        results = self.search(question)
        if not results or results[0].score < self.settings.minimum_relevance:
            return INSUFFICIENT_EVIDENCE
        return self.generator.answer(question, results)
