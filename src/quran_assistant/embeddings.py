"""Local embedding adapter backed by Ollama."""

from collections.abc import Sequence

import ollama


class OllamaEmbedder:
    """Generate embeddings without sending text to a remote service."""

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self.client = ollama.Client(host=host)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one or more documents."""

        if not texts:
            return []
        response = self.client.embed(model=self.model, input=list(texts))
        return [list(vector) for vector in response["embeddings"]]
