"""Strictly grounded answer generation through a local Ollama model."""

import ollama

from quran_assistant.models import SearchResult

INSUFFICIENT_EVIDENCE = (
    "I could not find sufficient Quranic evidence to confidently answer this question."
)

SYSTEM_PROMPT = """You are a Quran retrieval research assistant, not an Islamic authority.
Answer exclusively from the EVIDENCE supplied by the application. Never use prior knowledge.
Never invent, alter, or merge a verse, surah name, citation, or Arabic text.
Every factual claim must cite one or more supplied references in [surah:ayah] format.
Clearly label Quran text separately from AI explanation.
If the evidence does not support an answer, respond exactly with the insufficient-evidence sentence.
Do not add sources or claims that are absent from EVIDENCE.

Use exactly these headings:
Question
Summary Answer
Primary Evidence
Additional Relevant Verses
Reasoning (AI explanation)
Confidence

Confidence must be High, Medium, or Low. It reflects evidence relevance,
not certainty about theology.
"""


def format_evidence(results: list[SearchResult]) -> str:
    """Serialize retrieved verses without exposing hidden context."""

    blocks = []
    for result in results:
        verse = result.verse
        blocks.append(
            "\n".join(
                [
                    f"REFERENCE: [{verse.reference}]",
                    f"SURAH: {verse.surah_name}",
                    f"ARABIC: {verse.arabic}",
                    f"TRANSLATION: {verse.translation}",
                    f"RETRIEVAL_SCORE: {result.score:.3f}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


class GroundedGenerator:
    """Generate an explanation from an explicit, retrieved evidence packet."""

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self.client = ollama.Client(host=host)

    def answer(self, question: str, results: list[SearchResult]) -> str:
        """Ask Ollama to synthesize only the given results."""

        evidence = format_evidence(results)
        prompt = f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence}"
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0},
        )
        return str(response["message"]["content"]).strip()
