"""Grounded generation validation and deterministic rendering tests."""

import json
from typing import Any

from quran_assistant.generation import INSUFFICIENT_EVIDENCE, GroundedGenerator
from quran_assistant.models import SearchResult, Verse


class FakeClient:
    """Minimal Ollama client returning one configured response."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.last_arguments: dict[str, Any] = {}

    def chat(self, **arguments: Any) -> dict[str, object]:
        self.last_arguments = arguments
        return {"message": {"content": self.content}}


def _result(reference: str, score: float = 0.7) -> SearchResult:
    surah, ayah = (int(part) for part in reference.split(":"))
    verse = Verse(
        surah_number=surah,
        surah_name=f"Surah {surah}",
        ayah_number=ayah,
        arabic=f"Arabic text for {reference}",
        translation=f"Stored translation for {reference}",
    )
    return SearchResult(
        verse=verse,
        score=score,
        semantic_score=score,
        lexical_score=0.5,
    )


def _generator(content: dict[str, object] | str) -> tuple[GroundedGenerator, FakeClient]:
    serialized = json.dumps(content) if isinstance(content, dict) else content
    fake_client = FakeClient(serialized)
    generator = GroundedGenerator("test-model", "http://test")
    generator.client = fake_client  # type: ignore[assignment]
    return generator, fake_client


def test_valid_draft_renders_exact_stored_evidence() -> None:
    generator, client = _generator(
        {
            "sufficient_evidence": True,
            "summary_claims": [{"text": "The evidence discusses mercy.", "citations": ["17:24"]}],
            "reasoning_claims": [
                {"text": "This is an AI explanation.", "citations": ["17:24", "6:133"]}
            ],
            "primary_references": ["17:24"],
        }
    )
    answer = generator.answer(
        "What does the Quran say about mercy?", [_result("17:24"), _result("6:133", 0.6)]
    )

    assert "Arabic text for 17:24" in answer
    assert "Stored translation for 17:24" in answer
    assert "Arabic text for 6:133" in answer
    assert "The evidence discusses mercy. [17:24]" in answer
    assert "Summary Answer (AI explanation)" in answer
    assert "Primary Evidence (direct Quran text)" in answer
    assert isinstance(client.last_arguments["format"], dict)
    assert client.last_arguments["options"] == {"temperature": 0, "num_predict": 600}


def test_unknown_citation_fails_closed() -> None:
    generator, _ = _generator(
        {
            "sufficient_evidence": True,
            "summary_claims": [{"text": "Unsupported claim.", "citations": ["2:255"]}],
            "reasoning_claims": [],
            "primary_references": ["2:255"],
        }
    )

    assert generator.answer("Question", [_result("17:24")]) == INSUFFICIENT_EVIDENCE


def test_malformed_model_output_fails_closed() -> None:
    generator, _ = _generator("not valid JSON")

    assert generator.answer("Question", [_result("17:24")]) == INSUFFICIENT_EVIDENCE


def test_model_can_decline_insufficient_evidence() -> None:
    generator, _ = _generator(
        {
            "sufficient_evidence": False,
            "summary_claims": [],
            "reasoning_claims": [],
            "primary_references": [],
        }
    )

    assert generator.answer("Question", [_result("17:24")]) == INSUFFICIENT_EVIDENCE


def test_inline_reference_in_claim_text_fails_closed() -> None:
    generator, _ = _generator(
        {
            "sufficient_evidence": True,
            "summary_claims": [{"text": "This claim embeds [17:24].", "citations": ["17:24"]}],
            "reasoning_claims": [],
            "primary_references": ["17:24"],
        }
    )

    assert generator.answer("Question", [_result("17:24")]) == INSUFFICIENT_EVIDENCE
