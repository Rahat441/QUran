"""Validated, evidence-grounded answer generation through local Ollama."""

import logging
import re

import ollama
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from quran_assistant.models import SearchResult, Verse

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE = (
    "I could not find sufficient Quranic evidence to confidently answer this question."
)

REFERENCE_PATTERN = re.compile(r"^[1-9]\d{0,2}:[1-9]\d{0,2}$")
INLINE_REFERENCE_PATTERN = re.compile(r"\[?\b[1-9]\d{0,2}:[1-9]\d{0,2}\b\]?")

SYSTEM_PROMPT = """You are a Quran retrieval research assistant, not an Islamic authority.
Use exclusively the EVIDENCE supplied by the application. Never use prior knowledge.
Return only JSON matching the supplied schema.

For every claim:
- Write a concise AI explanation in the text field.
- Put every supporting reference in the citations list as surah:ayah without brackets.
- Use only references present in EVIDENCE.
- Do not quote or rewrite Arabic or translation text; the application renders Quran text itself.
- Do not place citations inside claim text.

Produce exactly one summary claim, zero to three reasoning claims, and one to three
primary_references from EVIDENCE. If the evidence cannot support a concise answer, set
sufficient_evidence to false and leave all lists empty.
"""


class GroundingValidationError(ValueError):
    """The model output does not obey the retrieved-evidence boundary."""


class GroundedClaim(BaseModel):
    """One AI-authored claim and the retrieved verses that support it."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=800)
    citations: list[str] = Field(min_length=1, max_length=8)

    @field_validator("text")
    @classmethod
    def reject_inline_references(cls, text: str) -> str:
        """Keep references in the machine-validated citation field."""

        if INLINE_REFERENCE_PATTERN.search(text):
            raise ValueError("claim text must not contain verse references")
        return text

    @field_validator("citations")
    @classmethod
    def validate_citation_shape(cls, citations: list[str]) -> list[str]:
        """Require canonical, unique surah:ayah references."""

        if any(not REFERENCE_PATTERN.fullmatch(reference) for reference in citations):
            raise ValueError("citations must use canonical surah:ayah references")
        if len(citations) != len(set(citations)):
            raise ValueError("citations must be unique within a claim")
        return citations


class GroundedDraft(BaseModel):
    """Structured model output before evidence validation and rendering."""

    model_config = ConfigDict(extra="forbid")

    sufficient_evidence: bool
    summary_claims: list[GroundedClaim] = Field(default_factory=list, max_length=1)
    reasoning_claims: list[GroundedClaim] = Field(default_factory=list, max_length=3)
    primary_references: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("primary_references")
    @classmethod
    def validate_primary_reference_shape(cls, references: list[str]) -> list[str]:
        if any(not REFERENCE_PATTERN.fullmatch(reference) for reference in references):
            raise ValueError("primary references must use canonical surah:ayah references")
        if len(references) != len(set(references)):
            raise ValueError("primary references must be unique")
        return references

    @model_validator(mode="after")
    def validate_evidence_decision(self) -> "GroundedDraft":
        if self.sufficient_evidence and not self.summary_claims:
            raise ValueError("sufficient evidence requires at least one summary claim")
        if self.sufficient_evidence and not self.primary_references:
            raise ValueError("sufficient evidence requires at least one primary reference")
        if not self.sufficient_evidence and (
            self.summary_claims or self.reasoning_claims or self.primary_references
        ):
            raise ValueError("insufficient evidence must not include claims or references")
        return self


def format_evidence(results: list[SearchResult]) -> str:
    """Serialize retrieved verses for the model's restricted evidence context."""

    blocks = []
    for result in results:
        verse = result.verse
        blocks.append(
            "\n".join(
                [
                    f"REFERENCE: {verse.reference}",
                    f"SURAH: {verse.surah_name}",
                    f"ARABIC: {verse.arabic}",
                    f"TRANSLATION: {verse.translation}",
                    f"RETRIEVAL_SCORE: {result.score:.3f}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def validate_draft(draft: GroundedDraft, results: list[SearchResult]) -> None:
    """Reject every model reference that was not retrieved by the application."""

    allowed = {result.verse.reference for result in results}
    supplied = set(draft.primary_references)
    for claim in [*draft.summary_claims, *draft.reasoning_claims]:
        supplied.update(claim.citations)
    unknown = supplied - allowed
    if unknown:
        raise GroundingValidationError(
            f"model cited references outside retrieved evidence: {sorted(unknown)}"
        )


def _format_claim(claim: GroundedClaim) -> str:
    text = " ".join(claim.text.split())
    citations = " ".join(f"[{reference}]" for reference in claim.citations)
    return f"- {text} {citations}"


def _format_verse(verse: Verse) -> str:
    """Render Quran evidence directly from trusted stored fields."""

    return "\n".join(
        [
            f"Surah: {verse.surah_name}",
            f"Ayah: {verse.reference}",
            "Arabic (direct Quran text):",
            verse.arabic,
            "Translation (stored edition):",
            verse.translation,
        ]
    )


def provisional_confidence(results: list[SearchResult]) -> str:
    """Return a deterministic placeholder until evaluation-based calibration is added."""

    top_score = results[0].score if results else 0.0
    if top_score >= 0.60:
        return "High"
    if top_score >= 0.35:
        return "Medium"
    return "Low"


def render_answer(question: str, draft: GroundedDraft, results: list[SearchResult]) -> str:
    """Render a fixed answer layout using exact evidence stored by the application."""

    by_reference = {result.verse.reference: result.verse for result in results}
    primary = [by_reference[reference] for reference in draft.primary_references]
    primary_references = set(draft.primary_references)
    additional = [
        result.verse for result in results if result.verse.reference not in primary_references
    ]

    sections = [
        "Question",
        question.strip(),
        "",
        "Summary Answer (AI explanation)",
        "\n".join(_format_claim(claim) for claim in draft.summary_claims),
        "",
        "Primary Evidence (direct Quran text)",
        "\n\n".join(_format_verse(verse) for verse in primary),
        "",
        "Additional Relevant Verses (direct Quran text)",
        "\n\n".join(_format_verse(verse) for verse in additional) or "None.",
        "",
        "Reasoning (AI explanation)",
        "\n".join(_format_claim(claim) for claim in draft.reasoning_claims)
        or "No additional AI reasoning was generated.",
        "",
        "Confidence",
        provisional_confidence(results),
    ]
    return "\n".join(sections)


class GroundedGenerator:
    """Generate validated claims and render exact retrieved Quran evidence."""

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self.client = ollama.Client(host=host)

    def answer(self, question: str, results: list[SearchResult]) -> str:
        """Return a validated grounded answer, failing closed on invalid model output."""

        if not results:
            return INSUFFICIENT_EVIDENCE
        evidence = format_evidence(results)
        prompt = f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence}"
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format=GroundedDraft.model_json_schema(),
            options={"temperature": 0, "num_predict": 600},
        )
        content = str(response["message"]["content"])
        try:
            draft = GroundedDraft.model_validate_json(content)
            if not draft.sufficient_evidence:
                return INSUFFICIENT_EVIDENCE
            validate_draft(draft, results)
        except (ValidationError, GroundingValidationError) as exc:
            logger.warning("Rejected invalid grounded model output: %s", exc)
            return INSUFFICIENT_EVIDENCE
        return render_answer(question, draft, results)
