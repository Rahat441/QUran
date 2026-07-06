"""Load and normalize Quran datasets without binding to one provider's schema."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from quran_assistant.models import Verse

ALIASES: dict[str, tuple[str, ...]] = {
    "surah_number": ("surah_number", "surah", "chapter", "chapter_number"),
    "surah_name": ("surah_name", "chapter_name", "name"),
    "ayah_number": ("ayah_number", "ayah", "verse", "verse_number"),
    "arabic": ("arabic", "text_ar", "arabic_text"),
    "translation": ("translation", "translation_en", "english", "text", "text_en"),
    "transliteration": ("transliteration", "transliteration_en"),
    "juz": ("juz", "juz_number"),
    "revelation_type": ("revelation_type", "revelation", "type"),
}


def _first(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def normalize_record(record: dict[str, Any]) -> Verse:
    """Normalize common Quran dataset field names into a Verse."""

    normalized = {
        field: value
        for field, aliases in ALIASES.items()
        if (value := _first(record, aliases)) is not None
    }
    return Verse.model_validate(normalized)


def _records_from_json(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("verses", payload.get("data"))
    if not isinstance(payload, list):
        raise ValueError("JSON dataset must be an array or contain a 'verses'/'data' array")
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Every verse must be a JSON object")
        yield item


def _records_from_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            yield item


def load_verses(path: Path) -> list[Verse]:
    """Load a UTF-8 JSON or JSONL Quran dataset."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records = (
        _records_from_jsonl(path) if path.suffix.lower() == ".jsonl" else _records_from_json(path)
    )
    verses = [normalize_record(record) for record in records]
    if not verses:
        raise ValueError("Dataset contains no verses")
    references = [verse.reference for verse in verses]
    if len(references) != len(set(references)):
        raise ValueError("Dataset contains duplicate surah:ayah references")
    return verses
