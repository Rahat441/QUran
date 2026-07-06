"""Dataset validation tests."""

import json

import pytest

from quran_assistant.loader import load_verses, normalize_record


def test_normalize_common_aliases() -> None:
    verse = normalize_record(
        {
            "chapter": 1,
            "chapter_name": "Example",
            "verse": 1,
            "arabic_text": "نص",
            "translation_en": "Example translation",
        }
    )
    assert verse.reference == "1:1"
    assert verse.translation == "Example translation"


def test_load_rejects_duplicate_references(tmp_path) -> None:
    path = tmp_path / "verses.json"
    record = {
        "surah_number": 1,
        "surah_name": "Example",
        "ayah_number": 1,
        "arabic": "نص",
        "translation": "Example translation",
    }
    path.write_text(json.dumps([record, record]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_verses(path)
