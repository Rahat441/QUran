"""QuranLab conversion validation tests."""

import pytest

from quran_assistant.quranlab import ARABIC_CONFIG, ENGLISH_CONFIG, build_dataset


def test_build_dataset_rejects_incomplete_editions() -> None:
    with pytest.raises(ValueError, match="expected 6236"):
        build_dataset(
            [{"verse_key": "1:1", "translation_id": ARABIC_CONFIG}],
            [{"verse_key": "1:1", "translation_id": ENGLISH_CONFIG}],
        )


def test_build_dataset_rejects_duplicate_references() -> None:
    row = {"verse_key": "1:1", "translation_id": ARABIC_CONFIG}
    with pytest.raises(ValueError, match="duplicate"):
        build_dataset([row, row], [])
