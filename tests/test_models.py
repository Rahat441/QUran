"""Domain model tests."""

from quran_assistant.models import Verse


def test_verse_builds_reference_and_document() -> None:
    verse = Verse(
        surah_number=2,
        surah_name="Example",
        ayah_number=3,
        arabic="نص",
        translation="A sample translation",
    )
    assert verse.reference == "2:3"
    assert "A sample translation" in verse.search_document
    assert verse.metadata()["surah_number"] == 2
