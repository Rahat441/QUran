"""Pure lexical ranking tests that need no model or database."""

from quran_assistant.models import Verse
from quran_assistant.retrieval import lexical_similarity


def test_lexical_similarity_rewards_matching_concepts() -> None:
    verse = Verse(
        surah_number=1,
        surah_name="Example",
        ayah_number=1,
        arabic="نص",
        translation="Practice patience during difficult trials.",
    )
    assert lexical_similarity("patience through trials", verse) > 0.5
    assert lexical_similarity("inheritance law", verse) == 0
