"""Domain models shared by ingestion, retrieval, and generation."""

from pydantic import BaseModel, ConfigDict, Field


class Verse(BaseModel):
    """One Quran verse and its searchable metadata."""

    model_config = ConfigDict(str_strip_whitespace=True)

    surah_number: int = Field(ge=1, le=114)
    surah_name: str = Field(min_length=1)
    ayah_number: int = Field(ge=1)
    arabic: str = Field(min_length=1)
    translation: str = Field(min_length=1)
    transliteration: str = ""
    juz: int | None = Field(default=None, ge=1, le=30)
    revelation_type: str | None = None

    @property
    def reference(self) -> str:
        """Return the canonical surah:ayah citation."""

        return f"{self.surah_number}:{self.ayah_number}"

    @property
    def search_document(self) -> str:
        """Text embedded and searched for this verse."""

        parts = [
            f"Surah {self.surah_name} {self.reference}",
            self.translation,
            self.transliteration,
            self.arabic,
        ]
        return "\n".join(part for part in parts if part)

    def metadata(self) -> dict[str, str | int]:
        """Return Chroma-compatible, non-null metadata."""

        values: dict[str, str | int] = {
            "surah_number": self.surah_number,
            "surah_name": self.surah_name,
            "ayah_number": self.ayah_number,
            "arabic": self.arabic,
            "translation": self.translation,
            "transliteration": self.transliteration,
        }
        if self.juz is not None:
            values["juz"] = self.juz
        if self.revelation_type:
            values["revelation_type"] = self.revelation_type
        return values

    @classmethod
    def from_metadata(cls, metadata: dict[str, object]) -> "Verse":
        """Rehydrate a verse stored in Chroma metadata."""

        return cls.model_validate(metadata)


class SearchResult(BaseModel):
    """A retrieved verse with a normalized relevance score."""

    verse: Verse
    score: float = Field(ge=0, le=1)
    semantic_score: float = Field(ge=0, le=1)
    lexical_score: float = Field(ge=0, le=1)
