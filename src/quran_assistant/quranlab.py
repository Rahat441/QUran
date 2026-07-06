"""Download and convert verse-aligned QuranLab editions."""

import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from quran_assistant.models import Verse

DATASET_URL = "https://huggingface.co/datasets/quranlab/quran"
DATASET_REVISION = "001a0d41dcc535c0faa76e269d43c6f157df4639"
RAW_URL = f"{DATASET_URL}/resolve/{DATASET_REVISION}"
ARABIC_CONFIG = "arabic-uthmani"
ENGLISH_CONFIG = "en-pickthall"
PARQUET_NAME = "train-00000-of-00001.parquet"
EXPECTED_VERSE_COUNT = 6236

PROVENANCE_FIELDS = ("translation_id", "source", "license", "source_url", "attribution")


def _download(config: str, destination: Path) -> None:
    """Download one QuranLab Parquet edition."""

    url = f"{RAW_URL}/{config}/{PARQUET_NAME}"
    request = Request(url, headers={"User-Agent": "quran-research-assistant/0.1"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    """Read Parquet lazily so pyarrow remains an optional data dependency."""

    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise RuntimeError("Data support is missing; run: pip install -e '.[data]'") from exc

    return list(parquet.read_table(path).to_pylist())


def _index_rows(rows: Iterable[dict[str, Any]], edition: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("verse_key", ""))
        if not key:
            raise ValueError(f"{edition} contains a row without verse_key")
        if key in indexed:
            raise ValueError(f"{edition} contains duplicate verse_key {key}")
        indexed[key] = row
    if len(indexed) != EXPECTED_VERSE_COUNT:
        raise ValueError(
            f"{edition} contains {len(indexed)} verses; expected {EXPECTED_VERSE_COUNT}"
        )
    return indexed


def _provenance(rows: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Extract stable edition-level attribution from row metadata."""

    first = next(iter(rows.values()))
    return {field: str(first.get(field, "")) for field in PROVENANCE_FIELDS}


def build_dataset(
    arabic_rows: Iterable[dict[str, Any]], english_rows: Iterable[dict[str, Any]]
) -> dict[str, object]:
    """Join QuranLab editions and validate every resulting Verse."""

    arabic = _index_rows(arabic_rows, ARABIC_CONFIG)
    english = _index_rows(english_rows, ENGLISH_CONFIG)
    if arabic.keys() != english.keys():
        missing_english = sorted(arabic.keys() - english.keys())
        missing_arabic = sorted(english.keys() - arabic.keys())
        raise ValueError(
            "Edition references do not match "
            f"(missing English: {missing_english[:3]}, missing Arabic: {missing_arabic[:3]})"
        )

    ordered = sorted(
        arabic.values(),
        key=lambda row: (int(row["surah"]), int(row["ayah"])),
    )
    verses = []
    for arabic_row in ordered:
        english_row = english[str(arabic_row["verse_key"])]
        verse = Verse(
            surah_number=int(arabic_row["surah"]),
            surah_name=str(arabic_row["surah_name_en"]),
            ayah_number=int(arabic_row["ayah"]),
            arabic=str(arabic_row["text"]),
            translation=str(english_row["text"]),
            transliteration="",
            juz=int(arabic_row["juz"]),
            revelation_type=str(arabic_row["revelation_place"]),
        )
        if verse.reference != arabic_row["verse_key"]:
            raise ValueError(f"Reference mismatch at {arabic_row['verse_key']}")
        verses.append(verse.model_dump())

    return {
        "metadata": {
            "dataset": "quranlab/quran",
            "dataset_url": DATASET_URL,
            "dataset_revision": DATASET_REVISION,
            "verse_count": len(verses),
            "arabic_config": ARABIC_CONFIG,
            "english_config": ENGLISH_CONFIG,
            "arabic_provenance": _provenance(arabic),
            "english_provenance": _provenance(english),
        },
        "verses": verses,
    }


def fetch_quranlab(output: Path) -> int:
    """Download, convert, and atomically write the configured QuranLab editions."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quranlab-") as temporary_directory:
        temporary = Path(temporary_directory)
        arabic_path = temporary / f"{ARABIC_CONFIG}.parquet"
        english_path = temporary / f"{ENGLISH_CONFIG}.parquet"
        _download(ARABIC_CONFIG, arabic_path)
        _download(ENGLISH_CONFIG, english_path)
        dataset = build_dataset(_read_parquet(arabic_path), _read_parquet(english_path))

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary_output = Path(handle.name)
    os.replace(temporary_output, output)
    output.chmod(0o644)
    metadata = dataset["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("Generated dataset metadata is invalid")
    return int(metadata["verse_count"])
