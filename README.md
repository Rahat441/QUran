# Quran Research Assistant

A local retrieval-augmented generation (RAG) CLI that finds relevant Quran verses and asks an Ollama model to explain only the retrieved evidence. It is a research aid, not an Islamic authority.

## Prerequisites

- Python 3.12 or newer
- [Ollama](https://ollama.com/) running locally
- A Quran dataset in JSON or JSONL format

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[data,dev]'
cp .env.example .env
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

## Dataset format

Download and convert the default license-clean QuranLab editions:

```bash
quran-ra fetch-data
```

This joins QuranLab's Tanzil Uthmani Arabic (CC BY 3.0) with Marmaduke Pickthall's
public-domain English translation and writes full provenance to `data/quran.json`.

The loader accepts either a JSON array, a `{ "verses": [...] }` object, or one JSON object per line. Each verse must contain:

```json
{
  "surah_number": 1,
  "surah_name": "Al-Fatihah",
  "ayah_number": 1,
  "arabic": "...",
  "translation": "...",
  "transliteration": "...",
  "juz": 1,
  "revelation_type": "Meccan"
}
```

Common aliases such as `chapter`, `verse`, `text`, `english`, and `translation_en` are normalized. Translation provenance matters: choose a properly licensed, trusted Quran translation and keep its attribution alongside the dataset.

## Use

```bash
quran-ra doctor
quran-ra ingest data/quran.json
quran-ra search "patience during trials"
quran-ra ask "What does the Quran say about mercy?"
quran-ra chat
```

`ask` performs semantic retrieval and lexical reranking, then asks Ollama for structured claims
based only on the retrieved verses. Citations are checked against the retrieved references, while
Arabic and translation evidence is rendered directly from stored verse data. Invalid model output
fails closed with the insufficient-evidence response.

## Architecture

```text
question -> hybrid retriever -> ChromaDB -> ranked verses -> grounded Ollama prompt -> answer
```

All settings use the `QRA_` prefix and can be placed in `.env`; see [.env.example](.env.example).

## Development

```bash
ruff check .
mypy src
pytest
```

## Licensing

The application code is available under the [MIT License](LICENSE). Quran text and translation
data retain their own terms and attribution; see [Third-Party Data Notices](THIRD_PARTY_NOTICES.md).
