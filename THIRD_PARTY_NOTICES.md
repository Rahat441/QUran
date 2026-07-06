# Third-Party Data Notices

The application code is licensed separately under the MIT License. Generated Quran data is
not covered by the application's MIT License.

## QuranLab

The `quran-ra fetch-data` command downloads two verse-aligned configurations from the
[QuranLab Quran dataset](https://huggingface.co/datasets/quranlab/quran), pinned to revision
`001a0d41dcc535c0faa76e269d43c6f157df4639`:

- `arabic-uthmani`
- `en-pickthall`

QuranLab records source, license, and attribution metadata for each edition. The converter
copies that provenance into the generated `data/quran.json` file.

## Tanzil Uthmani Quran Text

The Arabic text is the Tanzil Quran Text, Uthmani version 1.1.

> Tanzil Quran Text (Uthmani, v1.1), Copyright (C) 2007-2026 Tanzil Project,
> licensed under Creative Commons Attribution 3.0.

Source: [Tanzil Quran Text](https://tanzil.net/download/)

The text is retained verbatim. Tanzil's terms require attribution, a link to Tanzil, inclusion
of its copyright notice in derived files containing a substantial portion of the text, and no
changes to the Quran text.

## Marmaduke Pickthall English Translation

The English text is Marmaduke Pickthall's *The Meaning of the Glorious Koran* (1930), recorded
by QuranLab as public domain. QuranLab sources its packaged edition through
[fawazahmed0/quran-api](https://github.com/fawazahmed0/quran-api), whose packaging is released
under the Unlicense.

## Python and Runtime Dependencies

The application depends on independently licensed projects including ChromaDB, Ollama's Python
client, Pydantic, Typer, Rich, and PyArrow. Those projects retain their own licenses. Refer to
their installed package metadata and upstream repositories for their terms.

