"""Terminal interface for ingestion, retrieval, and questions."""

import logging
from pathlib import Path
from typing import Annotated

import ollama
import typer
from rich.console import Console
from rich.table import Table

from quran_assistant.config import get_settings
from quran_assistant.embeddings import OllamaEmbedder
from quran_assistant.loader import load_verses
from quran_assistant.quranlab import fetch_quranlab
from quran_assistant.service import QuranAssistant
from quran_assistant.store import VerseStore

app = typer.Typer(no_args_is_help=True, help="Local, citation-grounded Quran research.")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _assistant() -> QuranAssistant:
    return QuranAssistant(get_settings())


@app.command()
def fetch_data(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Converted Quran JSON destination"),
    ] = Path("data/quran.json"),
) -> None:
    """Download verified Arabic and public-domain English from QuranLab."""

    try:
        console.print("Downloading QuranLab Arabic (Uthmani) and Pickthall English…")
        count = fetch_quranlab(output)
        console.print(f"[green]Saved {count} validated verses to {output}.[/green]")
    except Exception as exc:
        console.print(f"[red]Dataset download failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def ingest(path: Annotated[Path, typer.Argument(help="Quran JSON or JSONL dataset")]) -> None:
    """Validate, embed, and index every verse."""

    try:
        verses = load_verses(path)
        settings = get_settings()
        store = VerseStore(
            settings.chroma_path,
            settings.collection_name,
            OllamaEmbedder(settings.embedding_model, settings.ollama_host),
        )
        count = store.upsert(verses)
        console.print(f"[green]Indexed {count} verses.[/green]")
    except Exception as exc:
        console.print(f"[red]Ingestion failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def search(
    question: Annotated[str, typer.Argument(help="Concept or question to retrieve")],
) -> None:
    """Retrieve verses without invoking the language model."""

    try:
        results = _assistant().search(question)
    except Exception as exc:
        console.print(f"[red]Search failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    table = Table("Score", "Reference", "Surah", "Translation")
    for result in results:
        verse = result.verse
        table.add_row(f"{result.score:.3f}", verse.reference, verse.surah_name, verse.translation)
    console.print(table)


@app.command()
def ask(question: Annotated[str, typer.Argument(help="Question about the Quran")]) -> None:
    """Retrieve evidence and generate a grounded answer."""

    try:
        console.print(_assistant().answer(question))
    except Exception as exc:
        console.print(f"[red]Request failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def chat() -> None:
    """Start an interactive question loop."""

    assistant = _assistant()
    console.print("Quran Research Assistant. Type 'quit' to leave.")
    while True:
        question = console.input("\n[bold cyan]Ask:[/bold cyan] ").strip()
        if question.casefold() in {"quit", "exit"}:
            return
        if not question:
            continue
        try:
            console.print(assistant.answer(question))
        except Exception as exc:
            console.print(f"[red]Request failed:[/red] {exc}")


@app.command()
def doctor() -> None:
    """Check Ollama, required models, and the verse index."""

    settings = get_settings()
    try:
        client = ollama.Client(host=settings.ollama_host)
        available = {model.model for model in client.list().models if model.model}
        console.print(f"[green]Ollama reachable:[/green] {settings.ollama_host}")
        for model in (settings.embedding_model, settings.llm_model):
            installed = model in available or any(
                name.startswith(f"{model}:") for name in available
            )
            marker = "green" if installed else "yellow"
            status = "available" if installed else "missing (run ollama pull)"
            console.print(f"[{marker}]{model}: {status}[/{marker}]")
        store = VerseStore(
            settings.chroma_path,
            settings.collection_name,
            OllamaEmbedder(settings.embedding_model, settings.ollama_host),
        )
        console.print(f"Indexed verses: {store.count}")
    except Exception as exc:
        console.print(f"[red]Doctor check failed:[/red] {exc}")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
