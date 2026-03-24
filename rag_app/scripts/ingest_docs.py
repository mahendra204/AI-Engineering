"""
scripts/ingest_docs.py
CLI tool to ingest documents into the RAG knowledge base.

Usage:
    python scripts/ingest_docs.py --source data/
    python scripts/ingest_docs.py --source data/report.pdf
    python scripts/ingest_docs.py --source data/ --chunk-size 800 --overlap 150
    python scripts/ingest_docs.py --reset
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="data/",
        help="Path to a file or directory to ingest",
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=None,
        help="Override chunk size (default: from .env)",
    )
    parser.add_argument(
        "--overlap", "-o",
        type=int,
        default=None,
        help="Override chunk overlap (default: from .env)",
    )
    parser.add_argument(
        "--strategy",
        choices=["recursive", "semantic", "sentence"],
        default=None,
        help="Chunking strategy (default: from .env)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="⚠️  Reset the entire vector store before ingesting",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all ingested sources and exit",
    )
    args = parser.parse_args()

    console.rule("[bold blue]RAG Ingestion CLI")

    from app.rag_pipeline import get_pipeline
    pipeline = get_pipeline()

    # ── List sources ──────────────────────────────────────────────────────────
    if args.list:
        sources = pipeline.list_sources()
        if not sources:
            console.print("[yellow]Knowledge base is empty.[/yellow]")
            return

        table = Table(title=f"Knowledge Base — {pipeline.count()} total chunks")
        table.add_column("Filename", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Chunks", justify="right", style="green")
        for src in sources:
            table.add_row(src["filename"], src["file_type"], str(src["chunks"]))
        console.print(table)
        return

    # ── Reset ─────────────────────────────────────────────────────────────────
    if args.reset:
        console.print("[bold red]⚠️  Resetting vector store...[/bold red]")
        pipeline.vs.reset()
        console.print("[green]✅ Vector store cleared.[/green]")

    # ── Ingest ────────────────────────────────────────────────────────────────
    source = Path(args.source)
    if not source.exists():
        console.print(f"[red]Error: Path not found: {source}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Source:[/bold] {source.resolve()}")
    console.print(f"[bold]Chunk size:[/bold] {args.chunk_size or 'default'}")
    console.print(f"[bold]Overlap:[/bold] {args.overlap or 'default'}")
    console.print(f"[bold]Strategy:[/bold] {args.strategy or 'default'}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting documents...", total=None)
        try:
            added = pipeline.ingest(
                source,
                chunk_size=args.chunk_size,
                chunk_overlap=args.overlap,
                chunk_strategy=args.strategy,
            )
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"\n[red]Error during ingestion: {e}[/red]")
            sys.exit(1)

    console.print(f"\n[bold green]✅ Done! Added {added} chunks.[/bold green]")
    console.print(f"[dim]Total chunks in knowledge base: {pipeline.count()}[/dim]")

    # Summary table
    sources = pipeline.list_sources()
    table = Table(title="Updated Knowledge Base")
    table.add_column("Filename", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Chunks", justify="right", style="green")
    for src in sources:
        table.add_row(src["filename"], src["file_type"], str(src["chunks"]))
    console.print(table)


if __name__ == "__main__":
    main()
