"""
scripts/query_cli.py
CLI tool to query the RAG knowledge base from the terminal.

Usage:
    python scripts/query_cli.py --question "What is the summary?"
    python scripts/query_cli.py --question "Explain the methodology" --top-k 8
    python scripts/query_cli.py --question "What are the risks?" --stream
    python scripts/query_cli.py --interactive
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint

console = Console()


def print_result(result, show_sources: bool = True):
    """Pretty-print a QueryResult."""
    console.print(Panel(
        Markdown(result.answer),
        title="[bold green]Answer[/bold green]",
        border_style="green",
    ))

    if show_sources and result.sources:
        table = Table(title="Sources Used", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("File", style="cyan")
        table.add_column("Page", style="magenta", width=6)
        table.add_column("Score", style="green", width=7)
        table.add_column("Excerpt", style="white")

        for i, src in enumerate(result.sources, 1):
            table.add_row(
                str(i),
                src.filename,
                str(src.page) if src.page else "—",
                f"{src.score:.3f}",
                src.content[:120] + "…",
            )
        console.print(table)

    console.print(
        f"\n[dim]⚡ {result.latency_ms:.0f}ms  ·  "
        f"{result.tokens_used} tokens  ·  "
        f"{result.retrieval_mode} retrieval  ·  {result.model}[/dim]"
    )


def interactive_loop(pipeline, top_k: int, no_sources: bool):
    """REPL-style interactive query loop."""
    console.print(Panel(
        "[bold]RAG Interactive Mode[/bold]\nType your question and press Enter.\n"
        "Commands: [cyan]exit[/cyan] / [cyan]quit[/cyan] to quit  ·  "
        "[cyan]sources[/cyan] to list knowledge base",
        border_style="blue",
    ))

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if question.lower() == "sources":
            for src in pipeline.list_sources():
                console.print(f"  📄 {src['filename']}  ({src['chunks']} chunks)")
            continue

        with console.status("[bold green]Thinking…"):
            try:
                result = pipeline.query(question, top_k=top_k)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue

        print_result(result, show_sources=not no_sources)


def main():
    parser = argparse.ArgumentParser(
        description="Query the RAG knowledge base from the terminal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--source-filter",
        type=str,
        default=None,
        help="Limit search to a specific filename",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the answer token-by-token",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't display source chunks",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter interactive REPL mode",
    )
    args = parser.parse_args()

    console.rule("[bold blue]RAG Query CLI")

    from app.rag_pipeline import get_pipeline
    pipeline = get_pipeline()

    if pipeline.count() == 0:
        console.print("[yellow]⚠️  Knowledge base is empty. Run ingest_docs.py first.[/yellow]")
        sys.exit(1)

    console.print(f"[dim]Knowledge base: {pipeline.count()} chunks across {len(pipeline.list_sources())} document(s)[/dim]\n")

    # ── Interactive mode ──────────────────────────────────────────────────────
    if args.interactive or not args.question:
        interactive_loop(pipeline, args.top_k, args.no_sources)
        return

    # ── Single query ──────────────────────────────────────────────────────────
    console.print(f"[bold]Question:[/bold] {args.question}\n")

    if args.stream:
        console.print("[bold green]Answer:[/bold green]")
        for token in pipeline.stream(args.question, top_k=args.top_k, filter_source=args.source_filter):
            console.print(token, end="")
        console.print()
    else:
        with console.status("[bold green]Thinking…"):
            result = pipeline.query(
                args.question,
                top_k=args.top_k,
                filter_source=args.source_filter,
            )
        print_result(result, show_sources=not args.no_sources)


if __name__ == "__main__":
    main()
