"""Seeker CLI - AI-powered job seeking agent."""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import anthropic
import typer
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

from .cv_parser import load_cv_text, parse_cv, build_search_preferences, prompt_candidate_wishes
from .models import CVProfile, SearchPreferences, JobListing
from .scrapers import scrape_all, ALL_SCRAPERS
from .matcher import rank_jobs, generate_summary
from .interactive import browse

load_dotenv()

app = typer.Typer(
    name="seeker",
    help="AI-powered job seeking agent — finds the best jobs for YOU.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()
logging.basicConfig(level=logging.WARNING)


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            Panel(
                "[bold red]ANTHROPIC_API_KEY not set.[/bold red]\n"
                "Export it or add it to a .env file:\n"
                "[cyan]export ANTHROPIC_API_KEY=sk-ant-...[/cyan]",
                title="Missing API Key",
                border_style="red",
            )
        )
        raise typer.Exit(1)
    return anthropic.Anthropic(api_key=api_key)


@app.command()
def search(
    cv: str = typer.Argument(..., help="Path to your CV (PDF or .txt file)"),
    remote: bool = typer.Option(True, "--remote/--no-remote", "-r", help="Remote-only jobs (default: on)"),
    days: int = typer.Option(14, "--days", "-d", help="Only show jobs posted in the last N days"),
    location: Optional[str] = typer.Option(None, "--location", "-l", help="Target location(s), comma-separated"),
    keywords: Optional[str] = typer.Option(None, "--keywords", "-k", help="Extra keywords, comma-separated"),
    exclude: Optional[str] = typer.Option(None, "--exclude", "-x", help="Keywords to exclude"),
    top: int = typer.Option(20, "--top", "-n", help="Number of top results to show"),
    min_score: float = typer.Option(50.0, "--min-score", "-m", help="Minimum match score (0-100) to include"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Comma-separated scrapers to use"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Export results to JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs"),
):
    """
    Search for the best jobs matching your CV.

    Examples:
      seeker search my_cv.pdf
      seeker search cv.pdf --remote --keywords "python, backend" --top 15
      seeker search cv.pdf --location "Berlin, London" --export results.json
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    client = _get_client()

    # ── Step 1: Parse CV ─────────────────────────────────────────────────────
    with console.status("[bold cyan]Parsing your CV with AI...", spinner="dots"):
        try:
            cv_text = load_cv_text(cv)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

        if not cv_text.strip():
            console.print("[red]Error:[/red] CV appears to be empty.")
            raise typer.Exit(1)

        profile = parse_cv(cv_text, client)

    _print_profile(profile)

    # ── Step 2: Candidate wishes prompt ──────────────────────────────────────
    wishes = prompt_candidate_wishes(profile)

    # ── Step 3: Build preferences ─────────────────────────────────────────────
    user_prefs: dict = {"remote_only": remote, "max_days_old": days, "wishes": wishes}
    if location:
        user_prefs["locations"] = [l.strip() for l in location.split(",")]
    if keywords:
        wishes["target_roles"] = [k.strip() for k in keywords.split(",")]
    if exclude:
        wishes["excluded_keywords"] += [e.strip() for e in exclude.split(",")]

    prefs = build_search_preferences(profile, user_prefs)
    selected_sources = [s.strip() for s in sources.split(",")] if sources else None

    _print_search_config(prefs, selected_sources)

    # ── Step 3: Scrape job boards ─────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scraping job boards...", total=None)
        jobs = scrape_all(profile, prefs, selected_sources)
        progress.update(task, completed=1, total=1)

    console.print(f"\n[bold green]✓[/bold green] Found [bold]{len(jobs)}[/bold] raw listings across all sources.\n")

    if not jobs:
        console.print(
            Panel(
                "No jobs found. Try:\n"
                "• Broadening your keywords\n"
                "• Removing location filters\n"
                "• Checking your internet connection",
                title="[yellow]No Results[/yellow]",
                border_style="yellow",
            )
        )
        raise typer.Exit(0)

    # ── Step 4: Score & rank ──────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Scoring {len(jobs)} jobs with AI (parallel)...", total=None)
        ranked = rank_jobs(jobs, profile, prefs, client, top_n=top, min_score=min_score)

    console.print(f"[bold green]✓[/bold green] Scored and ranked [bold]{len(ranked)}[/bold] top matches.\n")

    if not ranked:
        console.print(Panel(
            "No jobs scored above the threshold.\nTry [cyan]--min-score 30[/cyan] or [cyan]--days 30[/cyan] to widen results.",
            border_style="yellow",
        ))
        raise typer.Exit(0)

    # ── Step 5: AI summary ────────────────────────────────────────────────────
    with console.status("[bold cyan]Generating strategic summary...", spinner="dots"):
        summary = generate_summary(ranked, profile, client)

    console.print(Panel(
        Markdown(summary),
        title="[bold magenta]AI Career Advisor Summary[/bold magenta]",
        border_style="magenta",
        padding=(1, 2),
    ))

    # ── Step 6: Interactive browser ───────────────────────────────────────────
    browse(ranked, profile, client)

    # ── Step 7: Export ────────────────────────────────────────────────────────
    if export:
        _export_results(ranked, profile, export)
        console.print(f"\n[bold green]✓[/bold green] Exported {len(ranked)} results to [cyan]{export}[/cyan]")


@app.command()
def sources():
    """List all available job board scrapers."""
    table = Table(title="Available Job Board Scrapers", box=box.ROUNDED)
    table.add_column("Name", style="bold cyan")
    table.add_column("Type")
    table.add_column("Description")

    info = {
        "LinkedIn": ("HTML scraper", "LinkedIn public job search (no login, remote filter)"),
        "RemoteOK": ("JSON API", "Remote jobs via RemoteOK.com API"),
        "WeWorkRemotely": ("RSS Feeds", "Remote jobs via RSS (6 categories)"),
        "Remotive": ("JSON API", "Remote jobs via Remotive.com API"),
        "Indeed": ("RSS Feed", "Indeed remote jobs sorted by date"),
        "Jobicy": ("JSON API", "Remote jobs via Jobicy.com API"),
        "HackerNews": ("HN API", "Latest 'Who is Hiring?' Hacker News thread"),
        "ArbeitNow": ("JSON API", "Remote + EU tech jobs via ArbeitNow API"),
        "TheMuse": ("JSON API", "Company culture-focused jobs via The Muse API"),
    }
    for scraper in ALL_SCRAPERS:
        typ, desc = info.get(scraper.name, ("Scraper", ""))
        table.add_row(scraper.name, typ, desc)

    console.print(table)


@app.command()
def web(
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
):
    """Launch the Seeker web UI in your browser."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red] Run: [cyan]pip install uvicorn[standard][/cyan]")
        raise typer.Exit(1)

    url = f"http://{host}:{port}"
    console.print(
        Panel(
            f"Starting Seeker Web UI\n\n"
            f"Open [bold cyan link={url}]{url}[/bold cyan link] in your browser.\n\n"
            f"[dim]Press Ctrl+C to stop.[/dim]",
            title="[bold]Seeker Web[/bold]",
            border_style="cyan",
        )
    )
    uvicorn.run("seeker.web.app:app", host=host, port=port, reload=reload)


@app.command()
def demo():
    """Run a quick demo without a real CV."""
    console.print(
        Panel(
            "Running demo mode with a sample Python developer profile...\n"
            "Set [cyan]ANTHROPIC_API_KEY[/cyan] and run:\n\n"
            "  [bold]seeker search your_cv.pdf[/bold]\n\n"
            "Options:\n"
            "  [bold]--remote[/bold]          Remote-only jobs\n"
            "  [bold]--location Berlin[/bold] Filter by location\n"
            "  [bold]--keywords 'ML, LLM'[/bold] Extra search keywords\n"
            "  [bold]--top 30[/bold]          Show top 30 results\n"
            "  [bold]--export out.json[/bold] Save results to JSON",
            title="[bold cyan]Seeker — AI Job Search Agent[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print("\n[bold]Available sources:[/bold]")
    app_sources = typer.main.get_command(app).commands.get("sources")
    for scraper in ALL_SCRAPERS:
        console.print(f"  • {scraper.name}")


# ── Display helpers ───────────────────────────────────────────────────────────

def _print_profile(profile: CVProfile) -> None:
    lines = [
        f"[bold]{profile.name or 'Unknown'}[/bold]  {profile.email}",
        f"[dim]{profile.current_title}[/dim]  |  {profile.years_experience} years exp  |  {profile.location}",
        f"Skills: [cyan]{', '.join(profile.skills[:8])}{'...' if len(profile.skills) > 8 else ''}[/cyan]",
        f"Targets: [green]{', '.join(profile.target_titles)}[/green]",
    ]
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]Candidate Profile[/bold]",
            border_style="blue",
            padding=(0, 1),
        )
    )


def _print_search_config(prefs: SearchPreferences, sources: list[str] | None) -> None:
    parts = []
    if prefs.remote_only:
        parts.append("Remote only")
    if prefs.locations:
        parts.append(f"Locations: {', '.join(prefs.locations)}")
    if prefs.keywords:
        parts.append(f"Keywords: {', '.join(prefs.keywords[:5])}")
    if prefs.excluded_keywords:
        parts.append(f"Excluded: {', '.join(prefs.excluded_keywords)}")
    if sources:
        parts.append(f"Sources: {', '.join(sources)}")
    else:
        parts.append(f"Sources: all ({len(ALL_SCRAPERS)} scrapers)")

    config_str = "\n".join(f"  • {p}" for p in parts)
    console.print(Panel(config_str, title="Search Configuration", border_style="dim", padding=(0, 1)))
    console.print()


def _score_color(score: float) -> str:
    if score >= 80:
        return "bold green"
    if score >= 60:
        return "green"
    if score >= 40:
        return "yellow"
    return "red"


def _print_results(jobs: list[JobListing]) -> None:
    console.print(f"[bold]Top {len(jobs)} Job Matches[/bold]\n")

    for i, job in enumerate(jobs, 1):
        score_style = _score_color(job.match_score)
        score_bar = _make_bar(job.match_score)

        header = (
            f"[{score_style}]{i:02d}. [{job.match_score:.0f}/100][/{score_style}] "
            f"[bold]{job.title}[/bold] @ [cyan]{job.company}[/cyan]"
        )

        details = Text()
        details.append(f"  {score_bar}  ", style=score_style)
        details.append(f"{job.location}", style="dim")
        if job.salary:
            details.append(f"  |  {job.salary}", style="bold green")
        details.append(f"  |  {job.source}", style="dim")
        if job.posted_date:
            details.append(f"  |  {job.posted_date}", style="dim")

        reasons_text = ""
        if job.match_reasons:
            reasons_text = "\n  [green]✓[/green] " + "\n  [green]✓[/green] ".join(job.match_reasons[:2])
        concerns_text = ""
        if job.match_concerns:
            concerns_text = "\n  [yellow]⚠[/yellow] " + job.match_concerns[0]

        url_text = f"\n  [link={job.apply_url}][blue underline]{job.apply_url[:80]}[/blue underline][/link]"

        console.print(header)
        console.print(details)
        if reasons_text:
            console.print(reasons_text)
        if concerns_text:
            console.print(concerns_text)
        console.print(url_text)
        console.print()


def _make_bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _export_results(jobs: list[JobListing], profile: CVProfile, path: str) -> None:
    output = {
        "profile": {
            "name": profile.name,
            "email": profile.email,
            "current_title": profile.current_title,
            "years_experience": profile.years_experience,
            "skills": profile.skills,
            "target_titles": profile.target_titles,
        },
        "results": [
            {
                "rank": i + 1,
                "score": job.match_score,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "source": job.source,
                "salary": job.salary,
                "job_type": job.job_type,
                "posted_date": job.posted_date,
                "url": job.url,
                "apply_url": job.apply_url,
                "match_reasons": job.match_reasons,
                "match_concerns": job.match_concerns,
                "tags": job.tags,
                "description_excerpt": job.description[:300],
            }
            for i, job in enumerate(jobs)
        ],
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    app()
