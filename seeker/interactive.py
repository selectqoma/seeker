"""Interactive job browser — browse results and act on them."""
import json
from datetime import datetime
from pathlib import Path

import anthropic
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from .models import CVProfile, JobListing
from .generator import generate_cover_letter, suggest_cv_adaptations

SAVED_FILE = Path("saved_jobs.json")
console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 24) -> str:
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _score_style(score: float) -> str:
    if score >= 80: return "bold green"
    if score >= 60: return "green"
    if score >= 40: return "yellow"
    return "red"


def _print_job_card(job: JobListing, rank: int) -> None:
    style = _score_style(job.match_score)
    bar = _score_bar(job.match_score)

    scope_color = "green" if job.remote_scope == "Worldwide" else "yellow" if job.remote_scope not in ("", "Unspecified") else "dim"
    lines = [
        f"[bold]{job.title}[/bold]  @  [cyan]{job.company}[/cyan]",
        f"[dim]{job.location}[/dim]  [{scope_color}]({job.remote_scope or 'Remote'})[/{scope_color}]"
        + (f"  |  [bold green]{job.salary}[/bold green]" if job.salary else ""),
        f"[{style}]{bar}  {job.match_score:.0f}/100[/{style}]  "
        f"[dim]{job.source}  {job.posted_date}[/dim]",
    ]
    if job.match_reasons:
        lines.append(f"[green]✓[/green] {job.match_reasons[0]}")
    if job.match_concerns:
        lines.append(f"[yellow]⚠[/yellow]  {job.match_concerns[0]}")
    lines.append(f"[blue underline link={job.apply_url}]{job.apply_url[:80]}[/blue underline link]")

    console.print(Panel(
        "\n".join(lines),
        title=f"[bold dim]#{rank}[/bold dim]",
        border_style=style,
        padding=(0, 1),
    ))


def _print_full_job(job: JobListing, rank: int) -> None:
    style = _score_style(job.match_score)
    console.print(Rule(f"[bold]#{rank} — {job.title} @ {job.company}[/bold]", style=style))
    scope_color = "green" if job.remote_scope == "Worldwide" else "yellow" if job.remote_scope not in ("", "Unspecified") else "dim"
    meta = (
        f"[dim]Location:[/dim] {job.location}  "
        f"[{scope_color}][{job.remote_scope or 'Remote'}][/{scope_color}]  |  "
        f"[dim]Type:[/dim] {job.job_type or 'N/A'}  |  "
        f"[dim]Source:[/dim] {job.source}  |  "
        f"[dim]Posted:[/dim] {job.posted_date or 'unknown'}"
    )
    if job.salary:
        meta += f"  |  [bold green]Salary: {job.salary}[/bold green]"
    console.print(meta)
    if job.tags:
        console.print(f"[dim]Tags:[/dim] {', '.join(job.tags[:8])}")
    console.print(f"[dim]Score:[/dim] [{style}]{_score_bar(job.match_score)} {job.match_score:.0f}/100[/{style}]")
    if job.match_reasons:
        console.print(f"[green]✓ {job.match_reasons[0]}[/green]")
    if job.match_concerns:
        console.print(f"[yellow]⚠  {job.match_concerns[0]}[/yellow]")
    console.print(f"\n[dim]Apply:[/dim] [blue underline link={job.apply_url}]{job.apply_url}[/blue underline link]")
    if job.description:
        console.print("\n[bold dim]Description[/bold dim]")
        console.print(job.description[:800] + ("..." if len(job.description) > 800 else ""))


def _load_saved() -> list[dict]:
    if SAVED_FILE.exists():
        return json.loads(SAVED_FILE.read_text())
    return []


def _save_job(job: JobListing, rank: int) -> None:
    saved = _load_saved()
    entry = {
        "saved_at": datetime.now().isoformat(),
        "rank": rank,
        "score": job.match_score,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "salary": job.salary,
        "source": job.source,
        "posted_date": job.posted_date,
        "url": job.url,
        "apply_url": job.apply_url,
        "match_reasons": job.match_reasons,
        "match_concerns": job.match_concerns,
        "description_excerpt": job.description[:400],
    }
    # Avoid duplicates by URL
    if not any(s["url"] == job.url for s in saved):
        saved.append(entry)
        SAVED_FILE.write_text(json.dumps(saved, indent=2))
        console.print(f"[bold green]✓[/bold green] Saved to [cyan]{SAVED_FILE}[/cyan]  ({len(saved)} total saved)")
    else:
        console.print("[yellow]Already saved.[/yellow]")


def _write_output(filename: str, content: str) -> None:
    Path(filename).write_text(content, encoding="utf-8")
    console.print(f"[bold green]✓[/bold green] Written to [cyan]{filename}[/cyan]")


# ── Action menu ───────────────────────────────────────────────────────────────

def _job_action_menu(job: JobListing, rank: int, profile: CVProfile, client: anthropic.Anthropic) -> None:
    """Show full job details + action menu for a single job."""
    console.print()
    _print_full_job(job, rank)
    console.print()

    while True:
        console.print(
            "[bold]Actions:[/bold]  "
            "[cyan][c][/cyan] Cover letter  "
            "[cyan][a][/cyan] Adapt CV  "
            "[cyan][s][/cyan] Save for later  "
            "[cyan][b][/cyan] Back to list"
        )
        choice = Prompt.ask("Choose", choices=["c", "a", "s", "b"], default="b")

        if choice == "b":
            break

        elif choice == "s":
            _save_job(job, rank)

        elif choice == "c":
            with console.status("[bold cyan]Generating cover letter...", spinner="dots"):
                letter = generate_cover_letter(job, profile, client)
            console.print()
            console.print(Panel(
                Markdown(letter),
                title=f"[bold magenta]Cover Letter — {job.title} @ {job.company}[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            ))
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{job.company}_{job.title}")[:50]
            filename = f"cover_letter_{safe_name}.md"
            save = Prompt.ask(f"Save to file?", choices=["y", "n"], default="y")
            if save == "y":
                _write_output(filename, f"# Cover Letter — {job.title} @ {job.company}\n\n{letter}")

        elif choice == "a":
            with console.status("[bold cyan]Generating CV adaptation suggestions...", spinner="dots"):
                suggestions = suggest_cv_adaptations(job, profile, client)
            console.print()
            console.print(Panel(
                Markdown(suggestions),
                title=f"[bold blue]CV Adaptations — {job.title} @ {job.company}[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            ))
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{job.company}_{job.title}")[:50]
            filename = f"cv_adapt_{safe_name}.md"
            save = Prompt.ask("Save to file?", choices=["y", "n"], default="y")
            if save == "y":
                _write_output(filename, f"# CV Adaptations — {job.title} @ {job.company}\n\n{suggestions}")


# ── Main browse loop ──────────────────────────────────────────────────────────

def browse(jobs: list[JobListing], profile: CVProfile, client: anthropic.Anthropic) -> None:
    """Interactive job browser."""
    if not jobs:
        return

    page_size = 5
    page = 0
    total_pages = (len(jobs) - 1) // page_size + 1

    while True:
        start = page * page_size
        end = min(start + page_size, len(jobs))
        page_jobs = jobs[start:end]

        console.print()
        console.print(Rule(f"[bold]Jobs {start+1}–{end} of {len(jobs)}  (page {page+1}/{total_pages})[/bold]"))
        for i, job in enumerate(page_jobs, start=start+1):
            _print_job_card(job, i)

        # Build prompt choices
        choices = []
        valid_nums = [str(i) for i in range(start + 1, end + 1)]
        choices += valid_nums
        if page > 0:
            choices.append("p")
        if end < len(jobs):
            choices.append("n")
        choices.append("q")

        nav = "[cyan][n][/cyan] Next page  " if end < len(jobs) else ""
        nav += "[cyan][p][/cyan] Prev page  " if page > 0 else ""
        console.print(
            f"\n[bold]Enter job number[/bold] to view details  |  {nav}[cyan][q][/cyan] Quit browsing"
        )

        raw = Prompt.ask("Select").strip().lower()

        if raw == "q":
            break
        elif raw == "n" and end < len(jobs):
            page += 1
        elif raw == "p" and page > 0:
            page -= 1
        elif raw in valid_nums:
            idx = int(raw) - 1
            _job_action_menu(jobs[idx], int(raw), profile, client)
        else:
            console.print("[dim]Invalid choice.[/dim]")

    # Show saved summary on exit
    saved = _load_saved()
    if saved:
        console.print(
            Panel(
                f"[bold]{len(saved)} jobs saved[/bold] in [cyan]{SAVED_FILE}[/cyan]",
                border_style="green",
                padding=(0, 1),
            )
        )
