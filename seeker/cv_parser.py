"""CV parsing module - extracts structured profile from PDF or text CVs."""
import json
from pathlib import Path

import pdfplumber
import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .models import CVProfile, SearchPreferences

console = Console()


def extract_text_from_pdf(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def load_cv_text(cv_path: str) -> str:
    path = Path(cv_path)
    if not path.exists():
        raise FileNotFoundError(f"CV file not found: {cv_path}")
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    return path.read_text(encoding="utf-8")


PARSE_PROMPT = """You are a CV/resume parser. Extract structured information from the CV below.
Be strictly honest — do NOT embellish seniority or skills.

Seniority rules (for years_experience and target_titles):
- 0-2 years → junior titles only
- 3-5 years → mid-level titles (avoid "Senior" prefix)
- 6-9 years → senior titles are appropriate
- 10+ years → lead / principal / staff titles

Return ONLY valid JSON:
{{
  "name": "Full Name",
  "email": "email@example.com",
  "location": "City, Country",
  "summary": "One honest sentence about the candidate's current level",
  "skills": ["skill1", "skill2"],
  "languages": ["English"],
  "years_experience": 4,
  "current_title": "Most recent job title (exact, from CV)",
  "seniority_level": "junior|mid|senior|lead",
  "target_titles": ["Realistic next-step title 1", "Realistic next-step title 2"],
  "industries": ["Industry 1"],
  "education": ["Degree, University, Year"]
}}

Rules:
- years_experience: count only paid professional experience, not internships or studies
- target_titles: must match seniority level above — do NOT suggest senior titles for <5 years exp
- summary: describe what the person IS, not what they aspire to be
- If a field is unknown use empty string or empty array

CV TEXT:
{cv_text}"""


def parse_cv(cv_text: str, client: anthropic.Anthropic) -> CVProfile:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": PARSE_PROMPT.format(cv_text=cv_text)}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    years = int(data.get("years_experience", 0))
    profile = CVProfile(
        name=data.get("name", ""),
        email=data.get("email", ""),
        location=data.get("location", ""),
        summary=data.get("summary", ""),
        skills=data.get("skills", []),
        languages=data.get("languages", []),
        years_experience=years,
        current_title=data.get("current_title", ""),
        target_titles=data.get("target_titles", []),
        industries=data.get("industries", []),
        education=data.get("education", []),
        raw_text=cv_text,
    )
    return profile


def prompt_candidate_wishes(profile: CVProfile) -> dict:
    """Interactively ask the candidate what they're looking for."""
    console.print(Panel(
        "Answer a few questions to tailor the search.\n"
        "[dim]Press Enter to accept the default shown in brackets.[/dim]",
        title="[bold cyan]What are you looking for?[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    ))

    default_roles = ", ".join(profile.target_titles[:2]) if profile.target_titles else ""
    roles_input = Prompt.ask(
        f"Target role(s)",
        default=default_roles or "leave blank to use CV targets",
    )
    target_roles = [r.strip() for r in roles_input.split(",") if r.strip()] or profile.target_titles

    # Country — prefill from CV location
    default_country = profile.location.split(",")[-1].strip() if profile.location else ""
    country_input = Prompt.ask(
        "Your country (used to filter location eligibility)",
        default=default_country or "Belgium",
    )
    country = country_input.strip()

    remote_scope = Prompt.ask(
        "Remote scope preference",
        choices=["worldwide", "europe", "us", "uk", "canada", "australia", "other"],
        default="worldwide",
    )
    if remote_scope == "other":
        remote_scope = Prompt.ask("Specify region/country").strip().lower()

    employment_type = Prompt.ask(
        "Employment type",
        choices=["employee", "contractor", "both"],
        default="both",
    )

    salary_input = Prompt.ask(
        "Minimum annual salary (USD, leave blank to skip)",
        default="",
    )
    min_salary = None
    if salary_input.strip():
        cleaned = salary_input.replace(",", "").replace("k", "000").replace("$", "").strip()
        try:
            min_salary = int(cleaned)
        except ValueError:
            pass

    avoid_input = Prompt.ask(
        "Keywords / roles to avoid (comma-separated, or leave blank)",
        default="",
    )
    excluded = [e.strip() for e in avoid_input.split(",") if e.strip()]

    notes_input = Prompt.ask(
        "Anything else? (e.g. 'prefer startups', 'no consulting', 'equity important')",
        default="",
    )

    return {
        "target_roles": target_roles,
        "country": country,
        "remote_scope": remote_scope,
        "employment_type": employment_type,
        "min_salary": min_salary,
        "excluded_keywords": excluded,
        "extra_notes": notes_input.strip(),
    }


def build_search_preferences(profile: CVProfile, user_prefs: dict) -> SearchPreferences:
    wishes = user_prefs.get("wishes", {})
    keywords = wishes.get("target_roles") or profile.target_titles[:2] + profile.skills[:3]
    return SearchPreferences(
        remote_only=user_prefs.get("remote_only", True),
        remote_scope=wishes.get("remote_scope", "worldwide"),
        max_days_old=user_prefs.get("max_days_old", 14),
        locations=user_prefs.get("locations", [profile.location] if profile.location else []),
        min_salary=wishes.get("min_salary") or user_prefs.get("min_salary"),
        job_types=user_prefs.get("job_types", ["full-time"]),
        keywords=keywords,
        excluded_keywords=wishes.get("excluded_keywords", []) + user_prefs.get("excluded_keywords", []),
        experience_level=_infer_level(profile.years_experience),
        target_roles=wishes.get("target_roles", profile.target_titles),
        country=wishes.get("country", profile.location.split(",")[-1].strip() if profile.location else ""),
        employment_type=wishes.get("employment_type", "both"),
        extra_notes=wishes.get("extra_notes", ""),
    )


def _infer_level(years: int) -> str:
    if years <= 2:
        return "junior"
    if years <= 5:
        return "mid"
    if years <= 9:
        return "senior"
    return "lead"
