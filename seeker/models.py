"""Data models for the job seeking agent."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CVProfile:
    """Extracted profile from a CV."""
    name: str = ""
    email: str = ""
    location: str = ""
    summary: str = ""
    skills: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    years_experience: int = 0
    current_title: str = ""
    target_titles: list[str] = field(default_factory=list)
    industries: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class SearchPreferences:
    """User preferences for job search."""
    remote_only: bool = True
    remote_scope: str = "worldwide"  # worldwide | europe | us | country:<name>
    max_days_old: int = 14
    locations: list[str] = field(default_factory=list)
    min_salary: Optional[int] = None
    max_salary: Optional[int] = None
    job_types: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    excluded_keywords: list[str] = field(default_factory=list)
    experience_level: str = ""  # junior, mid, senior, lead, principal
    target_roles: list[str] = field(default_factory=list)
    country: str = ""              # candidate's country, e.g. "Belgium"
    employment_type: str = "both"  # employee | contractor | both
    extra_notes: str = ""


@dataclass
class JobListing:
    """A single job listing."""
    title: str
    company: str
    location: str
    url: str
    source: str
    description: str = ""
    salary: str = ""
    job_type: str = ""
    posted_date: str = ""
    tags: list[str] = field(default_factory=list)
    remote_scope: str = ""  # Worldwide | Europe | US only | UK only | etc.
    match_score: float = 0.0
    match_reasons: list[str] = field(default_factory=list)
    match_concerns: list[str] = field(default_factory=list)
    apply_url: str = ""
