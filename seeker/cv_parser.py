"""CV parsing module - extracts structured profile from PDF or text CVs."""
import json
from pathlib import Path

import pdfplumber
import anthropic

from .models import CVProfile, SearchPreferences


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


PARSE_PROMPT = """You are a CV/resume parser. Extract structured information from the CV text below.

Return ONLY valid JSON with this exact structure:
{{
  "name": "Full Name",
  "email": "email@example.com",
  "location": "City, Country",
  "summary": "Brief professional summary",
  "skills": ["skill1", "skill2", ...],
  "languages": ["English", "Spanish", ...],
  "years_experience": 5,
  "current_title": "Current or most recent job title",
  "target_titles": ["Job Title 1", "Job Title 2"],
  "industries": ["Industry 1", "Industry 2"],
  "education": ["Degree, University, Year"]
}}

Rules:
- years_experience: integer, estimate from work history
- target_titles: infer 2-4 realistic next-step job titles based on their background
- skills: include technical and soft skills, tools, frameworks, languages
- languages: spoken/written languages only
- If a field is unknown, use empty string or empty array

CV TEXT:
{cv_text}"""


def parse_cv(cv_text: str, client: anthropic.Anthropic) -> CVProfile:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": PARSE_PROMPT.format(cv_text=cv_text)}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    profile = CVProfile(
        name=data.get("name", ""),
        email=data.get("email", ""),
        location=data.get("location", ""),
        summary=data.get("summary", ""),
        skills=data.get("skills", []),
        languages=data.get("languages", []),
        years_experience=int(data.get("years_experience", 0)),
        current_title=data.get("current_title", ""),
        target_titles=data.get("target_titles", []),
        industries=data.get("industries", []),
        education=data.get("education", []),
        raw_text=cv_text,
    )
    return profile


def build_search_preferences(profile: CVProfile, user_prefs: dict) -> SearchPreferences:
    """Merge CV profile with user-supplied preferences."""
    return SearchPreferences(
        remote_only=user_prefs.get("remote_only", True),
        max_days_old=user_prefs.get("max_days_old", 14),
        locations=user_prefs.get("locations", [profile.location] if profile.location else []),
        min_salary=user_prefs.get("min_salary"),
        max_salary=user_prefs.get("max_salary"),
        job_types=user_prefs.get("job_types", ["full-time"]),
        keywords=user_prefs.get("keywords", profile.target_titles[:2] + profile.skills[:5]),
        excluded_keywords=user_prefs.get("excluded_keywords", []),
        experience_level=user_prefs.get("experience_level", _infer_level(profile.years_experience)),
    )


def _infer_level(years: int) -> str:
    if years <= 2:
        return "junior"
    if years <= 5:
        return "mid"
    if years <= 9:
        return "senior"
    return "lead"
