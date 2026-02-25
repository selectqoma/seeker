"""Job matching and ranking using Claude."""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import CVProfile, SearchPreferences, JobListing

logger = logging.getLogger(__name__)

# Compact schema — short responses = no truncation
SCORE_PROMPT = """Candidate: {current_title}, {years_experience}y exp ({seniority_level} level), skills: {skills}
Seeking: {target_roles}
Notes: {extra_notes}

Job: {title} @ {company} ({location}) [remote scope: {remote_scope}]
Tags: {tags}
Description: {description}

Score 0-100 fit. Be honest — penalise heavily if the job requires more seniority than the candidate has.
Return ONLY this JSON (keep strings under 80 chars):
{{"score":<int>,"fit":"one sentence why good fit","gap":"main concern or none"}}"""


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_api(client: anthropic.Anthropic, prompt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _parse_score_response(raw: str) -> dict:
    """Parse JSON response with fallback regex extraction."""
    # Strip markdown fences
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract score with regex if JSON is malformed
        m = re.search(r'"score"\s*:\s*(\d+)', raw)
        if m:
            fit_m = re.search(r'"fit"\s*:\s*"([^"]*)"', raw)
            gap_m = re.search(r'"gap"\s*:\s*"([^"]*)"', raw)
            return {
                "score": int(m.group(1)),
                "fit": fit_m.group(1) if fit_m else "",
                "gap": gap_m.group(1) if gap_m else "",
            }
        raise


def _score_job(
    job: JobListing,
    profile: CVProfile,
    prefs: SearchPreferences,
    client: anthropic.Anthropic,
) -> JobListing:
    from .cv_parser import _infer_level
    prompt = SCORE_PROMPT.format(
        current_title=profile.current_title,
        years_experience=profile.years_experience,
        seniority_level=_infer_level(profile.years_experience),
        skills=", ".join(profile.skills[:12]),
        target_roles=", ".join(prefs.target_roles or profile.target_titles),
        extra_notes=prefs.extra_notes or "none",
        title=job.title,
        company=job.company,
        location=job.location,
        remote_scope=job.remote_scope or "unknown",
        tags=", ".join(job.tags[:6]),
        description=job.description[:300],
    )
    try:
        raw = _call_api(client, prompt)
        data = _parse_score_response(raw)
        job.match_score = float(data.get("score", 0))
        fit = data.get("fit", "")
        gap = data.get("gap", "")
        if fit:
            job.match_reasons = [fit]
        if gap and gap.lower() != "none":
            job.match_concerns = [gap]
    except Exception as e:
        logger.warning(f"Scoring failed for '{job.title}' at '{job.company}': {e}")
        job.match_score = 0.0
    return job


def _is_relevant(job: JobListing, profile: CVProfile) -> bool:
    """Cheap keyword pre-filter — skip API call if job is clearly off-domain.

    Splits multi-word terms so 'machine learning' matches a job containing
    just 'machine' or 'learning', and checks against title+tags only (fast).
    """
    words = set()
    for term in profile.target_titles + profile.skills[:15] + profile.industries:
        for word in term.lower().split():
            if len(word) > 3:  # skip short noise words
                words.add(word)

    # Check title and tags first (most reliable signal, no HTML noise)
    haystack = (job.title + " " + " ".join(job.tags)).lower()
    if any(w in haystack for w in words):
        return True

    # Fall back to description snippet
    haystack_desc = job.description[:400].lower()
    return any(w in haystack_desc for w in words)


def rank_jobs(
    jobs: list[JobListing],
    profile: CVProfile,
    prefs: SearchPreferences,
    client: anthropic.Anthropic,
    top_n: int = 20,
    max_workers: int = 3,
    min_score: float = 50.0,
) -> list[JobListing]:
    """Score all jobs concurrently and return top N ranked above min_score."""
    if not jobs:
        return []

    # Drop explicitly excluded keywords
    if prefs.excluded_keywords:
        excluded = [k.lower() for k in prefs.excluded_keywords]
        jobs = [
            j for j in jobs
            if not any(e in (j.title + j.description).lower() for e in excluded)
        ]

    # Cheap pre-filter: skip jobs with zero domain overlap (saves API calls)
    before = len(jobs)
    jobs = [j for j in jobs if _is_relevant(j, profile)]
    skipped = before - len(jobs)
    if skipped:
        logger.info(f"Pre-filter dropped {skipped} off-domain jobs before scoring.")

    logger.info(f"Scoring {len(jobs)} jobs with Claude (workers={max_workers})...")

    scored = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_score_job, job, profile, prefs, client): job for job in jobs}
        for future in as_completed(futures):
            try:
                scored.append(future.result())
            except Exception as e:
                logger.warning(f"Future error: {e}")

    scored = [j for j in scored if j.match_score >= min_score]
    scored.sort(key=lambda j: j.match_score, reverse=True)
    return scored[:top_n]


def generate_summary(
    top_jobs: list[JobListing],
    profile: CVProfile,
    client: anthropic.Anthropic,
) -> str:
    """Generate a high-level search summary and recommendations."""
    if not top_jobs:
        return "No suitable jobs found. Try broadening your search criteria."

    job_list = "\n".join(
        f"- [{j.match_score:.0f}/100] {j.title} @ {j.company} ({j.location}) — {j.source}"
        for j in top_jobs[:10]
    )

    prompt = f"""You are a career advisor. A job seeker just ran an AI-powered job search.

Candidate: {profile.name or 'Job Seeker'} | {profile.current_title} | {profile.years_experience} yrs exp
Skills: {', '.join(profile.skills[:10])}

Top matched jobs:
{job_list}

Write a concise (3-5 sentence) strategic summary:
1. What the results reveal about the market for this candidate
2. Which top 2-3 jobs to prioritize and why
3. One actionable tip to improve their chances

Keep it practical, direct, and encouraging. No fluff."""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        return "Search complete. Review the ranked results above."
