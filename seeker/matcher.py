"""Job matching and ranking using Claude."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from .models import CVProfile, SearchPreferences, JobListing

logger = logging.getLogger(__name__)

SCORE_PROMPT = """You are an expert talent and recruiting advisor.

CANDIDATE PROFILE:
- Name: {name}
- Current Title: {current_title}
- Years of Experience: {years_experience}
- Skills: {skills}
- Target Roles: {target_titles}
- Industries: {industries}
- Summary: {summary}
- Education: {education}

JOB LISTING:
- Title: {title}
- Company: {company}
- Location: {location}
- Type: {job_type}
- Salary: {salary}
- Tags: {tags}
- Description:
{description}

Score this job for the candidate on a scale of 0-100.
Consider:
1. Title/role alignment with candidate's experience and targets
2. Skills match (required vs possessed)
3. Experience level match
4. Potential for growth
5. Any red flags or concerns

Return ONLY valid JSON:
{{
  "score": <integer 0-100>,
  "reasons": ["reason1", "reason2", "reason3"],
  "concerns": ["concern1"],
  "one_liner": "Brief 1-sentence summary of fit"
}}"""


def _score_job(job: JobListing, profile: CVProfile, client: anthropic.Anthropic) -> JobListing:
    prompt = SCORE_PROMPT.format(
        name=profile.name,
        current_title=profile.current_title,
        years_experience=profile.years_experience,
        skills=", ".join(profile.skills[:20]),
        target_titles=", ".join(profile.target_titles),
        industries=", ".join(profile.industries),
        summary=profile.summary[:400],
        education=", ".join(profile.education),
        title=job.title,
        company=job.company,
        location=job.location,
        job_type=job.job_type,
        salary=job.salary,
        tags=", ".join(job.tags[:10]),
        description=job.description[:800],
    )
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Fast + cheap for batch scoring
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        job.match_score = float(data.get("score", 0))
        job.match_reasons = data.get("reasons", [])
        job.match_concerns = data.get("concerns", [])
        # Store one_liner in reasons if present
        one_liner = data.get("one_liner", "")
        if one_liner:
            job.match_reasons = [one_liner] + job.match_reasons
    except Exception as e:
        logger.warning(f"Scoring failed for '{job.title}' at '{job.company}': {e}")
        job.match_score = 0.0
    return job


def rank_jobs(
    jobs: list[JobListing],
    profile: CVProfile,
    prefs: SearchPreferences,
    client: anthropic.Anthropic,
    top_n: int = 20,
    max_workers: int = 8,
) -> list[JobListing]:
    """Score all jobs concurrently and return top N ranked."""
    if not jobs:
        return []

    # Pre-filter: exclude keyword blocklist
    if prefs.excluded_keywords:
        excluded = [k.lower() for k in prefs.excluded_keywords]
        jobs = [
            j for j in jobs
            if not any(e in (j.title + j.description).lower() for e in excluded)
        ]

    logger.info(f"Scoring {len(jobs)} jobs with Claude (workers={max_workers})...")

    scored = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_score_job, job, profile, client): job for job in jobs}
        for future in as_completed(futures):
            try:
                scored.append(future.result())
            except Exception as e:
                logger.warning(f"Future error: {e}")

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
