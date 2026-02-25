"""Cover letter and CV adaptation generator."""
import anthropic
from .models import CVProfile, JobListing
from .cv_parser import _infer_level


COVER_LETTER_PROMPT = """You are an expert career coach writing a cover letter.

CANDIDATE:
Name: {name}
Current Title: {current_title}
Seniority: {seniority_level} ({years_experience} years experience)
Skills: {skills}
Summary: {summary}
Education: {education}

JOB:
Title: {title}
Company: {company}
Location: {location} ({remote_scope})
Description:
{description}

Write a compelling, honest cover letter (3-4 paragraphs):
1. Opens with a strong hook referencing the specific role and company
2. Connects the candidate's most relevant actual experience to the job — do NOT claim seniority or skills they don't have
3. If the role is a stretch (higher level than candidate), acknowledge the ambition naturally without overselling
4. Shows genuine interest in the company
5. Closes with a clear call to action

Rules:
- Do NOT use hollow phrases like "passionate about", "strong track record", "proven ability"
- Do NOT inflate years of experience or imply a seniority level they haven't reached
- Tone: direct, human, specific. Under 350 words.
- No address blocks or date headers — just the letter body."""


CV_ADAPT_PROMPT = """You are an expert resume coach. Be honest — do not suggest the candidate inflate their seniority or claim experience they don't have.

CANDIDATE:
Title: {current_title}
Seniority: {seniority_level} ({years_experience} years)
Skills: {skills}
Summary: {summary}

JOB:
Title: {title}
Company: {company}
Description:
{description}

Give specific, actionable CV adaptation advice. Return exactly this structure:

## Seniority fit
One honest sentence: is this role a good level match, a slight stretch, or overreaching? Be direct.

## Summary tweak
One revised summary sentence tailored to this role (under 40 words, no embellishment).

## Skills to emphasise
3-5 skills from the candidate's actual profile to surface more prominently for this role.

## Keywords to add
3-5 keywords from the job description the CV is missing — only suggest ones the candidate plausibly has.

## Bullet point suggestions
2-3 achievement-style bullet points they could add, using [metric] placeholders where specifics are unknown. Do NOT invent experience.

Be specific. Do not pad."""


def generate_cover_letter(job: JobListing, profile: CVProfile, client: anthropic.Anthropic) -> str:
    prompt = COVER_LETTER_PROMPT.format(
        name=profile.name or "the candidate",
        current_title=profile.current_title,
        seniority_level=_infer_level(profile.years_experience),
        years_experience=profile.years_experience,
        skills=", ".join(profile.skills[:15]),
        summary=profile.summary[:400],
        education=", ".join(profile.education),
        title=job.title,
        company=job.company,
        location=job.location,
        remote_scope=job.remote_scope or "Remote",
        description=job.description[:1200],
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def suggest_cv_adaptations(job: JobListing, profile: CVProfile, client: anthropic.Anthropic) -> str:
    prompt = CV_ADAPT_PROMPT.format(
        current_title=profile.current_title,
        seniority_level=_infer_level(profile.years_experience),
        years_experience=profile.years_experience,
        skills=", ".join(profile.skills[:15]),
        summary=profile.summary[:400],
        title=job.title,
        company=job.company,
        description=job.description[:1200],
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()
