"""Cover letter and CV adaptation generator."""
import anthropic
from .models import CVProfile, JobListing


COVER_LETTER_PROMPT = """You are an expert career coach writing a cover letter.

CANDIDATE:
Name: {name}
Current Title: {current_title}
Years of Experience: {years_experience}
Skills: {skills}
Summary: {summary}
Education: {education}

JOB:
Title: {title}
Company: {company}
Location: {location}
Description:
{description}

Write a compelling, concise cover letter (3-4 paragraphs) that:
1. Opens with a strong hook referencing the specific role and company
2. Connects the candidate's most relevant experience to the job requirements
3. Shows genuine interest in the company (infer from description)
4. Closes with a clear call to action

Tone: professional but human. No generic filler. Keep it under 350 words.
Do NOT include address blocks or date headers — just the letter body."""


CV_ADAPT_PROMPT = """You are an expert resume coach.

CANDIDATE CV SUMMARY:
Title: {current_title}
Skills: {skills}
Summary: {summary}

JOB:
Title: {title}
Company: {company}
Description:
{description}

Give specific, actionable CV adaptation advice to maximize fit for this role.
Return exactly this structure:

## Summary tweak
One revised summary sentence tailored to this role (keep it under 40 words).

## Skills to emphasise
List 3-5 skills from the candidate's profile to move to the top / make more prominent.

## Keywords to add
List 3-5 keywords from the job description the CV is missing (only if the candidate plausibly has them).

## Bullet point suggestions
Write 2-3 new achievement-style bullet points the candidate could add (use placeholders like [metric] where specifics are needed).

Be specific. Do not pad."""


def generate_cover_letter(job: JobListing, profile: CVProfile, client: anthropic.Anthropic) -> str:
    prompt = COVER_LETTER_PROMPT.format(
        name=profile.name or "the candidate",
        current_title=profile.current_title,
        years_experience=profile.years_experience,
        skills=", ".join(profile.skills[:15]),
        summary=profile.summary[:400],
        education=", ".join(profile.education),
        title=job.title,
        company=job.company,
        location=job.location,
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
