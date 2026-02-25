"""Web scrapers for multiple job boards."""
import re
import time
import logging
from abc import ABC, abstractmethod
from urllib.parse import urlencode, quote_plus

import httpx
import feedparser
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import JobListing, CVProfile, SearchPreferences

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _get(url: str, timeout: int = 15) -> httpx.Response:
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=timeout) as client:
        return client.get(url)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


class BaseScraper(ABC):
    name: str = "base"

    @abstractmethod
    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        pass

    def _build_query(self, profile: CVProfile, prefs: SearchPreferences) -> str:
        terms = []
        if prefs.keywords:
            terms.extend(prefs.keywords[:3])
        elif profile.target_titles:
            terms.extend(profile.target_titles[:2])
        if not terms and profile.current_title:
            terms.append(profile.current_title)
        return " ".join(terms)


class RemoteOKScraper(BaseScraper):
    """Scrapes RemoteOK.com via their JSON API."""
    name = "RemoteOK"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        try:
            resp = _get("https://remoteok.com/api", timeout=20)
            if resp.status_code != 200:
                return jobs
            data = resp.json()
            # First item is a legal notice
            listings = [d for d in data if isinstance(d, dict) and d.get("position")]
            query_terms = self._build_query(profile, prefs).lower().split()
            skill_terms = [s.lower() for s in profile.skills[:15]]

            for item in listings:
                title = item.get("position", "")
                company = item.get("company", "")
                url = item.get("url", "") or f"https://remoteok.com/l/{item.get('id', '')}"
                description = item.get("description", "")
                tags = item.get("tags", []) or []
                salary = ""
                if item.get("salary_min") and item.get("salary_max"):
                    salary = f"${item['salary_min']:,} - ${item['salary_max']:,}"
                date = item.get("date", "")

                # Relevance filter: title or tags must match query terms or skills
                combined = (title + " " + " ".join(tags)).lower()
                if not any(t in combined for t in query_terms + skill_terms):
                    continue

                jobs.append(JobListing(
                    title=title,
                    company=company,
                    location="Remote",
                    url=url,
                    source=self.name,
                    description=_clean_html(description)[:1500],
                    salary=salary,
                    job_type="full-time",
                    posted_date=date[:10] if date else "",
                    tags=tags,
                    apply_url=url,
                ))
            logger.info(f"RemoteOK: found {len(jobs)} jobs")
        except Exception as e:
            logger.warning(f"RemoteOK scraper error: {e}")
        return jobs[:30]


class WeWorkRemotelyScraper(BaseScraper):
    """Scrapes We Work Remotely via RSS feeds."""
    name = "WeWorkRemotely"

    FEEDS = [
        "https://weworkremotely.com/categories/remote-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
        "https://weworkremotely.com/categories/remote-design-jobs.rss",
        "https://weworkremotely.com/categories/remote-product-jobs.rss",
        "https://weworkremotely.com/categories/remote-marketing-jobs.rss",
        "https://weworkremotely.com/categories/remote-sales-jobs.rss",
        "https://weworkremotely.com/categories/remote-data-science-jobs.rss",
        "https://weworkremotely.com/remote-jobs.rss",
    ]

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        query_terms = self._build_query(profile, prefs).lower().split()
        skill_terms = [s.lower() for s in profile.skills[:10]]

        for feed_url in self.FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get("title", "")
                    company = _extract_wwr_company(title)
                    clean_title = _extract_wwr_title(title)
                    url = entry.get("link", "")
                    summary = _clean_html(entry.get("summary", ""))[:1500]
                    date = entry.get("published", "")

                    combined = (title + " " + summary[:200]).lower()
                    if not any(t in combined for t in query_terms + skill_terms):
                        continue

                    jobs.append(JobListing(
                        title=clean_title,
                        company=company,
                        location="Remote",
                        url=url,
                        source=self.name,
                        description=summary,
                        job_type="full-time",
                        posted_date=date[:16] if date else "",
                        apply_url=url,
                    ))
            except Exception as e:
                logger.warning(f"WWR feed {feed_url} error: {e}")

        logger.info(f"WeWorkRemotely: found {len(jobs)} jobs")
        return jobs[:30]


class RemotiveScraper(BaseScraper):
    """Scrapes Remotive.com via their public API."""
    name = "Remotive"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        query = self._build_query(profile, prefs)
        try:
            url = f"https://remotive.com/api/remote-jobs?search={quote_plus(query)}&limit=50"
            resp = _get(url, timeout=20)
            if resp.status_code != 200:
                return jobs
            data = resp.json().get("jobs", [])
            skill_terms = [s.lower() for s in profile.skills[:10]]

            for item in data:
                title = item.get("title", "")
                company = item.get("company_name", "")
                url = item.get("url", "")
                description = _clean_html(item.get("description", ""))[:1500]
                salary = item.get("salary", "")
                tags = item.get("tags", [])
                date = item.get("publication_date", "")

                # Secondary filter on skills
                combined = (title + " " + " ".join(tags) + " " + description[:200]).lower()
                if skill_terms and not any(s in combined for s in skill_terms):
                    # Still include if title matches query
                    if not any(t in title.lower() for t in query.lower().split()):
                        continue

                jobs.append(JobListing(
                    title=title,
                    company=company,
                    location=item.get("candidate_required_location", "Remote"),
                    url=url,
                    source=self.name,
                    description=description,
                    salary=salary,
                    job_type=item.get("job_type", "full-time"),
                    posted_date=date[:10] if date else "",
                    tags=tags,
                    apply_url=url,
                ))
            logger.info(f"Remotive: found {len(jobs)} jobs")
        except Exception as e:
            logger.warning(f"Remotive scraper error: {e}")
        return jobs[:30]


class JobicyScraper(BaseScraper):
    """Scrapes Jobicy.com via their public API."""
    name = "Jobicy"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        try:
            resp = _get("https://jobicy.com/api/v2/remote-jobs?count=50", timeout=20)
            if resp.status_code != 200:
                return jobs
            data = resp.json().get("jobs", [])
            query_terms = self._build_query(profile, prefs).lower().split()
            skill_terms = [s.lower() for s in profile.skills[:10]]

            for item in data:
                title = item.get("jobTitle", "")
                company = item.get("companyName", "")
                url = item.get("url", "")
                description = _clean_html(item.get("jobDescription", ""))[:1500]
                salary = item.get("annualSalaryMin", "")
                if salary and item.get("annualSalaryMax"):
                    salary = f"${salary:,} - ${item['annualSalaryMax']:,}"
                tags = item.get("jobIndustry", []) + item.get("jobType", [])
                date = item.get("pubDate", "")

                combined = (title + " " + description[:300]).lower()
                if not any(t in combined for t in query_terms + skill_terms):
                    continue

                jobs.append(JobListing(
                    title=title,
                    company=company,
                    location=item.get("jobGeo", "Remote"),
                    url=url,
                    source=self.name,
                    description=description,
                    salary=str(salary) if salary else "",
                    job_type=", ".join(item.get("jobType", [])),
                    posted_date=str(date)[:10] if date else "",
                    tags=tags,
                    apply_url=url,
                ))
            logger.info(f"Jobicy: found {len(jobs)} jobs")
        except Exception as e:
            logger.warning(f"Jobicy scraper error: {e}")
        return jobs[:20]


class HNHiringParser(BaseScraper):
    """Parses the monthly HN 'Who is Hiring?' thread."""
    name = "HackerNews"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        try:
            # Get the latest "Ask HN: Who is hiring?" post
            search_url = "https://hn.algolia.com/api/v1/search?query=Ask+HN+Who+is+hiring&tags=ask_hn&numericFilters=points>100"
            resp = _get(search_url, timeout=15)
            if resp.status_code != 200:
                return jobs

            hits = resp.json().get("hits", [])
            hiring_post = None
            for hit in hits:
                if "who is hiring" in hit.get("title", "").lower():
                    hiring_post = hit
                    break

            if not hiring_post:
                return jobs

            post_id = hiring_post["objectID"]
            comments_url = f"https://hn.algolia.com/api/v1/search?tags=comment,story_{post_id}&hitsPerPage=100"
            resp = _get(comments_url, timeout=15)
            if resp.status_code != 200:
                return jobs

            comments = resp.json().get("hits", [])
            query_terms = self._build_query(profile, prefs).lower().split()
            skill_terms = [s.lower() for s in profile.skills[:10]]

            for comment in comments:
                text = comment.get("comment_text", "") or ""
                clean_text = _clean_html(text)
                combined = clean_text.lower()

                if not any(t in combined for t in query_terms + skill_terms):
                    continue
                if len(clean_text) < 50:
                    continue

                # Extract company and title from first line
                first_line = clean_text.split("\n")[0][:100]
                comment_url = f"https://news.ycombinator.com/item?id={comment.get('objectID', '')}"

                jobs.append(JobListing(
                    title=first_line,
                    company=_extract_hn_company(first_line),
                    location=_extract_hn_location(clean_text),
                    url=comment_url,
                    source=self.name,
                    description=clean_text[:1500],
                    job_type=_extract_hn_type(clean_text),
                    apply_url=comment_url,
                ))

            logger.info(f"HackerNews: found {len(jobs)} jobs")
        except Exception as e:
            logger.warning(f"HN scraper error: {e}")
        return jobs[:20]


class ArbeitNowScraper(BaseScraper):
    """Scrapes ArbeitNow API for remote + EU jobs."""
    name = "ArbeitNow"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        query = self._build_query(profile, prefs)
        try:
            params = {"search": query, "page": 1}
            if prefs.remote_only:
                params["remote"] = "true"
            url = "https://www.arbeitnow.com/api/job-board-api?" + urlencode(params)
            resp = _get(url, timeout=20)
            if resp.status_code != 200:
                return jobs
            data = resp.json().get("data", [])
            skill_terms = [s.lower() for s in profile.skills[:10]]

            for item in data:
                title = item.get("title", "")
                company = item.get("company_name", "")
                url = item.get("url", "")
                description = _clean_html(item.get("description", ""))[:1500]
                tags = item.get("tags", [])
                date = item.get("created_at", "")
                location = item.get("location", "")
                if item.get("remote"):
                    location = "Remote" + (f" / {location}" if location else "")

                combined = (title + " " + " ".join(tags) + " " + description[:300]).lower()
                if skill_terms and not any(s in combined for s in skill_terms):
                    if not any(t in title.lower() for t in query.lower().split()):
                        continue

                jobs.append(JobListing(
                    title=title,
                    company=company,
                    location=location,
                    url=url,
                    source=self.name,
                    description=description,
                    job_type="full-time",
                    posted_date=str(date)[:10] if date else "",
                    tags=tags,
                    apply_url=url,
                ))
            logger.info(f"ArbeitNow: found {len(jobs)} jobs")
        except Exception as e:
            logger.warning(f"ArbeitNow scraper error: {e}")
        return jobs[:20]


class TheMuseScraper(BaseScraper):
    """Scrapes The Muse via their public API."""
    name = "TheMuse"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        query = self._build_query(profile, prefs)
        try:
            params = {"category": query, "page": 1, "descending": "true"}
            url = "https://www.themuse.com/api/public/jobs?" + urlencode(params)
            resp = _get(url, timeout=20)
            if resp.status_code != 200:
                return jobs
            data = resp.json().get("results", [])
            skill_terms = [s.lower() for s in profile.skills[:10]]
            query_terms = query.lower().split()

            for item in data:
                title = item.get("name", "")
                company = item.get("company", {}).get("name", "")
                refs = item.get("refs", {})
                landing_url = refs.get("landing_page", "")
                levels = [l.get("name", "") for l in item.get("levels", [])]
                locations = item.get("locations", [])
                loc_str = ", ".join(l.get("name", "") for l in locations) or "Unknown"
                categories = [c.get("name", "") for c in item.get("categories", [])]
                published = item.get("publication_date", "")

                combined = (title + " " + " ".join(categories)).lower()
                if not any(t in combined for t in query_terms):
                    continue

                jobs.append(JobListing(
                    title=title,
                    company=company,
                    location=loc_str,
                    url=landing_url,
                    source=self.name,
                    description=f"Level: {', '.join(levels)}. Categories: {', '.join(categories)}",
                    job_type="full-time",
                    posted_date=published[:10] if published else "",
                    tags=categories,
                    apply_url=landing_url,
                ))
            logger.info(f"TheMuse: found {len(jobs)} jobs")
        except Exception as e:
            logger.warning(f"TheMuse scraper error: {e}")
        return jobs[:15]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean_html(html: str) -> str:
    """Strip HTML tags and clean whitespace."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_wwr_company(title: str) -> str:
    parts = title.split(" at ", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _extract_wwr_title(title: str) -> str:
    parts = title.split(" at ", 1)
    return parts[0].strip() if parts else title


def _extract_hn_company(first_line: str) -> str:
    # Format: "Company | Role | Location"
    parts = first_line.split("|")
    return parts[0].strip() if parts else first_line


def _extract_hn_location(text: str) -> str:
    remote_patterns = ["remote", "anywhere", "worldwide", "global"]
    lower = text.lower()[:300]
    for p in remote_patterns:
        if p in lower:
            return "Remote"
    return "Unknown"


def _extract_hn_type(text: str) -> str:
    lower = text.lower()[:200]
    if "contract" in lower:
        return "contract"
    if "part-time" in lower or "part time" in lower:
        return "part-time"
    return "full-time"


ALL_SCRAPERS: list[BaseScraper] = [
    RemoteOKScraper(),
    WeWorkRemotelyScraper(),
    RemotiveScraper(),
    JobicyScraper(),
    HNHiringParser(),
    ArbeitNowScraper(),
    TheMuseScraper(),
]


def scrape_all(
    profile: CVProfile,
    prefs: SearchPreferences,
    sources: list[str] | None = None,
) -> list[JobListing]:
    """Run all scrapers and return deduplicated results."""
    scrapers = ALL_SCRAPERS
    if sources:
        scrapers = [s for s in ALL_SCRAPERS if s.name.lower() in [x.lower() for x in sources]]

    all_jobs: list[JobListing] = []
    for scraper in scrapers:
        try:
            jobs = scraper.search(profile, prefs)
            all_jobs.extend(jobs)
        except Exception as e:
            logger.warning(f"Scraper {scraper.name} failed: {e}")

    # Deduplicate by URL
    seen = set()
    deduped = []
    for job in all_jobs:
        key = job.url.rstrip("/")
        if key not in seen:
            seen.add(key)
            deduped.append(job)

    return deduped
