"""Web scrapers for multiple job boards."""
import re
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode, quote_plus

import httpx
import feedparser
from bs4 import BeautifulSoup

from .models import JobListing, CVProfile, SearchPreferences

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _get(url: str, timeout: int = 20, extra_headers: dict = None) -> httpx.Response:
    headers = {**HEADERS, **(extra_headers or {})}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout) as client:
        return client.get(url)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_date(raw: str) -> datetime | None:
    """Try to parse various date formats into a UTC datetime."""
    if not raw:
        return None
    raw = raw.strip()
    # ISO format
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw[:19], fmt[:len(fmt)])
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    # RFC 2822 (RSS)
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc)
    except Exception:
        pass
    # "X days ago" / "X hours ago"
    m = re.search(r"(\d+)\s+(day|hour|minute)", raw.lower())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = timedelta(days=n) if unit == "day" else timedelta(hours=n)
        return datetime.now(timezone.utc) - delta
    return None


def _is_recent(posted_date: str, max_days: int) -> bool:
    """Return True if the job was posted within max_days, or date is unknown."""
    if not posted_date:
        return True  # unknown date → keep
    dt = _parse_date(posted_date)
    if dt is None:
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)
    return dt >= cutoff


def _parse_remote_scope(location: str, description: str = "") -> str:
    """Classify the remote scope of a job from its location string and description."""
    combined = (location + " " + description[:300]).lower()

    us_patterns = [
        "us only", "us-only", "united states only", "usa only",
        "us residents", "must be in the us", "us based", "based in us",
        "north america only", "us or canada", "us/canada",
    ]
    europe_patterns = [
        "europe only", "eu only", "europe remote", "remote in europe",
        "european union", "emea", "cet timezone", "cest", "within europe",
    ]
    uk_patterns = ["uk only", "uk remote", "remote uk", "united kingdom only", "based in uk"]
    canada_patterns = ["canada only", "canada remote", "canadian residents"]
    australia_patterns = ["australia only", "australia remote", "aus only", "apac"]
    latam_patterns = ["latam", "latin america", "south america remote"]
    worldwide_patterns = [
        "worldwide", "globally", "anywhere in the world", "any country",
        "all countries", "global remote", "fully remote", "work from anywhere",
        "location independent",
    ]

    if any(p in combined for p in us_patterns):
        return "US only"
    if any(p in combined for p in uk_patterns):
        return "UK only"
    if any(p in combined for p in canada_patterns):
        return "Canada only"
    if any(p in combined for p in australia_patterns):
        return "Australia/APAC"
    if any(p in combined for p in latam_patterns):
        return "LATAM"
    if any(p in combined for p in europe_patterns):
        return "Europe"
    if any(p in combined for p in worldwide_patterns):
        return "Worldwide"

    # If location just says "Remote" with no qualifier, mark as unspecified
    loc = location.strip().lower()
    if loc in ("remote", "anywhere", ""):
        return "Unspecified"

    # Has a specific country/city name — infer it's local remote
    return f"Remote ({location.strip()})"


def _scope_matches_pref(job_scope: str, pref: str) -> bool:
    """Return True if the job's remote scope is acceptable given candidate preference."""
    if pref == "worldwide":
        return True  # accept everything
    if job_scope in ("Worldwide", "Unspecified"):
        return True  # globally open or unknown — always include

    pref_lower = pref.lower()
    scope_lower = job_scope.lower()

    if pref_lower == "europe" and ("europe" in scope_lower or "eu" in scope_lower or "emea" in scope_lower):
        return True
    if pref_lower == "us" and "us" in scope_lower:
        return True
    if pref_lower == "uk" and "uk" in scope_lower:
        return True
    if pref_lower == "canada" and "canada" in scope_lower:
        return True
    if pref_lower == "australia" and ("australia" in scope_lower or "apac" in scope_lower):
        return True
    # custom region: check substring match
    if pref_lower in scope_lower:
        return True
    return False


def _build_queries(profile: CVProfile, prefs: SearchPreferences) -> list[str]:
    """Generate multiple search queries from profile for broader coverage."""
    queries = []
    # Primary: user-supplied keywords
    if prefs.keywords:
        queries.append(" ".join(prefs.keywords[:3]))
    # Target title combinations
    for title in profile.target_titles[:3]:
        queries.append(title)
    # Title + top skill
    if profile.target_titles and profile.skills:
        queries.append(f"{profile.target_titles[0]} {profile.skills[0]}")
    # Current title
    if profile.current_title and profile.current_title not in queries:
        queries.append(profile.current_title)
    # Deduplicate, keep first 4
    seen, unique = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    return unique[:4]


class BaseScraper(ABC):
    name: str = "base"

    @abstractmethod
    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        pass

    def _primary_query(self, profile: CVProfile, prefs: SearchPreferences) -> str:
        if prefs.keywords:
            return " ".join(prefs.keywords[:3])
        if profile.target_titles:
            return profile.target_titles[0]
        return profile.current_title


# ── Scrapers ──────────────────────────────────────────────────────────────────

class LinkedInScraper(BaseScraper):
    """Scrapes LinkedIn public job search (no login required)."""
    name = "LinkedIn"

    # f_WT=2 = remote, f_TPR = time posted, f_JT=F = full-time
    BASE = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        queries = _build_queries(profile, prefs)
        days = prefs.max_days_old
        # LinkedIn time filter: r86400=1d, r604800=1w, r2592000=1mo
        tpr = "r86400" if days <= 1 else "r604800" if days <= 7 else "r2592000"

        for query in queries[:3]:
            try:
                jobs.extend(self._fetch_page(query, tpr, start=0))
                jobs.extend(self._fetch_page(query, tpr, start=25))
            except Exception as e:
                logger.warning(f"LinkedIn query '{query}' failed: {e}")

        # Also try the standard search page for more results
        for query in queries[:2]:
            try:
                jobs.extend(self._fetch_search_page(query, tpr))
            except Exception as e:
                logger.warning(f"LinkedIn search page '{query}' failed: {e}")

        deduped = {j.url: j for j in jobs if j.url}.values()
        result = [j for j in deduped if _is_recent(j.posted_date, days)]
        logger.info(f"LinkedIn: found {len(result)} jobs")
        return list(result)[:40]

    def _fetch_page(self, query: str, tpr: str, start: int) -> list[JobListing]:
        params = {
            "keywords": query,
            "location": "Worldwide",
            "f_WT": "2",
            "f_TPR": tpr,
            "f_JT": "F",
            "start": start,
        }
        url = f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"
        resp = _get(url, extra_headers={"Accept": "text/html"})
        if resp.status_code != 200:
            return []
        return self._parse_jobs(resp.text)

    def _fetch_search_page(self, query: str, tpr: str) -> list[JobListing]:
        params = {
            "keywords": query,
            "location": "Worldwide",
            "f_WT": "2",
            "f_TPR": tpr,
        }
        url = f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"
        resp = _get(url, extra_headers={"Accept": "text/html"})
        if resp.status_code != 200:
            return []
        return self._parse_jobs(resp.text)

    def _parse_jobs(self, html: str) -> list[JobListing]:
        soup = _soup(html)
        jobs = []
        cards = soup.select("div.base-card, li.jobs-search-results__list-item, div.job-search-card")
        for card in cards:
            try:
                title_el = card.select_one("h3.base-search-card__title, h3, .job-search-card__title")
                company_el = card.select_one("h4.base-search-card__subtitle, h4, .job-search-card__company-name")
                location_el = card.select_one("span.job-search-card__location, .job-search-card__location")
                link_el = card.select_one("a.base-card__full-link, a[href*='/jobs/view/']")
                time_el = card.select_one("time")

                title = title_el.get_text(strip=True) if title_el else ""
                company = company_el.get_text(strip=True) if company_el else ""
                location = location_el.get_text(strip=True) if location_el else "Remote"
                url = link_el["href"].split("?")[0] if link_el and link_el.get("href") else ""
                posted = time_el.get("datetime", "") if time_el else ""

                if not title or not url:
                    continue

                jobs.append(JobListing(
                    title=title,
                    company=company,
                    location=location,
                    url=url,
                    source=self.name,
                    job_type="full-time",
                    posted_date=posted,
                    apply_url=url,
                ))
            except Exception:
                continue
        return jobs


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
            listings = [d for d in data if isinstance(d, dict) and d.get("position")]
            queries = _build_queries(profile, prefs)
            query_words = set(" ".join(queries).lower().split())
            skill_words = {s.lower() for s in profile.skills[:15]}
            signal = query_words | skill_words

            for item in listings:
                title = item.get("position", "")
                company = item.get("company", "")
                url = item.get("url", "") or f"https://remoteok.com/l/{item.get('id', '')}"
                description = _clean_html(item.get("description", ""))[:1500]
                tags = item.get("tags", []) or []
                salary = ""
                if item.get("salary_min") and item.get("salary_max"):
                    salary = f"${item['salary_min']:,} - ${item['salary_max']:,}"
                posted = item.get("date", "")[:10] if item.get("date") else ""

                if not _is_recent(posted, prefs.max_days_old):
                    continue
                combined = (title + " " + " ".join(tags)).lower()
                if not any(w in combined for w in signal):
                    continue

                jobs.append(JobListing(
                    title=title, company=company, location="Remote",
                    url=url, source=self.name, description=description,
                    salary=salary, job_type="full-time", posted_date=posted,
                    tags=tags, apply_url=url,
                ))
            logger.info(f"RemoteOK: found {len(jobs)} jobs")
        except Exception as e:
            logger.warning(f"RemoteOK error: {e}")
        return jobs[:30]


class WeWorkRemotelyScraper(BaseScraper):
    """Scrapes We Work Remotely via RSS feeds."""
    name = "WeWorkRemotely"

    FEEDS = [
        "https://weworkremotely.com/categories/remote-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
        "https://weworkremotely.com/categories/remote-design-jobs.rss",
        "https://weworkremotely.com/categories/remote-product-jobs.rss",
        "https://weworkremotely.com/categories/remote-data-science-jobs.rss",
        "https://weworkremotely.com/remote-jobs.rss",
    ]

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        queries = _build_queries(profile, prefs)
        query_words = set(" ".join(queries).lower().split())
        skill_words = {s.lower() for s in profile.skills[:10]}
        signal = query_words | skill_words

        for feed_url in self.FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get("title", "")
                    url = entry.get("link", "")
                    summary = _clean_html(entry.get("summary", ""))[:1500]
                    published = entry.get("published", "")

                    if not _is_recent(published, prefs.max_days_old):
                        continue
                    combined = (title + " " + summary[:200]).lower()
                    if not any(w in combined for w in signal):
                        continue

                    jobs.append(JobListing(
                        title=_extract_wwr_title(title),
                        company=_extract_wwr_company(title),
                        location="Remote",
                        url=url, source=self.name, description=summary,
                        job_type="full-time", posted_date=published[:16],
                        apply_url=url,
                    ))
            except Exception as e:
                logger.warning(f"WWR feed error: {e}")

        logger.info(f"WeWorkRemotely: found {len(jobs)} jobs")
        return jobs[:30]


class RemotiveScraper(BaseScraper):
    """Scrapes Remotive.com via their public API."""
    name = "Remotive"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        queries = _build_queries(profile, prefs)

        for query in queries[:3]:
            try:
                url = f"https://remotive.com/api/remote-jobs?search={quote_plus(query)}&limit=50"
                resp = _get(url)
                if resp.status_code != 200:
                    continue
                data = resp.json().get("jobs", [])
                for item in data:
                    posted = item.get("publication_date", "")[:10]
                    if not _is_recent(posted, prefs.max_days_old):
                        continue
                    jobs.append(JobListing(
                        title=item.get("title", ""),
                        company=item.get("company_name", ""),
                        location=item.get("candidate_required_location", "Remote"),
                        url=item.get("url", ""),
                        source=self.name,
                        description=_clean_html(item.get("description", ""))[:1500],
                        salary=item.get("salary", ""),
                        job_type=item.get("job_type", "full-time"),
                        posted_date=posted,
                        tags=item.get("tags", []),
                        apply_url=item.get("url", ""),
                    ))
            except Exception as e:
                logger.warning(f"Remotive error: {e}")

        deduped = list({j.url: j for j in jobs}.values())
        logger.info(f"Remotive: found {len(deduped)} jobs")
        return deduped[:40]


class JobicyScraper(BaseScraper):
    """Scrapes Jobicy.com via their public API."""
    name = "Jobicy"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        try:
            resp = _get("https://jobicy.com/api/v2/remote-jobs?count=50&geo=anywhere")
            if resp.status_code != 200:
                return jobs
            data = resp.json().get("jobs", [])
            queries = _build_queries(profile, prefs)
            signal = set(" ".join(queries).lower().split()) | {s.lower() for s in profile.skills[:10]}

            for item in data:
                posted = str(item.get("pubDate", ""))[:10]
                if not _is_recent(posted, prefs.max_days_old):
                    continue
                title = item.get("jobTitle", "")
                desc = _clean_html(item.get("jobDescription", ""))[:1500]
                combined = (title + " " + desc[:300]).lower()
                if not any(w in combined for w in signal):
                    continue
                salary = ""
                if item.get("annualSalaryMin") and item.get("annualSalaryMax"):
                    salary = f"${item['annualSalaryMin']:,} - ${item['annualSalaryMax']:,}"
                jobs.append(JobListing(
                    title=title,
                    company=item.get("companyName", ""),
                    location=item.get("jobGeo", "Remote"),
                    url=item.get("url", ""),
                    source=self.name,
                    description=desc,
                    salary=salary,
                    job_type=", ".join(item.get("jobType", [])),
                    posted_date=posted,
                    tags=item.get("jobIndustry", []) + item.get("jobType", []),
                    apply_url=item.get("url", ""),
                ))
            logger.info(f"Jobicy: found {len(jobs)} jobs")
        except Exception as e:
            logger.warning(f"Jobicy error: {e}")
        return jobs[:25]


class IndeedRSSScraper(BaseScraper):
    """Scrapes Indeed via RSS feed (remote, sorted by date)."""
    name = "Indeed"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        queries = _build_queries(profile, prefs)
        days = prefs.max_days_old
        fromage = min(days, 14)  # Indeed supports max 14

        for query in queries[:3]:
            try:
                params = {
                    "q": query,
                    "l": "remote",
                    "sort": "date",
                    "fromage": fromage,
                    "limit": 50,
                }
                url = "https://www.indeed.com/rss?" + urlencode(params)
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    title = entry.get("title", "")
                    link = entry.get("link", "")
                    summary = _clean_html(entry.get("summary", ""))[:1500]
                    published = entry.get("published", "")
                    company = entry.get("author", "") or _extract_indeed_company(title)

                    if not _is_recent(published, days):
                        continue

                    jobs.append(JobListing(
                        title=_extract_indeed_title(title),
                        company=company,
                        location="Remote",
                        url=link,
                        source=self.name,
                        description=summary,
                        job_type="full-time",
                        posted_date=published[:16] if published else "",
                        apply_url=link,
                    ))
            except Exception as e:
                logger.warning(f"Indeed RSS error for '{query}': {e}")

        deduped = list({j.url: j for j in jobs}.values())
        logger.info(f"Indeed: found {len(deduped)} jobs")
        return deduped[:40]


class HNHiringParser(BaseScraper):
    """Parses the monthly HN 'Who is Hiring?' thread."""
    name = "HackerNews"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        try:
            search_url = (
                "https://hn.algolia.com/api/v1/search"
                "?query=Ask+HN+Who+is+hiring&tags=ask_hn&numericFilters=points>100"
            )
            resp = _get(search_url)
            if resp.status_code != 200:
                return jobs

            hits = resp.json().get("hits", [])
            hiring_post = next(
                (h for h in hits if "who is hiring" in h.get("title", "").lower()), None
            )
            if not hiring_post:
                return jobs

            post_id = hiring_post["objectID"]
            comments_url = (
                f"https://hn.algolia.com/api/v1/search"
                f"?tags=comment,story_{post_id}&hitsPerPage=200"
            )
            resp = _get(comments_url)
            if resp.status_code != 200:
                return jobs

            queries = _build_queries(profile, prefs)
            signal = set(" ".join(queries).lower().split()) | {s.lower() for s in profile.skills[:10]}

            for comment in resp.json().get("hits", []):
                text = _clean_html(comment.get("comment_text", "") or "")
                if len(text) < 50:
                    continue
                # Remote filter
                if prefs.remote_only and not any(
                    w in text.lower()[:300] for w in ["remote", "anywhere", "worldwide"]
                ):
                    continue
                if not any(w in text.lower() for w in signal):
                    continue

                created = comment.get("created_at", "")
                if not _is_recent(created, prefs.max_days_old * 30):  # HN thread is monthly
                    continue

                first_line = text.split("\n")[0][:120]
                cid = comment.get("objectID", "")
                url = f"https://news.ycombinator.com/item?id={cid}"
                jobs.append(JobListing(
                    title=first_line,
                    company=_extract_hn_company(first_line),
                    location=_extract_hn_location(text),
                    url=url, source=self.name,
                    description=text[:1500],
                    job_type=_extract_hn_type(text),
                    apply_url=url,
                ))

            logger.info(f"HackerNews: found {len(jobs)} jobs")
        except Exception as e:
            logger.warning(f"HN error: {e}")
        return jobs[:25]


class ArbeitNowScraper(BaseScraper):
    """Scrapes ArbeitNow API for remote + EU jobs."""
    name = "ArbeitNow"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        queries = _build_queries(profile, prefs)

        for query in queries[:2]:
            try:
                params = {"search": query, "page": 1, "remote": "true"}
                url = "https://www.arbeitnow.com/api/job-board-api?" + urlencode(params)
                resp = _get(url)
                if resp.status_code != 200:
                    continue
                for item in resp.json().get("data", []):
                    posted = str(item.get("created_at", ""))[:10]
                    if not _is_recent(posted, prefs.max_days_old):
                        continue
                    loc = item.get("location", "")
                    if item.get("remote"):
                        loc = "Remote" + (f" / {loc}" if loc else "")
                    jobs.append(JobListing(
                        title=item.get("title", ""),
                        company=item.get("company_name", ""),
                        location=loc or "Remote",
                        url=item.get("url", ""),
                        source=self.name,
                        description=_clean_html(item.get("description", ""))[:1500],
                        job_type="full-time",
                        posted_date=posted,
                        tags=item.get("tags", []),
                        apply_url=item.get("url", ""),
                    ))
            except Exception as e:
                logger.warning(f"ArbeitNow error: {e}")

        deduped = list({j.url: j for j in jobs}.values())
        logger.info(f"ArbeitNow: found {len(deduped)} jobs")
        return deduped[:25]


class TheMuseScraper(BaseScraper):
    """Scrapes The Muse via their public API."""
    name = "TheMuse"

    def search(self, profile: CVProfile, prefs: SearchPreferences) -> list[JobListing]:
        jobs = []
        queries = _build_queries(profile, prefs)

        for query in queries[:2]:
            try:
                params = {"category": query, "page": 1, "descending": "true"}
                url = "https://www.themuse.com/api/public/jobs?" + urlencode(params)
                resp = _get(url)
                if resp.status_code != 200:
                    continue
                query_words = set(query.lower().split())
                for item in resp.json().get("results", []):
                    published = item.get("publication_date", "")
                    if not _is_recent(published, prefs.max_days_old):
                        continue
                    title = item.get("name", "")
                    if not any(w in title.lower() for w in query_words):
                        continue
                    locations = item.get("locations", [])
                    # Remote filter
                    if prefs.remote_only:
                        loc_names = [l.get("name", "").lower() for l in locations]
                        if not any("remote" in l or "anywhere" in l for l in loc_names):
                            continue
                    loc_str = ", ".join(l.get("name", "") for l in locations) or "Remote"
                    refs = item.get("refs", {})
                    landing = refs.get("landing_page", "")
                    cats = [c.get("name", "") for c in item.get("categories", [])]
                    jobs.append(JobListing(
                        title=title,
                        company=item.get("company", {}).get("name", ""),
                        location=loc_str,
                        url=landing, source=self.name,
                        description=f"Categories: {', '.join(cats)}",
                        job_type="full-time",
                        posted_date=published[:10],
                        tags=cats, apply_url=landing,
                    ))
            except Exception as e:
                logger.warning(f"TheMuse error: {e}")

        deduped = list({j.url: j for j in jobs}.values())
        logger.info(f"TheMuse: found {len(deduped)} jobs")
        return deduped[:15]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_wwr_company(title: str) -> str:
    parts = title.split(" at ", 1)
    return parts[1].strip() if len(parts) > 1 else ""

def _extract_wwr_title(title: str) -> str:
    parts = title.split(" at ", 1)
    return parts[0].strip() if parts else title

def _extract_indeed_title(title: str) -> str:
    # "Job Title - Company Name" or "Job Title"
    return title.split(" - ")[0].strip()

def _extract_indeed_company(title: str) -> str:
    parts = title.split(" - ")
    return parts[-1].strip() if len(parts) > 1 else ""

def _extract_hn_company(first_line: str) -> str:
    parts = first_line.split("|")
    return parts[0].strip() if parts else first_line

def _extract_hn_location(text: str) -> str:
    lower = text.lower()[:300]
    for p in ["remote", "anywhere", "worldwide", "global"]:
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
    LinkedInScraper(),
    RemoteOKScraper(),
    WeWorkRemotelyScraper(),
    RemotiveScraper(),
    IndeedRSSScraper(),
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
    """Run all scrapers and return deduplicated, date-filtered results."""
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
    seen, deduped = set(), []
    for job in all_jobs:
        key = job.url.rstrip("/")
        if key and key not in seen:
            seen.add(key)
            deduped.append(job)

    # Enrich each job with parsed remote scope
    for job in deduped:
        job.remote_scope = _parse_remote_scope(job.location, job.description)

    # Filter by candidate's remote scope preference
    if prefs.remote_scope and prefs.remote_scope != "worldwide":
        before = len(deduped)
        deduped = [j for j in deduped if _scope_matches_pref(j.remote_scope, prefs.remote_scope)]
        dropped = before - len(deduped)
        if dropped:
            logger.info(f"Remote scope filter ({prefs.remote_scope}): dropped {dropped} out-of-region jobs")

    return deduped
