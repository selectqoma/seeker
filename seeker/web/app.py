"""FastAPI web application for Seeker."""
import asyncio
import json
import os
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from seeker.cv_parser import build_search_preferences, load_cv_text, parse_cv
from seeker.web import cv_builder as cvb
from seeker.web import pretty_cv
from seeker.generator import generate_cover_letter, suggest_cv_adaptations
from seeker.matcher import generate_summary, rank_jobs
from seeker.models import CVProfile, JobListing, SearchPreferences
from seeker.scrapers import scrape_all

load_dotenv()

# ── Local storage paths ───────────────────────────────────────────────────────

DATA_DIR = Path.home() / ".seeker"
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "searches").mkdir(exist_ok=True)

PROFILE_PATH = DATA_DIR / "profile.json"
PREFS_PATH = DATA_DIR / "preferences.json"
SAVED_JOBS_PATH = DATA_DIR / "saved_jobs.json"

# ── In-memory search state ────────────────────────────────────────────────────

_searches: dict = {}
_searches_lock = threading.Lock()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Seeker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


# ── Profile endpoints ─────────────────────────────────────────────────────────


@app.get("/api/profile")
async def get_profile():
    if not PROFILE_PATH.exists():
        return JSONResponse({"exists": False})
    data = json.loads(PROFILE_PATH.read_text())
    # Strip raw_text before sending to frontend (large + private)
    data.pop("raw_text", None)
    return JSONResponse({"exists": True, "profile": data})


@app.post("/api/cv/upload")
async def upload_cv(file: UploadFile = File(...)):
    """Upload a CV (PDF or .txt), extract profile via Claude, save locally."""
    content = await file.read()
    suffix = Path(file.filename or "cv.pdf").suffix.lower()
    tmp_path = DATA_DIR / f"cv_upload{suffix}"
    tmp_path.write_bytes(content)
    try:
        cv_text = load_cv_text(str(tmp_path))
        if not cv_text.strip():
            raise HTTPException(status_code=400, detail="CV appears to be empty")
        client = _get_client()
        profile = parse_cv(cv_text, client)
        profile_dict = asdict(profile)
        PROFILE_PATH.write_text(json.dumps(profile_dict, indent=2))
        profile_dict.pop("raw_text", None)
        return JSONResponse({"success": True, "profile": profile_dict})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Preferences endpoints ─────────────────────────────────────────────────────


@app.get("/api/preferences")
async def get_preferences():
    if not PREFS_PATH.exists():
        return JSONResponse({"exists": False})
    return JSONResponse({"exists": True, "preferences": json.loads(PREFS_PATH.read_text())})


@app.post("/api/preferences")
async def save_preferences(body: dict):
    PREFS_PATH.write_text(json.dumps(body, indent=2))
    return JSONResponse({"success": True})


# ── Search endpoints ──────────────────────────────────────────────────────────


def _run_search(
    search_id: str,
    profile_dict: dict,
    user_prefs: dict,
    top_n: int,
    min_score: float,
) -> None:
    """Background thread: scrape → score → summarize, updating shared state."""
    with _searches_lock:
        _searches[search_id] = {
            "status": "scraping",
            "scraped": 0,
            "scored": 0,
            "total": 0,
            "jobs": [],
            "summary": "",
            "error": None,
        }
    try:
        client = _get_client()
        # Restore full profile (with raw_text) from disk for scoring
        full_profile_dict = json.loads(PROFILE_PATH.read_text())
        profile = CVProfile(**full_profile_dict)
        prefs = build_search_preferences(profile, user_prefs)

        # Stage 1: scrape
        jobs = scrape_all(profile, prefs)
        with _searches_lock:
            _searches[search_id]["scraped"] = len(jobs)
            _searches[search_id]["status"] = "scoring"
            _searches[search_id]["total"] = len(jobs)

        # Stage 2: score
        ranked = rank_jobs(jobs, profile, prefs, client, top_n=top_n, min_score=min_score)
        with _searches_lock:
            _searches[search_id]["scored"] = len(ranked)
            _searches[search_id]["jobs"] = [asdict(j) for j in ranked]
            _searches[search_id]["status"] = "summarizing"

        # Stage 3: summary
        summary = generate_summary(ranked, profile, client) if ranked else "No matches found above the minimum score threshold."
        with _searches_lock:
            _searches[search_id]["summary"] = summary
            _searches[search_id]["status"] = "complete"

        # Persist to disk
        (DATA_DIR / "searches" / f"{search_id}.json").write_text(
            json.dumps(_searches[search_id], indent=2)
        )
    except Exception as e:
        with _searches_lock:
            _searches[search_id]["status"] = "error"
            _searches[search_id]["error"] = str(e)


@app.post("/api/search")
async def start_search(body: dict):
    if not PROFILE_PATH.exists():
        raise HTTPException(status_code=400, detail="No CV profile found. Please upload your CV first.")

    user_prefs = body.get("preferences")
    if not user_prefs:
        raise HTTPException(status_code=400, detail="Missing preferences")

    top_n = int(body.get("top_n", 20))
    min_score = float(body.get("min_score", 50))
    search_id = uuid.uuid4().hex[:8]

    profile_dict = json.loads(PROFILE_PATH.read_text())
    t = threading.Thread(
        target=_run_search,
        args=(search_id, profile_dict, user_prefs, top_n, min_score),
        daemon=True,
    )
    t.start()
    return JSONResponse({"search_id": search_id})


@app.get("/api/search/{search_id}/events")
async def search_events(search_id: str):
    """Server-Sent Events stream for live search progress."""
    async def generator():
        # Wait briefly for thread to register
        for _ in range(20):
            with _searches_lock:
                if search_id in _searches:
                    break
            await asyncio.sleep(0.1)

        while True:
            with _searches_lock:
                state = _searches.get(search_id)

            if state is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Search not found'})}\n\n"
                break

            payload = {
                "type": "status",
                "status": state["status"],
                "scraped": state["scraped"],
                "scored": state["scored"],
                "total": state["total"],
                "error": state["error"],
            }
            yield f"data: {json.dumps(payload)}\n\n"

            if state["status"] in ("complete", "error"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/api/search/{search_id}/results")
async def get_results(search_id: str):
    with _searches_lock:
        state = _searches.get(search_id)

    if state is None:
        path = DATA_DIR / "searches" / f"{search_id}.json"
        if path.exists():
            state = json.loads(path.read_text())
        else:
            raise HTTPException(status_code=404, detail="Search not found")

    return JSONResponse(state)


# ── Generation endpoints ──────────────────────────────────────────────────────


@app.post("/api/cover-letter")
async def gen_cover_letter(body: dict):
    if not PROFILE_PATH.exists():
        raise HTTPException(status_code=400, detail="No profile found")
    job_dict = body.get("job")
    if not job_dict:
        raise HTTPException(status_code=400, detail="Missing job")

    profile = CVProfile(**json.loads(PROFILE_PATH.read_text()))
    job = JobListing(**job_dict)
    client = _get_client()

    try:
        letter = generate_cover_letter(job, profile, client)
        return JSONResponse({"cover_letter": letter})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/adapt-cv")
async def adapt_cv(body: dict):
    if not PROFILE_PATH.exists():
        raise HTTPException(status_code=400, detail="No profile found")
    job_dict = body.get("job")
    if not job_dict:
        raise HTTPException(status_code=400, detail="Missing job")

    profile = CVProfile(**json.loads(PROFILE_PATH.read_text()))
    job = JobListing(**job_dict)
    client = _get_client()

    try:
        adaptation = suggest_cv_adaptations(job, profile, client)
        return JSONResponse({"adaptation": adaptation})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Saved jobs endpoints ──────────────────────────────────────────────────────


@app.get("/api/saved-jobs")
async def get_saved_jobs():
    if not SAVED_JOBS_PATH.exists():
        return JSONResponse([])
    return JSONResponse(json.loads(SAVED_JOBS_PATH.read_text()))


@app.post("/api/saved-jobs")
async def save_job(body: dict):
    saved = json.loads(SAVED_JOBS_PATH.read_text()) if SAVED_JOBS_PATH.exists() else []
    if any(j.get("url") == body.get("url") for j in saved):
        return JSONResponse({"success": True, "already_saved": True})
    saved.append(body)
    SAVED_JOBS_PATH.write_text(json.dumps(saved, indent=2))
    return JSONResponse({"success": True, "already_saved": False})


@app.delete("/api/saved-jobs/{url:path}")
async def unsave_job(url: str):
    if not SAVED_JOBS_PATH.exists():
        return JSONResponse({"success": True})
    saved = json.loads(SAVED_JOBS_PATH.read_text())
    saved = [j for j in saved if j.get("url") != url]
    SAVED_JOBS_PATH.write_text(json.dumps(saved, indent=2))
    return JSONResponse({"success": True})


# ── CV Builder endpoints ──────────────────────────────────────────────────────


@app.post("/api/cv-builder/start")
async def cv_builder_start():
    if not PROFILE_PATH.exists():
        raise HTTPException(status_code=400, detail="No CV profile found. Please upload your CV first.")
    full = json.loads(PROFILE_PATH.read_text())
    raw_text = full.pop("raw_text", "")
    client = _get_client()
    try:
        session_id, message, draft = cvb.start_session(full, raw_text, client)
        return JSONResponse({"session_id": session_id, "message": message, "cv": draft})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cv-builder/chat")
async def cv_builder_chat(body: dict):
    session_id = body.get("session_id", "")
    user_msg = body.get("message", "").strip()
    if not session_id or not user_msg:
        raise HTTPException(status_code=400, detail="Missing session_id or message")
    client = _get_client()
    try:
        reply, draft, done = cvb.chat(session_id, user_msg, client)
        return JSONResponse({"message": reply, "cv": draft, "done": done})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cv-builder/{session_id}/preview")
async def cv_builder_preview(session_id: str):
    with cvb._sessions_lock:
        session = cvb._sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    from fastapi.responses import HTMLResponse
    return HTMLResponse(cvb.render_html(session["cv_draft"]))


@app.get("/api/cv-builder/{session_id}/pretty-pdf")
async def cv_builder_pretty_pdf(session_id: str):
    """Design agent (Claude) generates beautiful HTML → Playwright renders to PDF."""
    with cvb._sessions_lock:
        session = cvb._sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    client = _get_client()
    try:
        pdf_bytes = await pretty_cv.generate_pretty_pdf(session["cv_draft"], client)
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    name = session["cv_draft"].get("name", "cv").replace(" ", "_").lower()
    from fastapi.responses import Response
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{name}_cv.pdf"'},
    )


@app.get("/api/cv-builder/{session_id}/pdf")
async def cv_builder_pdf(session_id: str):
    with cvb._sessions_lock:
        session = cvb._sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        pdf_bytes = cvb.render_pdf(session["cv_draft"])
    except ImportError:
        raise HTTPException(status_code=501, detail="fpdf2 not installed. Run: pip install fpdf2")
    name = session["cv_draft"].get("name", "cv").replace(" ", "_").lower()
    from fastapi.responses import Response
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{name}_cv.pdf"'},
    )


# ── Static files + root ───────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(_STATIC_DIR / "index.html")
