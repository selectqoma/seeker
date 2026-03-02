"""CV Builder Expert — iterative CV refinement through Claude conversation."""
import json
import re
import threading
import uuid

import anthropic

# ── In-memory session store ───────────────────────────────────────────────────

_sessions: dict = {}
_sessions_lock = threading.Lock()

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """You are an expert CV/resume consultant. Help the candidate produce the strongest possible version of their CV through a focused, iterative conversation.

You have their current profile extracted from their existing CV. Work through it section by section.

## Process
1. Start with a brief diagnostic: what's already strong, what needs the most work (2-3 bullet points max)
2. Then work through sections in this order: **Summary → Experience → Skills → Education → Extras**
3. Ask one focused question at a time — don't overwhelm with a wall of questions
4. Always provide a concrete rewritten version alongside any suggestion

## Writing rules
- **Achievement bullets**: [Strong action verb] + [what you did] + [quantified impact]
  ✓ "Reduced API p95 latency by 40% by migrating session storage to Redis, lifting checkout conversion 8%"
  ✗ "Responsible for improving API performance"
- **Summary**: 2–3 sentences, no hollow phrases ("passionate about", "results-driven", "proven track record")
- **Seniority**: be honest — do NOT inflate. 3 years exp ≠ "Senior". Flag if a section oversells.
- **Skills**: group into logical categories (Languages, Frameworks, Cloud, Tools, etc.)
- **NEVER fabricate** — only use what the candidate explicitly confirms

## Output format
After EVERY response where CV content has changed, append a JSON update block (no markdown fences around it):

<cv_update>
{"field": value}
</cv_update>

Valid fields:
- name, email, phone, location, linkedin, github, website → string
- headline → string (one powerful line, e.g. "Senior Backend Engineer · Python · distributed systems")
- summary → string (2–3 sentences)
- experience → [{"company":"","title":"","period":"Jan 2022 – Present","location":"","bullets":["verb + action + impact"]}]
- skills → {"Category": ["skill1","skill2"]}  (dict, not list)
- education → [{"institution":"","degree":"","year":"","notes":""}]
- languages → ["English (Native)", "French (B2)"]
- certifications → ["AWS Solutions Architect – Associate, 2023"]

When the CV is complete and the user says they are happy with it, end your message with the exact token: <cv_complete/>

Keep messages conversational but efficient. Don't repeat information already established."""


# ── Session lifecycle ─────────────────────────────────────────────────────────


def _profile_to_draft(profile: dict) -> dict:
    """Seed the CV draft from the CVProfile dict."""
    skills_raw = profile.get("skills", [])
    skills = {"Technical Skills": skills_raw} if skills_raw else {}

    education = []
    for edu_str in profile.get("education", []):
        if isinstance(edu_str, dict):
            education.append(edu_str)
        else:
            education.append({"institution": edu_str, "degree": "", "year": "", "notes": ""})

    return {
        "name":           profile.get("name", ""),
        "email":          profile.get("email", ""),
        "phone":          "",
        "location":       profile.get("location", ""),
        "linkedin":       "",
        "github":         "",
        "website":        "",
        "headline":       profile.get("current_title", ""),
        "summary":        profile.get("summary", ""),
        "experience":     [],   # parser doesn't give structured work history; Claude extracts it
        "skills":         skills,
        "education":      education,
        "languages":      profile.get("languages", []),
        "certifications": [],
    }


def _apply_update(draft: dict, update: dict) -> None:
    for key, val in update.items():
        if key in draft:
            draft[key] = val


def _parse_reply(raw: str) -> tuple[str, dict | None, bool]:
    """Split Claude's raw reply into (display_text, cv_update_dict, is_done)."""
    cv_update = None
    done = "<cv_complete/>" in raw

    m = re.search(r"<cv_update>\s*(.*?)\s*</cv_update>", raw, re.DOTALL)
    if m:
        try:
            cv_update = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
        raw = raw[: m.start()] + raw[m.end() :]

    display = raw.replace("<cv_complete/>", "").strip()
    return display, cv_update, done


def start_session(profile: dict, raw_cv_text: str, client: anthropic.Anthropic) -> tuple[str, str, dict]:
    """Initialise a new session. Returns (session_id, first_assistant_message, cv_draft)."""
    draft = _profile_to_draft(profile)
    session_id = uuid.uuid4().hex[:8]

    system = (
        _SYSTEM
        + f"\n\n## Current CV draft (seeded from existing CV):\n```json\n{json.dumps(draft, indent=2)}\n```"
    )
    if raw_cv_text:
        system += f"\n\n## Original CV raw text (for context):\n{raw_cv_text[:3000]}"

    seed_msg = (
        "Hi! I want to work through my CV with you and make it as strong as possible. "
        "Please start by reviewing what you already have about me, tell me the 2–3 biggest "
        "improvement opportunities, and then guide me through improving each section."
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": seed_msg}],
    )
    raw_reply = resp.content[0].text
    display, cv_update, done = _parse_reply(raw_reply)
    if cv_update:
        _apply_update(draft, cv_update)

    with _sessions_lock:
        _sessions[session_id] = {
            "messages": [
                {"role": "user",      "content": seed_msg},
                {"role": "assistant", "content": raw_reply},
            ],
            "cv_draft": draft,
            "system":   system,
            "done":     done,
        }

    return session_id, display, draft


def chat(session_id: str, user_message: str, client: anthropic.Anthropic) -> tuple[str, dict, bool]:
    """Continue the conversation. Returns (display_reply, updated_cv_draft, is_done)."""
    with _sessions_lock:
        session = _sessions.get(session_id)
    if not session:
        raise ValueError("Session not found — please start a new CV builder session.")

    session["messages"].append({"role": "user", "content": user_message})

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=session["system"],
        messages=session["messages"],
    )
    raw_reply = resp.content[0].text
    display, cv_update, done = _parse_reply(raw_reply)

    if cv_update:
        _apply_update(session["cv_draft"], cv_update)

    session["messages"].append({"role": "assistant", "content": raw_reply})
    session["done"] = done

    return display, session["cv_draft"], done


# ── HTML renderer (print-ready) ───────────────────────────────────────────────


def _e(s: object) -> str:
    """HTML-escape a value."""
    return (
        str(s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_html(cv: dict) -> str:
    """Return a print-ready, standalone HTML page for the CV."""
    contact_parts = [
        cv.get("email"), cv.get("phone"), cv.get("location"),
        cv.get("linkedin"), cv.get("github"), cv.get("website"),
    ]
    contact = " · ".join(p for p in contact_parts if p)

    # Experience
    exp_html = ""
    for job in cv.get("experience", []):
        meta = " · ".join(p for p in [job.get("period"), job.get("location")] if p)
        bullets = "".join(f"<li>{_e(b)}</li>" for b in job.get("bullets", []))
        exp_html += f"""
        <div class="job">
          <div class="job-header">
            <span class="company">{_e(job.get("company",""))}</span>
            <span class="period">{_e(meta)}</span>
          </div>
          <div class="job-title">{_e(job.get("title",""))}</div>
          {"<ul>" + bullets + "</ul>" if bullets else ""}
        </div>"""

    # Skills
    skills_html = ""
    raw_skills = cv.get("skills", {})
    if isinstance(raw_skills, dict):
        for cat, items in raw_skills.items():
            skills_html += f'<p><span class="sk-cat">{_e(cat)}:</span> {_e(", ".join(items if isinstance(items, list) else []))}</p>'
    elif isinstance(raw_skills, list):
        skills_html = f'<p>{_e(", ".join(raw_skills))}</p>'

    # Education
    edu_html = ""
    for edu in cv.get("education", []):
        if isinstance(edu, str):
            edu_html += f'<div class="edu"><strong>{_e(edu)}</strong></div>'
            continue
        meta = " · ".join(p for p in [edu.get("year"), edu.get("notes")] if p)
        edu_html += f"""
        <div class="edu">
          <strong>{_e(edu.get("institution",""))}</strong>
          {"<span class='dim'> — " + _e(edu.get("degree","")) + "</span>" if edu.get("degree") else ""}
          {"<span class='dim'> · " + _e(meta) + "</span>" if meta else ""}
        </div>"""

    # Extras
    extras = []
    if cv.get("languages"):
        extras.append(f'<p><strong>Languages:</strong> {_e(", ".join(cv["languages"]))}</p>')
    if cv.get("certifications"):
        extras.append(f'<p><strong>Certifications:</strong> {_e(", ".join(cv["certifications"]))}</p>')

    def section(title: str, body: str) -> str:
        if not body.strip():
            return ""
        return f'<div class="section"><div class="sec-title">{title}</div>{body}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{_e(cv.get("name", "CV"))}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 10.5pt; color: #111;
    max-width: 780px; margin: 0 auto; padding: 36px 28px; line-height: 1.5;
  }}
  h1 {{ font-size: 22pt; letter-spacing: -0.5px; font-weight: bold; }}
  .contact {{ color: #555; font-size: 9pt; margin: 4px 0 6px; }}
  .headline {{ font-size: 11pt; font-style: italic; color: #444; margin-bottom: 18px; }}
  .section {{ margin-top: 16px; }}
  .sec-title {{
    font-size: 8.5pt; font-weight: bold; text-transform: uppercase;
    letter-spacing: 1.8px; border-bottom: 1px solid #111; padding-bottom: 3px; margin-bottom: 8px;
  }}
  .job {{ margin-bottom: 12px; }}
  .job-header {{ display: flex; justify-content: space-between; align-items: baseline; }}
  .company {{ font-weight: bold; font-size: 10.5pt; }}
  .period {{ font-size: 9pt; color: #666; }}
  .job-title {{ font-style: italic; font-size: 9.5pt; color: #444; margin: 1px 0 4px; }}
  ul {{ padding-left: 15px; }}
  li {{ margin-bottom: 2px; font-size: 10pt; }}
  .edu {{ margin-bottom: 7px; }}
  .dim {{ color: #666; font-weight: normal; }}
  .sk-cat {{ font-weight: bold; }}
  p {{ margin: 4px 0; }}
  .print-btn {{
    position: fixed; bottom: 20px; right: 20px;
    background: #111; color: #fff; border: none;
    padding: 10px 22px; border-radius: 8px; cursor: pointer;
    font-family: system-ui, sans-serif; font-size: 13px; font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }}
  .print-btn:hover {{ background: #333; }}
  @media print {{
    .print-btn {{ display: none !important; }}
    body {{ padding: 0; max-width: none; }}
    @page {{ margin: 18mm 15mm; size: A4; }}
    .section {{ break-inside: avoid; }}
    .job {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<button class="print-btn" onclick="window.print()">⬇ Save as PDF</button>
<h1>{_e(cv.get("name", ""))}</h1>
<div class="contact">{_e(contact)}</div>
{"<div class='headline'>" + _e(cv.get("headline","")) + "</div>" if cv.get("headline") else ""}
{section("Professional Summary", "<p>" + _e(cv.get("summary","")) + "</p>")}
{section("Experience", exp_html)}
{section("Skills", skills_html)}
{section("Education", edu_html)}
{"".join(extras)}
</body>
</html>"""


# ── PDF renderer (fpdf2) ──────────────────────────────────────────────────────


def render_pdf(cv: dict) -> bytes:
    """Generate a PDF using fpdf2. Raises ImportError if fpdf2 is not installed."""
    from fpdf import FPDF  # type: ignore

    def s(v: object) -> str:
        """Safe encode to latin-1 (fpdf2 built-in fonts are latin-1 only)."""
        return str(v or "").encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    # ── Name ──────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(W, 9, s(cv.get("name", "")), new_x="LMARGIN", new_y="NEXT")

    # ── Contact ───────────────────────────────────────────────────────────────
    contact_parts = [
        cv.get("email"), cv.get("phone"), cv.get("location"),
        cv.get("linkedin"), cv.get("github"),
    ]
    contact = " | ".join(p for p in contact_parts if p)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(W, 4, s(contact), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    # ── Headline ──────────────────────────────────────────────────────────────
    if cv.get("headline"):
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(W, 5, s(cv.get("headline")), new_x="LMARGIN", new_y="NEXT")

    def section_header(title: str) -> None:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(W, 4, title.upper(), new_x="LMARGIN", new_y="NEXT")
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(2)

    # ── Summary ───────────────────────────────────────────────────────────────
    if cv.get("summary"):
        section_header("Professional Summary")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(W, 4.5, s(cv.get("summary")))

    # ── Experience ────────────────────────────────────────────────────────────
    if cv.get("experience"):
        section_header("Experience")
        for exp in cv["experience"]:
            company_title = " — ".join(p for p in [exp.get("company"), exp.get("title")] if p)
            period = s(exp.get("period", ""))

            pdf.set_font("Helvetica", "B", 9)
            title_w = W - 38
            pdf.cell(title_w, 5, s(company_title), new_x="RIGHT", new_y="LAST")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(38, 5, period, new_x="LMARGIN", new_y="NEXT", align="R")
            pdf.set_text_color(0, 0, 0)

            if exp.get("location"):
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(120, 120, 120)
                pdf.cell(W, 3.5, s(exp["location"]), new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)

            pdf.set_font("Helvetica", "", 9)
            for bullet in exp.get("bullets", []):
                pdf.cell(4, 4.5, chr(149), new_x="RIGHT", new_y="LAST")  # bullet char
                pdf.multi_cell(W - 4, 4.5, s(bullet))
            pdf.ln(2)

    # ── Skills ────────────────────────────────────────────────────────────────
    raw_skills = cv.get("skills", {})
    if raw_skills:
        section_header("Skills")
        pdf.set_font("Helvetica", "", 9)
        if isinstance(raw_skills, dict):
            for cat, items in raw_skills.items():
                label = s(cat) + ": "
                lw = pdf.get_string_width(label) + 1
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(lw, 4.5, label, new_x="RIGHT", new_y="LAST")
                pdf.set_font("Helvetica", "", 9)
                items_str = s(", ".join(items if isinstance(items, list) else []))
                pdf.multi_cell(W - lw, 4.5, items_str)
        elif isinstance(raw_skills, list):
            pdf.multi_cell(W, 4.5, s(", ".join(raw_skills)))

    # ── Education ─────────────────────────────────────────────────────────────
    if cv.get("education"):
        section_header("Education")
        for edu in cv["education"]:
            if isinstance(edu, str):
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(W, 5, s(edu), new_x="LMARGIN", new_y="NEXT")
                continue
            line = " — ".join(p for p in [edu.get("institution"), edu.get("degree")] if p)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(W, 5, s(line), new_x="LMARGIN", new_y="NEXT")
            meta = " · ".join(p for p in [edu.get("year"), edu.get("notes")] if p)
            if meta:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(110, 110, 110)
                pdf.cell(W, 4, s(meta), new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)
            pdf.ln(1.5)

    # ── Languages + Certifications ────────────────────────────────────────────
    if cv.get("languages") or cv.get("certifications"):
        section_header("Additional")
        pdf.set_font("Helvetica", "", 9)
        if cv.get("languages"):
            label = "Languages: "
            lw = pdf.get_string_width(label) + 1
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(lw, 4.5, label, new_x="RIGHT", new_y="LAST")
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(W - lw, 4.5, s(", ".join(cv["languages"])))
        if cv.get("certifications"):
            label = "Certs: "
            lw = pdf.get_string_width(label) + 1
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(lw, 4.5, label, new_x="RIGHT", new_y="LAST")
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(W - lw, 4.5, s(", ".join(cv["certifications"])))

    return bytes(pdf.output())
