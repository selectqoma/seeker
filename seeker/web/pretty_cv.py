"""
CV design agent — Claude generates beautiful HTML, Playwright renders it to PDF.

Pipeline:
  cv dict  →  (Claude Sonnet: design agent)  →  styled HTML  →  (Playwright: headless Chromium)  →  PDF bytes
"""

import json
import re

import anthropic

# ── Design prompt ─────────────────────────────────────────────────────────────

_DESIGN_PROMPT = """\
You are an expert CV/resume designer. Your task is to produce a beautiful, print-ready, \
standalone HTML document for the CV data provided.

Follow every specification below precisely.

────────────────────────────────────────────
VISUAL DESIGN
────────────────────────────────────────────
Color palette (use ONLY these):
  • Header background:  #1e293b  (dark slate)
  • Accent / links:     #3b82f6  (blue-500)
  • Body text:          #1f2937  (gray-800)
  • Secondary text:     #6b7280  (gray-500)
  • Borders / rules:    #e5e7eb  (gray-200)
  • Background:         #ffffff

Typography:
  • Font stack: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif
  • Name (h1): 26px, font-weight 800, color #ffffff, letter-spacing -0.5px
  • Headline: 13px, color #93c5fd, font-style italic, margin-top 4px
  • Section titles: 8.5px, font-weight 700, text-transform UPPERCASE, letter-spacing 2px,
                    color #3b82f6, border-bottom 1.5px solid #3b82f6, padding-bottom 3px
  • Body: 10.5px, line-height 1.6, color #1f2937
  • Secondary / meta: 9.5px, color #6b7280

────────────────────────────────────────────
LAYOUT — single column, generous whitespace
────────────────────────────────────────────
Header block (full-width, background #1e293b, padding 28px 36px 24px):
  • Name on its own line (h1 style above)
  • Headline / current title on next line
  • Contact bar: email  ·  phone  ·  location  ·  linkedin  ·  github
    (10px, color #cbd5e1, items separated by " · ", omit empty fields)

Body (padding 28px 36px):
  • Sections in order: Summary → Experience → Skills → Education → Languages/Certs
  • 22px gap between sections
  • Section title bar (see Typography above), margin-bottom 10px

Experience entries:
  • Flex row: company name (bold, 11px) left  +  period (9.5px, #6b7280) right
  • Job title below: 10px italic, color #374151, margin-bottom 5px
  • Bullets: disc list, padding-left 18px, font-size 10px, line-height 1.65
  • 14px gap between entries, break-inside: avoid

Skills:
  • Each category: label in bold + colon, then skill names as inline comma list on same line
  • Skill names: small pill spans — background #eff6ff, color #1d4ed8, border 1px solid #bfdbfe,
    padding 1px 7px, border-radius 9999px, font-size 9px, display inline-block, margin 2px 2px

Education:
  • Institution (bold) — Degree (regular)   ·   Year (color #6b7280)
  • Notes below if present (9px, italic, #6b7280)
  • 8px gap between entries

────────────────────────────────────────────
PRINT CSS (MANDATORY)
────────────────────────────────────────────
@page { size: A4; margin: 0; }
@media print {
  body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .no-print { display: none !important; }
  .job { break-inside: avoid; }
  .section { break-inside: avoid; }
}
All margins / padding must be inside the content — no reliance on @page margin.
Total body width should not exceed 210mm.

────────────────────────────────────────────
RULES
────────────────────────────────────────────
• ALL CSS in one <style> block — NO external resources, NO @import, NO CDN links
• Produce a COMPLETE valid HTML5 document starting with <!DOCTYPE html>
• Do NOT wrap output in markdown fences or add any explanation text
• Omit sections / fields that are empty — do not show empty section headers
• Escape all user content properly (< > & characters)

────────────────────────────────────────────
CV DATA
────────────────────────────────────────────
{cv_json}
"""


# ── Agent call ────────────────────────────────────────────────────────────────


def generate_pretty_html(cv: dict, client: anthropic.Anthropic) -> str:
    """
    Design agent: call Claude Sonnet to produce a beautiful HTML CV.
    Returns the raw HTML string.
    """
    prompt = _DESIGN_PROMPT.replace("{cv_json}", json.dumps(cv, indent=2))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    html = response.content[0].text.strip()

    # Strip accidental markdown code fences
    if html.startswith("```"):
        lines = html.split("\n")
        end = -1 if lines[-1].strip().startswith("```") else len(lines)
        html = "\n".join(lines[1:end]).strip()

    return html


# ── PDF renderer (Playwright / headless Chromium) ─────────────────────────────


async def html_to_pdf(html: str) -> bytes:
    """
    Render HTML to PDF bytes via Playwright (headless Chromium).
    Raises ImportError if playwright is not installed.
    Run once after install:  playwright install chromium
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError:
        raise ImportError(
            "playwright is not installed.\n"
            "Fix: pip install playwright && playwright install chromium"
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html, wait_until="domcontentloaded")
        pdf_bytes = await page.pdf(
            format="A4",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        await browser.close()

    return pdf_bytes


# ── Convenience: full pipeline ────────────────────────────────────────────────


async def generate_pretty_pdf(cv: dict, client: anthropic.Anthropic) -> bytes:
    """Design agent → Playwright → PDF bytes. One call does it all."""
    html = generate_pretty_html(cv, client)
    return await html_to_pdf(html)
