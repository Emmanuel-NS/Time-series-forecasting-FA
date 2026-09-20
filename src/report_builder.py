"""Build the submission PDF from the Markdown report.

The assignment requires a PDF. Rather than depending on a LaTeX installation,
the report is written in Markdown, rendered to a single self-contained HTML file
with print-oriented CSS, and printed to PDF by a headless Chromium browser
(Edge or Chrome, both present by default on Windows).

Two conveniences make the Markdown source pleasant to maintain:

``{{table:name}}``
    Inlined as an HTML table rendered from ``results/name.csv``, so tables in the
    report are always the numbers the code actually produced. There is no
    opportunity for a copy-paste error between results and prose.

``{{value:file.csv:column:row}}``
    Inlined as a single formatted number from a results CSV, for quoting figures
    inside sentences.
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown
import pandas as pd

from . import config as C

BROWSER_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Georgia", "Cambria", "Times New Roman", serif;
  font-size: 11pt; line-height: 1.42; color: #14181f; margin: 0;
  text-align: justify; hyphens: auto;
}
h1 { font-size: 16pt; line-height: 1.25; text-align: center; margin: 0 0 4pt; }
h1 + p { text-align: center; font-size: 10.5pt; }
h2 {
  font-size: 13pt; margin: 14pt 0 6pt; padding-bottom: 2pt;
  border-bottom: 0.6pt solid #b9c2cf; page-break-after: avoid;
}
h3 { font-size: 11.5pt; margin: 10pt 0 4pt; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 8pt 0 3pt; font-style: italic; page-break-after: avoid; }
p { margin: 0 0 6pt; }
ul, ol { margin: 0 0 6pt; padding-left: 18pt; }
li { margin-bottom: 2.5pt; }
code {
  font-family: "Consolas", "DejaVu Sans Mono", monospace;
  font-size: 9.5pt; background: #f2f4f7; padding: 0 2px; border-radius: 2px;
}
pre { background: #f6f8fa; border: 0.5pt solid #dde3ea; padding: 6pt 8pt;
      font-size: 9pt; overflow-x: auto; border-radius: 3px; }
pre code { background: none; }

table {
  border-collapse: collapse; width: 100%; margin: 6pt 0 9pt;
  font-family: "Helvetica Neue", Arial, sans-serif; font-size: 9pt;
  page-break-inside: avoid;
}
th, td { border: 0.4pt solid #c3ccd8; padding: 3pt 5pt; text-align: right; }
th { background: #eef1f6; font-weight: 600; text-align: center; }
td:first-child, th:first-child { text-align: left; }
tbody tr:nth-child(even) { background: #fafbfd; }

figure { margin: 5pt 0 7pt; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 7.2cm; object-fit: contain; }
figcaption {
  font-family: "Helvetica Neue", Arial, sans-serif; font-size: 9pt;
  color: #3d4550; margin-top: 2pt; text-align: justify;
}
blockquote {
  margin: 6pt 0; padding: 5pt 9pt; background: #f4f7fb;
  border-left: 2.4pt solid #6b8bb5; font-size: 10.5pt;
}
.caption-label { font-weight: 600; }
a { color: #1d4e89; text-decoration: underline; }
hr { border: none; border-top: 0.5pt solid #ccd4de; margin: 10pt 0; }
.tight-table table { font-size: 8.5pt; }
"""


# ---------------------------------------------------------------------------
# Markdown preprocessing
# ---------------------------------------------------------------------------


def _csv_to_markdown(path: Path, index: bool = True, floatfmt: str = "{:,.3f}") -> str:
    frame = pd.read_csv(path, index_col=0 if index else None)

    def fmt(value):
        if isinstance(value, float):
            if pd.isna(value):
                return "-"
            if abs(value) >= 1000 or (abs(value) < 0.01 and value != 0):
                return f"{value:,.4g}"
            return floatfmt.format(value)
        return str(value)

    formatted = frame.map(fmt) if hasattr(frame, "map") else frame.applymap(fmt)
    return formatted.to_markdown(index=index)


def _expand_directives(text: str) -> str:
    """Replace ``{{table:...}}`` and ``{{value:...}}`` placeholders."""

    def table(match: re.Match) -> str:
        spec = match.group(1)
        name, *flags = spec.split("|")
        path = C.RESULTS_DIR / f"{name.strip()}.csv"
        if not path.exists():
            return f"*[table `{name.strip()}` not available - run the pipeline]*"
        index = "noindex" not in flags
        return _csv_to_markdown(path, index=index)

    def value(match: re.Match) -> str:
        try:
            filename, column, row = (p.strip() for p in match.group(1).split(":"))
            frame = pd.read_csv(C.RESULTS_DIR / filename, index_col=0)
            raw = frame.loc[row, column]
            return f"{raw:,.2f}" if isinstance(raw, float) else str(raw)
        except Exception:
            return "*[n/a]*"

    text = re.sub(r"\{\{table:([^}]+)\}\}", table, text)
    text = re.sub(r"\{\{value:([^}]+)\}\}", value, text)
    return text


def _embed_images(html: str) -> str:
    """Inline every local image as a data URI so the HTML is self-contained."""

    def repl(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = (C.PROJECT_ROOT / src).resolve()
        if not path.exists():
            path = (C.FIGURES_DIR / Path(src).name).resolve()
        if not path.exists():
            print(f"  WARNING: figure not found: {src}")
            return match.group(0)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{encoded}"'

    return re.sub(r'src="([^"]+)"', repl, html)


def _wrap_figures(html: str) -> str:
    """Turn ``<p><img ...><em>caption</em></p>`` into a proper ``<figure>``."""
    pattern = re.compile(
        r"<p>\s*(<img[^>]+>)\s*(?:<br\s*/?>)?\s*(?:<em>(.*?)</em>)?\s*</p>",
        re.DOTALL,
    )

    def repl(match: re.Match) -> str:
        img, caption = match.group(1), match.group(2)
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        return f"<figure>{img}{cap}</figure>"

    return pattern.sub(repl, html)


def _autolink_urls(html: str) -> str:
    """Turn bare http(s) URLs into ``<a href>`` so they survive into the PDF."""

    url_re = re.compile(r"https?://[^\s<>\"']+")

    def linkify_text(text: str) -> str:
        def repl(match: re.Match) -> str:
            url = match.group(0)
            trailing = ""
            while url and url[-1] in ".,;:)]":
                trailing = url[-1] + trailing
                url = url[:-1]
            if not url:
                return match.group(0)
            return f'<a href="{url}">{url}</a>{trailing}'

        return url_re.sub(repl, text)

    # Leave existing anchors untouched; only linkify text outside them.
    parts = re.split(r"(<a\b[^>]*>.*?</a>)", html, flags=re.IGNORECASE | re.DOTALL)
    out: list[str] = []
    for part in parts:
        if part.lower().startswith("<a"):
            out.append(part)
        else:
            # Also skip URLs that already appear inside tag attributes.
            chunks = re.split(r"(<[^>]+>)", part)
            for i, chunk in enumerate(chunks):
                out.append(chunk if i % 2 == 1 else linkify_text(chunk))
    return "".join(out)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _find_browser() -> Path:
    for candidate in BROWSER_CANDIDATES:
        if candidate.exists():
            return candidate
    for name in ("chrome", "msedge", "chromium"):
        found = shutil.which(name)
        if found:
            return Path(found)
    raise RuntimeError(
        "No Chromium-based browser found for PDF export. Install Chrome or Edge, "
        "or convert report/report.html to PDF manually."
    )


def build(source: Path | None = None, pdf: Path | None = None) -> Path:
    source = source or C.REPORT_DIR / "report.md"
    pdf = pdf or C.REPORT_DIR / "report.pdf"
    html_path = C.REPORT_DIR / "report.html"

    if not source.exists():
        raise FileNotFoundError(source)

    print(f"[report] reading {source.name}")
    text = _expand_directives(source.read_text(encoding="utf-8"))

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "attr_list", "footnotes", "md_in_html", "sane_lists"],
    )
    body = _wrap_figures(body)
    body = _autolink_urls(body)
    body = _embed_images(body)

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Comparative Analysis of Sequential Models for Mobile Network "
        "Traffic Forecasting</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")
    print(f"[report] html -> {html_path}  ({len(html) / 1e6:.1f} MB with embedded figures)")

    browser = _find_browser()
    print(f"[report] printing with {browser.name}")
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--user-data-dir={profile}",
                "--no-pdf-header-footer",
                "--virtual-time-budget=20000",
                f"--print-to-pdf={pdf}",
                html_path.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )

    if not pdf.exists():
        raise RuntimeError("PDF was not produced")
    print(f"[report] pdf -> {pdf}  ({pdf.stat().st_size / 1e6:.2f} MB)")
    return pdf
