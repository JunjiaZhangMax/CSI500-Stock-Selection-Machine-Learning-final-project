"""Convert REPORT_DRAFT.md to a styled academic-paper PDF via pandoc + weasyprint."""
import subprocess, re, sys
from pathlib import Path

ROOT   = Path(__file__).parent.parent
SRC    = ROOT / "REPORT_DRAFT.md"
TMP    = ROOT / "outputs" / "report_tmp.html"
OUT    = ROOT / "outputs" / "CSI500_Report_ZhangJunjia.pdf"

TMP.parent.mkdir(exist_ok=True)

# ── Step 1: pandoc MD → standalone HTML with MathML ──────────────
print("Running pandoc...")
r = subprocess.run(
    ["pandoc", str(SRC), "--to", "html5", "--standalone",
     "--mathml", "--metadata", "lang=en", "-o", str(TMP)],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("pandoc error:", r.stderr); sys.exit(1)

# ── Step 2: inject CSS ────────────────────────────────────────────
CSS = """
/* ── Page setup ── */
@page {
    size: A4;
    margin: 2.2cm 2.4cm 2.8cm 2.4cm;
    @bottom-center {
        content: counter(page);
        font-family: "Times New Roman", Times, serif;
        font-size: 10pt;
        color: #555;
    }
    @top-right {
        content: "CSI500 Stock Selection — Junjia Zhang";
        font-family: "Times New Roman", Times, serif;
        font-size: 9pt;
        color: #888;
    }
}
@page :first { @top-right { content: none; } @bottom-center { content: none; } }

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
body {
    font-family: "Times New Roman", Times, serif;
    font-size: 11.5pt;
    line-height: 1.55;
    color: #111;
    max-width: 100%;
    margin: 0;
    padding: 0;
    text-align: justify;
    hyphens: auto;
}

/* ── Title block ── */
header, .title-block {
    text-align: center;
    margin-bottom: 1.8em;
    border-bottom: 2px solid #1a3a6b;
    padding-bottom: 1em;
}
h1.title {
    font-size: 20pt;
    font-weight: bold;
    color: #1a3a6b;
    margin: 0 0 0.3em 0;
    line-height: 1.25;
}
.subtitle { font-size: 13pt; color: #444; margin: 0.2em 0; }
.author   { font-size: 11.5pt; color: #333; margin: 0.5em 0 0.2em; font-style: italic; }
.date     { font-size: 10.5pt; color: #666; margin: 0; }

/* ── Sections ── */
h1, h2, h3, h4 {
    font-family: "Times New Roman", Times, serif;
    color: #1a3a6b;
    page-break-after: avoid;
}
h2 {
    font-size: 14pt;
    font-weight: bold;
    margin-top: 1.8em;
    margin-bottom: 0.4em;
    border-bottom: 1px solid #c0cfe0;
    padding-bottom: 3px;
}
h3 {
    font-size: 12.5pt;
    font-weight: bold;
    margin-top: 1.2em;
    margin-bottom: 0.3em;
}
h4 {
    font-size: 11.5pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 1em;
    margin-bottom: 0.25em;
    color: #2c5282;
}

/* ── Paragraphs ── */
p { margin: 0.5em 0 0.7em; }

/* ── Tables ── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 10.5pt;
    margin: 1em 0 1.2em;
    page-break-inside: avoid;
}
thead tr {
    background-color: #1a3a6b;
    color: #fff;
}
thead th {
    padding: 7px 10px;
    text-align: center;
    font-weight: bold;
    border: 1px solid #1a3a6b;
}
tbody tr:nth-child(even) { background-color: #eef2f8; }
tbody tr:nth-child(odd)  { background-color: #fff; }
tbody td {
    padding: 5px 10px;
    border: 1px solid #c5d0e0;
    vertical-align: top;
}
tbody tr:hover { background-color: #dde8f5; }

/* ── Code blocks ── */
pre, code {
    font-family: "Courier New", Courier, monospace;
    font-size: 9.5pt;
    background: #f5f7fa;
    border: 1px solid #d0d7e4;
    border-radius: 4px;
}
code { padding: 1px 4px; }
pre  { padding: 10px 14px; overflow-x: auto; line-height: 1.4;
       margin: 0.8em 0; white-space: pre-wrap; }
pre code { background: none; border: none; padding: 0; }

/* ── Math ── */
math { font-size: 11pt; }
.math, math[display="block"] {
    display: block;
    text-align: center;
    margin: 0.8em auto;
    overflow-x: auto;
}

/* ── Lists ── */
ul, ol { margin: 0.4em 0 0.8em 1.8em; padding: 0; }
li { margin-bottom: 0.3em; }

/* ── Horizontal rule ── */
hr {
    border: none;
    border-top: 1px solid #c0cfe0;
    margin: 1.5em 0;
}

/* ── Bold / italic ── */
strong { color: #1a3a6b; }

/* ── Block quote (used for findings) ── */
blockquote {
    border-left: 3px solid #1a3a6b;
    margin: 0.8em 0 0.8em 1em;
    padding: 0.4em 0.8em;
    color: #333;
    background: #f0f4fb;
    font-style: normal;
}

/* ── Footer note ── */
footer, .footnotes {
    font-size: 9.5pt;
    color: #666;
    border-top: 1px solid #ccc;
    margin-top: 2em;
    padding-top: 0.5em;
}

/* ── Page breaks ── */
h2 { page-break-before: auto; }

/* ── Override pandoc defaults (max-width:36em, excess padding) ── */
body {
    max-width: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    width: 100% !important;
}
"""

print("Injecting CSS...")
html = TMP.read_text(encoding="utf-8")

# Fix title block rendering from pandoc metadata
html = re.sub(r'<title>(.*?)</title>',
              r'<title>CSI500 Stock Selection Report</title>', html)

# Make sure there's a proper title section rendered from H1
# pandoc puts the title in a header div when using standalone
if '<div id="title-block-header">' not in html and '<header' not in html:
    # inject title manually after <body>
    title_html = """
<div style="text-align:center;border-bottom:2px solid #1a3a6b;padding-bottom:1em;margin-bottom:1.8em">
  <div style="font-size:20pt;font-weight:bold;color:#1a3a6b;margin-bottom:0.3em">
    CSI500 Stock Selection Report</div>
  <div style="font-size:11.5pt;color:#444;margin:0.3em 0">Machine Learning Competition</div>
  <div style="font-size:11.5pt;font-style:italic;color:#333;margin:0.4em 0">
    Junjia (Max) Zhang &nbsp;·&nbsp; jz7842@nyu.edu</div>
  <div style="font-size:10.5pt;color:#666">May 2026</div>
</div>
"""
    html = html.replace("<body>", "<body>\n" + title_html, 1)
else:
    # Style pandoc's title block
    html = html.replace('<div id="title-block-header">',
                        '<div id="title-block-header" style="text-align:center;'
                        'border-bottom:2px solid #1a3a6b;padding-bottom:1em;'
                        'margin-bottom:1.8em">')

# Strip the H1 that duplicates the title (pandoc renders it in body too)
html = re.sub(r'<h1[^>]*>CSI500 Stock Selection Report</h1>', '', html)

# Strip pandoc's embedded body constraints (max-width, padding, margin)
html = re.sub(
    r'(body\s*\{[^}]*?)max-width\s*:\s*[^;]+;',
    r'\1max-width: none;',
    html, flags=re.DOTALL
)
html = re.sub(
    r'(body\s*\{[^}]*?)padding-left\s*:\s*[^;]+;',
    r'\1padding-left: 0;',
    html, flags=re.DOTALL
)
html = re.sub(
    r'(body\s*\{[^}]*?)padding-right\s*:\s*[^;]+;',
    r'\1padding-right: 0;',
    html, flags=re.DOTALL
)
html = re.sub(
    r'(body\s*\{[^}]*?)padding-top\s*:\s*[^;]+;',
    r'\1padding-top: 0;',
    html, flags=re.DOTALL
)
html = re.sub(
    r'(body\s*\{[^}]*?)padding-bottom\s*:\s*[^;]+;',
    r'\1padding-bottom: 0;',
    html, flags=re.DOTALL
)

# Inject CSS into <head>
html = html.replace("</head>", f"<style>\n{CSS}\n</style>\n</head>")

TMP.write_text(html, encoding="utf-8")

# ── Step 3: Chrome headless HTML → PDF ───────────────────────────
print("Rendering PDF with Chrome headless...")
chrome_candidates = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]
chrome = next((c for c in chrome_candidates if Path(c).exists()), None)
if chrome is None:
    print("Chrome not found. HTML saved at:", TMP); sys.exit(1)

r2 = subprocess.run([
    chrome,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={OUT}",
    "--no-margins",
    "--print-to-pdf-no-header",
    f"file:///{TMP}",
], capture_output=True, text=True)

if OUT.exists():
    print(f"\nDone! → {OUT}")
    print(f"Size: {OUT.stat().st_size / 1024:.0f} KB")
else:
    print("Chrome failed:", r2.stderr)
    sys.exit(1)
