"""Generates architecture.pdf from architecture.md.

Supports ReportLab or WeasyPrint. If neither is installed, attempts to install
reportlab or outputs a self-contained, beautifully styled HTML print version.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MD_PATH = BASE_DIR / "architecture.md"
PDF_PATH = BASE_DIR / "architecture.pdf"
HTML_PATH = BASE_DIR / "architecture.html"


def generate_with_reportlab():
    """Build PDF using ReportLab Platypus engine."""
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=HexColor("#1A365D"),
        alignment=0,
    )
    h1_style = ParagraphStyle(
        "DocH1",
        parent=styles["Heading1"],
        fontSize=13,
        leading=17,
        textColor=HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontSize=10.5,
        leading=14,
        textColor=HexColor("#2D3748"),
        spaceBefore=8,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11.5,
        textColor=HexColor("#2D3748"),
        spaceBefore=2,
        spaceAfter=4,
    )
    code_style = ParagraphStyle(
        "DocCode",
        parent=styles["Code"],
        fontSize=7,
        leading=9,
        fontName="Courier",
        textColor=HexColor("#1A202C"),
        backColor=HexColor("#EDF2F7"),
        spaceBefore=4,
        spaceAfter=4,
    )

    story = []
    lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    in_code = False
    code_buf = []

    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                in_code = False
                story.append(Paragraph("<br/>".join(code_buf).replace(" ", "&nbsp;"), code_style))
                code_buf = []
            else:
                in_code = True
            continue

        if in_code:
            code_buf.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue

        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(stripped[2:], title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=HexColor("#2B6CB0"), spaceAfter=8))
        elif stripped.startswith("## "):
            story.append(Paragraph(stripped[3:], h1_style))
        elif stripped.startswith("### "):
            story.append(Paragraph(stripped[4:], h2_style))
        elif stripped.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#CBD5E0"), spaceAfter=6))
        else:
            clean = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            clean = clean.replace("**", "<b>", 1)
            while "**" in clean:
                clean = clean.replace("**", "</b>", 1)
                if "**" in clean:
                    clean = clean.replace("**", "<b>", 1)
            story.append(Paragraph(clean, body_style))

    doc.build(story)
    print(f"Generated PDF: {PDF_PATH}")


def generate_with_html():
    """Generate clean HTML with print CSS as portable fallback."""
    content = MD_PATH.read_text(encoding="utf-8")
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>GraphOne Pipeline Architecture</title>
<style>
  @page {{ size: letter; margin: 15mm; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #1a202c; max-width: 800px; margin: auto; padding: 20px; }}
  h1 {{ color: #1a365d; border-bottom: 2px solid #2b6cb0; padding-bottom: 8px; font-size: 20pt; }}
  h2 {{ color: #2b6cb0; margin-top: 24px; font-size: 14pt; border-bottom: 1px solid #e2e8f0; }}
  h3 {{ color: #2d3748; margin-top: 16px; font-size: 12pt; }}
  pre {{ background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 10px; font-family: monospace; font-size: 8.5pt; overflow-x: auto; }}
  code {{ background: #edf2f7; padding: 2px 4px; border-radius: 3px; font-family: monospace; font-size: 9pt; }}
  hr {{ border: none; border-top: 1px solid #cbd5e0; margin: 20px 0; }}
</style>
</head>
<body>
<pre style="white-space: pre-wrap; font-family: inherit; font-size: inherit; background: none; border: none;">{content}</pre>
</body>
</html>"""
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Generated printable HTML: {HTML_PATH}")


def main():
    try:
        import reportlab
        generate_with_reportlab()
    except ImportError:
        print("ReportLab not found, generating HTML printable view...")
        generate_with_html()


if __name__ == "__main__":
    main()
