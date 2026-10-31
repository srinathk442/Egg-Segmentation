"""Build report.pdf from the markdown sections in reports/ (single source of truth for the text).

Supports the markdown subset used there: # headings, paragraphs, * / - / 1. lists, | tables |, **bold**, *italic*,
`code`, [links](url), ![caption](image) and {{include file.md}}.
usage: python scripts/build_report.py  ->  report.pdf
"""
import os, re
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from reportlab.lib.fonts import addMapping

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REP = os.path.join(ROOT, "reports")
SECTIONS = ["00_overview.md", "task1_sam.md", "task2_classical.md", "task3_analysis.md", "task4_tracking.md",
            "ai_disclosure.md"]
CACHE = os.path.join(ROOT, "outputs", "report_figs", "_pdf_cache")
os.makedirs(CACHE, exist_ok=True)

F = "C:/Windows/Fonts/"
for name, file in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf"),
                   ("Arial-BoldItalic", "arialbi.ttf"), ("Consolas", "consola.ttf")]:
    pdfmetrics.registerFont(TTFont(name, F + file))
addMapping("Arial", 0, 0, "Arial"); addMapping("Arial", 1, 0, "Arial-Bold")
addMapping("Arial", 0, 1, "Arial-Italic"); addMapping("Arial", 1, 1, "Arial-BoldItalic")

INK, INK2, ACC, RULE, HEAD_BG = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#256abf"), \
    colors.HexColor("#d8d7d2"), colors.HexColor("#eef3fb")
ST = {
    "h1": ParagraphStyle("h1", keepWithNext=1, fontName="Arial-Bold", fontSize=17, leading=21, textColor=INK, spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", keepWithNext=1, fontName="Arial-Bold", fontSize=12.5, leading=16, textColor=ACC, spaceBefore=10, spaceAfter=5),
    "h3": ParagraphStyle("h3", keepWithNext=1, fontName="Arial-Bold", fontSize=10.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Arial", fontSize=9.3, leading=13, textColor=INK, spaceAfter=5),
    "li": ParagraphStyle("li", fontName="Arial", fontSize=9.3, leading=12.6, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="Arial-Italic", fontSize=8.3, leading=11, textColor=INK2, alignment=TA_CENTER,
                          spaceBefore=3, spaceAfter=9),
    "td": ParagraphStyle("td", fontName="Arial", fontSize=8, leading=10, textColor=INK),
    "th": ParagraphStyle("th", fontName="Arial-Bold", fontSize=8, leading=10, textColor=INK),
    "tds": ParagraphStyle("tds", fontName="Arial", fontSize=6.8, leading=8.4, textColor=INK),
    "ths": ParagraphStyle("ths", fontName="Arial-Bold", fontSize=6.8, leading=8.4, textColor=INK),
}
PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", r'<font name="Consolas" size="8.3">\1</font>', t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#256abf"><u>\1</u></link>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<i>\1</i>", t)
    return t


def image(path, caption, base):
    frac = 1.0                                     # "caption|w=0.6" -> figure at 60% of the text width
    if "|w=" in caption:
        caption, f = caption.rsplit("|w=", 1); frac = float(f)
    p = os.path.normpath(os.path.join(base, path))
    im = PILImage.open(p)
    w, h = im.size
    if w > 1800:                                   # downscale large figures for a small PDF
        im = im.convert("RGB").resize((1800, round(h * 1800 / w)), PILImage.LANCZOS); w, h = im.size
    cp = os.path.join(CACHE, os.path.basename(p).rsplit(".", 1)[0] + ".jpg")
    im.convert("RGB").save(cp, quality=85)
    max_h = PAGE_H - 2 * MARGIN - 60
    sw = CONTENT_W * frac; sh = sw * h / w
    if sh > max_h:
        sh = max_h; sw = sh * w / h
    return KeepTogether([Image(cp, width=sw, height=sh), Paragraph(inline(caption), ST["cap"])])


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [cells[0]] + [r for r in cells[1:] if not re.fullmatch(r"[\s:\-]*", "".join(r))]
    n = len(cells[0])
    small = n >= 7
    td, th = (ST["tds"], ST["ths"]) if small else (ST["td"], ST["th"])
    data = [[Paragraph(inline(c), th if i == 0 else td) for c in r] for i, r in enumerate(cells)]
    if n == 7:                                     # experiment log
        ratios = [0.12, 0.27, 0.21, 0.065, 0.06, 0.06, 0.215]
    else:
        lens = [max(len(re.sub(r"[*`]", "", r[j])) for r in cells) for j in range(n)]
        lens = [min(max(l, 6), 60) for l in lens]
        ratios = [l / sum(lens) for l in lens]
    t = Table(data, colWidths=[r * CONTENT_W for r in ratios], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACC),
                           ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return [t, Spacer(1, 7)]


def parse(md, base, first_section):
    lines = md.split("\n")
    i, out, para = 0, [], []

    def flush():
        if para:
            out.append(Paragraph(inline(" ".join(para)), ST["p"])); para.clear()

    while i < len(lines):
        ln = lines[i]; s = ln.strip()
        m_inc = re.fullmatch(r"\{\{include (.+)\}\}", s)
        if m_inc:
            flush(); out += parse(open(os.path.join(base, m_inc.group(1)), encoding="utf-8").read(), base, True)
            i += 1; continue
        if not s:
            flush(); i += 1; continue
        m = re.match(r"^(#{1,3}) (.*)", s)
        if m:
            flush(); lvl = len(m.group(1))
            if lvl == 1 and not first_section:
                out.append(PageBreak())
            out.append(Paragraph(inline(m.group(2)), ST[f"h{lvl}"])); i += 1; continue
        m = re.fullmatch(r"!\[(.*)\]\((.+)\)", s)
        if m:
            flush(); out.append(image(m.group(2), m.group(1), base)); i += 1; continue
        if s.startswith("|"):
            flush(); rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i]); i += 1
            out += table(rows); continue
        m = re.match(r"^(\*|-|\d+\.) (.*)", s)
        if m:
            flush(); items, numbered = [], m.group(1)[0].isdigit()
            while i < len(lines):
                s2 = lines[i].strip()
                m2 = re.match(r"^(\*|-|\d+\.) (.*)", s2)
                if m2:
                    items.append(m2.group(2))
                elif s2 and lines[i].startswith(" ") and items:
                    items[-1] += " " + s2              # continuation line of a list item
                else:
                    break
                i += 1
            out.append(ListFlowable([ListItem(Paragraph(inline(t), ST["li"]), leftIndent=12) for t in items],
                                    bulletType="1" if numbered else "bullet", start="1" if numbered else None,
                                    bulletFormat="%s." if numbered else None,
                                    bulletFontName="Arial", bulletFontSize=8 if not numbered else 9, leftIndent=14,
                                    spaceAfter=5))
            continue
        para.append(s); i += 1
    flush()
    return out


def on_page(c, doc):
    c.saveState()
    c.setFont("Arial", 7.5); c.setFillColor(INK2)
    c.drawString(MARGIN, 9 * mm, "Egg segmentation & tracking – CV intern task")
    c.drawRightString(PAGE_W - MARGIN, 9 * mm, f"{doc.page}")
    c.restoreState()


story = []
for k, sec in enumerate(SECTIONS):
    story += parse(open(os.path.join(REP, sec), encoding="utf-8").read(), REP, k == 0)
out = os.path.join(ROOT, "report.pdf")
SimpleDocTemplate(out, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=16 * mm,
                  title="Egg segmentation and tracking", author="CV intern task").build(
    story, onFirstPage=on_page, onLaterPages=on_page)
print("written", out, f"{os.path.getsize(out) / 1e6:.1f} MB")
