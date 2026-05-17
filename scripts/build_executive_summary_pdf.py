"""Render the 2-slide executive summary as a landscape PDF.

Mirrors the layout of build_executive_summary.py but produces
reports/output/executive_summary.pdf — useful when the reviewer doesn't have
PowerPoint or LibreOffice installed. The .pptx remains the brief deliverable;
this PDF is the no-Office preview.

Run:
    python scripts/build_executive_summary_pdf.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
OUT_PDF = ROOT / "reports" / "output" / "executive_summary.pdf"

NAVY = colors.HexColor("#1A3552")
BLUE = colors.HexColor("#1F77B4")
RED = colors.HexColor("#D62728")
GREY = colors.HexColor("#555555")
LIGHT_GREY = colors.HexColor("#F3F6FA")
WHITE = colors.white


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "BandTitle": ParagraphStyle(
            "BandTitle", parent=base["Title"],
            fontSize=20, leading=23, alignment=TA_LEFT,
            textColor=WHITE, spaceAfter=0, fontName="Helvetica-Bold",
        ),
        "BandMeta": ParagraphStyle(
            "BandMeta", parent=base["Normal"],
            fontSize=10, leading=12, alignment=TA_LEFT,
            textColor=WHITE, spaceAfter=0, fontName="Helvetica",
        ),
        "Hero": ParagraphStyle(
            "Hero", parent=base["Normal"],
            fontSize=34, leading=38, alignment=TA_LEFT,
            textColor=BLUE, fontName="Helvetica-Bold", spaceAfter=2,
        ),
        "HeroSub": ParagraphStyle(
            "HeroSub", parent=base["Normal"],
            fontSize=10, leading=12, alignment=TA_LEFT,
            textColor=GREY, fontName="Helvetica-Oblique", spaceAfter=8,
        ),
        "HeroDelta": ParagraphStyle(
            "HeroDelta", parent=base["Normal"],
            fontSize=13, leading=15, alignment=TA_LEFT,
            textColor=RED, fontName="Helvetica-Bold", spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=10, leading=13, alignment=TA_JUSTIFY,
            textColor=NAVY, spaceAfter=4, fontName="Helvetica",
        ),
        "SectionTitle": ParagraphStyle(
            "SectionTitle", parent=base["Heading2"],
            fontSize=14, leading=17, alignment=TA_LEFT,
            textColor=NAVY, fontName="Helvetica-Bold", spaceAfter=4,
        ),
        "Bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=10, leading=13, alignment=TA_JUSTIFY,
            textColor=NAVY, fontName="Helvetica", spaceAfter=4,
            leftIndent=8, bulletIndent=0,
        ),
        "Caption": ParagraphStyle(
            "Caption", parent=base["Normal"],
            fontSize=8, leading=10, alignment=TA_CENTER,
            textColor=GREY, fontName="Helvetica-Oblique", spaceAfter=4,
        ),
        "Footer": ParagraphStyle(
            "Footer", parent=base["Normal"],
            fontSize=7.5, leading=9, alignment=TA_LEFT,
            textColor=GREY, fontName="Helvetica-Oblique",
        ),
    }


def _bullet_para(lead: str, rest: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(f"<b>&bull; {lead}</b> {rest}", style)


def _scaled(path: Path, max_w: float, max_h: float) -> Image:
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    scale = min(max_w / iw, max_h / ih)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    return img


def _band(text_top: str, text_meta: str, styles: dict[str, ParagraphStyle]) -> Table:
    """Top navy band that runs across the page."""
    page_w = landscape(A4)[0]
    inner = Table(
        [[Paragraph(text_top, styles["BandTitle"])],
         [Paragraph(text_meta, styles["BandMeta"])]],
        colWidths=[page_w - 1.6 * cm],
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
    ]))
    return inner


def _two_column(left: list[Any], right: list[Any], left_w: float, right_w: float,
                row_padding: int = 4) -> Table:
    """Lay two flowable stacks side-by-side via a single-row table."""
    tbl = Table([[left, right]], colWidths=[left_w, right_w])
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), row_padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), row_padding),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return tbl


def build() -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story: list[Any] = []

    # ── SLIDE 1: HEADLINE ─────────────────────────────────────────
    story.append(_band(
        "Forecasting Hourly Household Energy Consumption",
        "LUISS Artificial Intelligence Techniques &nbsp;&middot;&nbsp; "
        "Industry partner: Enel Global ICT &nbsp;&middot;&nbsp; "
        "Thierry Ishimwe, Linda Carla Zorzoli, Mariavittoria Giurato, Cesar Dushimimana",
        styles,
    ))
    story.append(Spacer(1, 0.4 * cm))

    # Hero column (left) + leaderboard image (right)
    hero_block = [
        Paragraph("MAE = 0.336 kW", styles["Hero"]),
        Paragraph("Cross-fold mean, XGBoost (leakage-safe pipeline)", styles["HeroSub"]),
        Paragraph(
            "Four model families (SARIMA, XGBoost, LSTM, GRU) compared on a uniform "
            "rolling-origin protocol with five metrics. Same-timestamp UCI columns "
            "(<i>Voltage</i>, <i>Global_intensity</i>, sub-metering) are mechanically "
            "excluded from the feature matrix; short-gap imputation is past-only. The "
            "pipeline ships with two regression tests guarding both guarantees.",
            styles["Body"],
        ),
    ]
    right_image = [_scaled(FIGURES / "leaderboard_bar.png",
                           max_w=14 * cm, max_h=10 * cm)]
    story.append(_two_column(hero_block, right_image, left_w=12.5 * cm, right_w=14.5 * cm))

    story.append(Spacer(1, 0.4 * cm))
    # Findings strip across the bottom of slide 1
    findings_block = [
        Paragraph("Headline findings", styles["SectionTitle"]),
        _bullet_para(
            "Top three are a statistical tie.",
            "XGBoost 0.336, GRU 0.339, LSTM 0.341, SARIMA 0.345. Diebold&ndash;Mariano: "
            "XGBoost vs SARIMA (p=0.022) is the only significant pair at &alpha;=0.05.",
            styles["Bullet"],
        ),
        _bullet_para(
            "GRU competitive with LSTM at lower complexity.",
            "Same MAE within noise; the simpler GRU cell captures sequential structure "
            "as well or better at this resolution.",
            styles["Bullet"],
        ),
        _bullet_para(
            "Weather augmentation validates the future-work priority.",
            "Adding 4 Paris&ndash;Montsouris weather columns to XGBoost: &minus;0.77% MAE, "
            "concentrated on heating-season folds.",
            styles["Bullet"],
        ),
        _bullet_para(
            "Leakage-safe by construction.",
            "Registry blocks same-timestamp components of the target; past-only imputation; "
            "two regression tests guard the policy.",
            styles["Bullet"],
        ),
    ]
    story.extend(findings_block)

    story.append(PageBreak())

    # ── SLIDE 2: IMPLICATIONS + FUTURE WORK ────────────────────────
    story.append(_band(
        "Where Models Win, Where They Lose, and What's Next",
        "Per-segment performance, operational implications, and the priority queue",
        styles,
    ))
    story.append(Spacer(1, 0.3 * cm))

    # Left = per-segment figure + caption
    segment_block: list[Any] = [
        _scaled(FIGURES / "segment_mae_by_dimension.png", max_w=14.0 * cm, max_h=14 * cm),
        Paragraph(
            "MAE by hour-of-day, day-of-week, and validation month. "
            "Peak hours and October's heating-ramp dominate the error budget.",
            styles["Caption"],
        ),
    ]

    # Right = implications + future work
    right_block: list[Any] = [
        Paragraph("Operational implications", styles["SectionTitle"]),
        _bullet_para(
            "Peak-hour MAE is ~2.5&times; trough-hour MAE.",
            "XGBoost wins quiet hours; GRU wins peaks. For peak-cost-weighted operations, "
            "GRU may be the better deployment choice.",
            styles["Bullet"],
        ),
        _bullet_para(
            "August easiest, October hardest.",
            "August = <i>grandes vacances</i> (empty household, flat load). October = "
            "regulated <i>p&eacute;riode de chauffe</i> (heating ramp, MAE ~2&times; higher).",
            styles["Bullet"],
        ),
        _bullet_para(
            "Stacking does NOT break the 3% cluster.",
            "Base-model errors are too correlated. The bottleneck is input information, "
            "not model variety.",
            styles["Bullet"],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("Future work", styles["SectionTitle"]),
        _bullet_para(
            "Extend weather to SARIMA / LSTM / GRU.",
            "The &minus;0.77% gain on XGBoost is a lower bound. ~70 min DL retrain on the "
            "already-built augmented matrix.",
            styles["Bullet"],
        ),
        _bullet_para(
            "24-step-ahead day-ahead forecasting.",
            "Current results are 1-step-ahead at hourly resolution. Production at Enel "
            "uses 24-step horizons made once per day from a fixed cutoff.",
            styles["Bullet"],
        ),
    ]

    story.append(_two_column(segment_block, right_block, left_w=14.5 * cm, right_w=12.5 * cm))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "github.com/ThierryIshimwe/Energy-Forecasting &nbsp;&middot;&nbsp; "
        "Brief: reports/Project_Work_Luiss_AI_Techniques_200326.pdf &nbsp;&middot;&nbsp; "
        "Full analysis: notebooks/00_main_deliverable.ipynb",
        styles["Footer"],
    ))

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=landscape(A4),
        leftMargin=0.8 * cm,
        rightMargin=0.8 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
        title="Energy Forecasting — Executive Summary",
        author="Thierry Ishimwe, Linda Carla Zorzoli, Mariavittoria Giurato, Cesar Dushimimana",
    )
    doc.build(story)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    build()
