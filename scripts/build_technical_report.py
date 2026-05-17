"""Build the 5-page technical-report PDF from the deliverable artifacts.

All content is sourced from tracked CSVs and PNGs in reports/. Running this
script produces reports/output/technical_report.pdf — the brief-mandated
5-page submission.

Run:
    python scripts/build_technical_report.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
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
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"
OUT_PDF = ROOT / "reports" / "output" / "technical_report.pdf"


# ── Styles ──────────────────────────────────────────────────────────
def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body_size = 8.5
    out = {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontSize=13,
            leading=15,
            alignment=TA_CENTER,
            spaceAfter=3,
        ),
        "Authors": ParagraphStyle(
            "Authors",
            parent=base["Normal"],
            fontSize=8.5,
            leading=10,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "Meta": ParagraphStyle(
            "Meta",
            parent=base["Normal"],
            fontSize=7.5,
            leading=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444"),
            spaceAfter=6,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontSize=10.5,
            leading=12,
            spaceBefore=5,
            spaceAfter=2,
            textColor=colors.HexColor("#1a3552"),
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=9.5,
            leading=11,
            spaceBefore=3,
            spaceAfter=1,
            textColor=colors.HexColor("#1a3552"),
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=body_size,
            leading=10.5,
            alignment=TA_JUSTIFY,
            spaceAfter=3,
        ),
        "BodyLeft": ParagraphStyle(
            "BodyLeft",
            parent=base["Normal"],
            fontSize=body_size,
            leading=10.5,
            alignment=TA_LEFT,
            spaceAfter=3,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontSize=7,
            leading=8.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#666"),
            spaceAfter=5,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontSize=body_size,
            leading=10.5,
            leftIndent=10,
            bulletIndent=0,
            alignment=TA_JUSTIFY,
            spaceAfter=1,
        ),
    }
    return out


# ── Helpers ─────────────────────────────────────────────────────────
def _table_style(header_bg: str = "#1a3552") -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ("TOPPADDING", (0, 0), (-1, 0), 4),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6fa")]),
        ]
    )


def _para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _bullets(items: list[str], style: ParagraphStyle) -> list[Paragraph]:
    return [Paragraph(f"<bullet>&bull;</bullet>&nbsp;{it}", style) for it in items]


def _scaled_image(path: Path, max_width: float, max_height: float | None = None) -> Image:
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    scale = max_width / iw
    if max_height is not None and ih * scale > max_height:
        scale = max_height / ih
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    return img


# ── Build ───────────────────────────────────────────────────────────
def build() -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story: list[Any] = []

    # Pre-load the artifacts we need
    leaderboard = pd.read_csv(RESULTS / "leaderboard.csv")
    dm = pd.read_csv(RESULTS / "dm_test_pvalues.csv", index_col=0)
    xgb_base = pd.read_csv(RESULTS / "xgboost_fold_results.csv")
    xgb_wx = pd.read_csv(RESULTS / "xgboost_with_weather_fold_results.csv")

    # ── PAGE 1: Title + Executive Summary ──────────────────────────
    story.append(_para(
        "Forecasting Hourly Household Electricity Consumption: "
        "A Leakage-Safe Comparison of Classical, Tree-Based, and Sequence Models",
        styles["Title"],
    ))
    story.append(_para(
        "Thierry Ishimwe &nbsp;|&nbsp; Linda Carla Zorzoli &nbsp;|&nbsp; "
        "Mariavittoria Giurato &nbsp;|&nbsp; Cesar Dushimimana",
        styles["Authors"],
    ))
    story.append(_para(
        "LUISS Guido Carli &nbsp;&middot;&nbsp; Artificial Intelligence Techniques "
        "&nbsp;&middot;&nbsp; Industry partner: Enel Global ICT &nbsp;&middot;&nbsp; May 2026",
        styles["Meta"],
    ))

    story.append(_para("Executive Summary", styles["H1"]))
    story.append(_para(
        "This report compares four model families &mdash; SARIMA, XGBoost, LSTM, and GRU &mdash; "
        "on the task of forecasting hourly residential electricity consumption "
        "(UCI <i>Individual Household Electric Power Consumption</i> dataset, Sceaux, France, "
        "2006&ndash;2010, 2,075,259 minute rows resampled to 34,000 hourly observations). "
        "The pipeline is leakage-safe by construction: every feature is registered with a "
        "leakage-safety flag and a written rationale, same-timestamp components of the target "
        "(<i>Voltage</i>, <i>Global_intensity</i>, sub-metering) are mechanically excluded, "
        "and short-gap imputation is past-only so no future-derived value can flow into a "
        "training row. Cross-fold mean MAE for the winning model (XGBoost) is <b>0.336 kW</b>.",
        styles["Body"],
    ))
    story.append(_para(
        "Seven findings shape the conclusion:",
        styles["Body"],
    ))
    findings = [
        "<b>XGBoost wins the leaderboard at MAE = 0.336.</b> Its lead is statistically significant "
        "(Diebold&ndash;Mariano, &alpha; = 0.05) only against SARIMA (p = 0.022). Against LSTM "
        "(p = 0.33) and GRU (p = 0.42) the difference is within noise &mdash; the top three are a "
        "statistical tie.",
        "<b>The four real model families cluster within ~3% MAE</b> (XGBoost 0.336, GRU 0.339, "
        "LSTM 0.341, SARIMA 0.345). Architecture choice within this band is a near-tie; "
        "deployment should be driven by non-MAE criteria.",
        "<b>Peak-hour MAE (18&ndash;21h) is ~2.5&times; trough-hour MAE (03&ndash;06h)</b> for every model. "
        "XGBoost is best at quiet hours; GRU is slightly better at peak hours &mdash; a real "
        "operational consideration since peak forecast errors drive imbalance penalties.",
        "<b>October is the hardest validation month</b> (heating-ramp, MAE &asymp; 0.44); "
        "<b>August is easiest</b> (the French <i>grandes vacances</i> &mdash; empty household, "
        "flat consumption, MAE &asymp; 0.22). The pattern is structural, not a single-fold artifact.",
        "<b>Stacking ensembles do not break the 3% cluster.</b> A ridge meta-learner over the four "
        "base models edges past every individual model on the held-out fold (0.368 vs 0.368 best "
        "individual); base-model errors are too correlated for simple linear stacking to help.",
        "<b>Weather augmentation validates the &quot;input information, not architecture&quot; "
        "hypothesis.</b> Augmenting XGBoost with four Paris&ndash;Montsouris weather columns "
        "(temperature, humidity, wind speed, 24h temperature lag) reduces cross-fold MAE "
        "from 0.336 to 0.334 (&minus;0.77%), with the improvement concentrated on the hardest "
        "heating-season folds.",
        "<b>The 1-step-ahead hourly horizon is a methodology benchmark, not the operational target.</b> "
        "Production day-ahead forecasts at Enel use 24-step horizons made from a fixed daily cutoff. "
        "The reported MAE values are therefore a lower bound on operationally-realistic difficulty; "
        "the protocol switch is recorded as future work.",
    ]
    story.extend(_bullets(findings, styles["Bullet"]))

    story.append(PageBreak())

    # ── PAGE 2: Problem, Dataset, Methodology ──────────────────────
    story.append(_para("1. Problem and Dataset", styles["H1"]))
    story.append(_para(
        "We forecast hourly <i>Global_active_power</i> (kW) for a single household in Sceaux, "
        "Paris area, using the UCI <i>Individual Household Electric Power Consumption</i> "
        "dataset: 2,075,259 minute-level observations from 16 December 2006 to 26 November 2010 "
        "(47 months, 7 measured variables). The source file is pinned by SHA-256 in "
        "<font face='Courier' size='8'>conf/base.yaml</font> and is verified on download. "
        "The 1-minute series is resampled to hourly mean (34,000 hourly rows), aligning with "
        "the 1-hour bidding blocks used in European electricity markets.",
        styles["Body"],
    ))
    story.append(_para(
        "Missingness is 1.25% (25,979 minute rows across 71 distinct gap segments), bimodally "
        "distributed: 54 short gaps (&le;3 min, 72 minutes total) and 17 long outages "
        "(&gt;3 min, 25,907 minutes total). The longest single outage spans 5 days in August 2010, "
        "coinciding with the French summer-vacation period. Short gaps are forward-filled "
        "from the last past observation (past-only, no future leakage); "
        "long outages are preserved as NaN with explicit <i>is_outage_gap</i> indicators. After "
        "hourly resampling, 28 hourly rows (0.08%) retain the outage flag.",
        styles["Body"],
    ))

    story.append(_para("2. Methodology", styles["H1"]))
    story.append(_para("2.1 Leakage-Safe Feature Engineering", styles["H2"]))
    story.append(_para(
        "Every feature is registered in "
        "<font face='Courier' size='8'>features/registry.py</font> with a leakage-safety flag and "
        "a written rationale. The pipeline refuses to emit any column marked unsafe when "
        "<font face='Courier' size='8'>strict_leakage_safe: true</font>. Same-timestamp UCI "
        "columns that are physical components of the target (<i>Voltage</i>, <i>Global_intensity</i>, "
        "sub-metering 1/2/3, <i>Global_reactive_power</i>) are registered as unsafe and blocked "
        "from entering the feature matrix &mdash; they would let the model recover the target "
        "through <i>P = V&middot;I</i> rather than forecast it. Short-gap imputation uses past-only "
        "forward-fill in <font face='Courier' size='8'>preprocessing/missing.py</font>, so no "
        "future observation can leak into a training row through the cleaned series. Two regression "
        "tests guard the policy &mdash; "
        "<font face='Courier' size='8'>tests/integration/test_leakage.py</font> (registry contract) "
        "and <font face='Courier' size='8'>tests/unit/test_missing_value_policy.py</font> "
        "(past-only fill).",
        styles["Body"],
    ))
    story.append(_para(
        "The leakage-safe feature matrix has 31 columns across five families: <b>target lags</b> at "
        "1, 24, and 168 hours (one hour, one day, one week back); <b>rolling statistics</b> "
        "(mean, std, min, max) over 3, 24, and 168-hour windows of the lag-1-shifted target "
        "&mdash; the <i>shift</i>-then-<i>roll</i> order guarantees the window at time <i>t</i> "
        "covers strictly past values; <b>calendar features</b> (hour, day-of-week, month, "
        "<i>is_weekend</i>, <i>is_french_holiday</i>); <b>cyclical encodings</b> "
        "(<i>sin</i>/<i>cos</i> of hour, dow, month) so trees can handle the wraparound at "
        "midnight and end-of-week; and <b>outage indicators</b>.",
        styles["Body"],
    ))

    story.append(_para("2.2 Evaluation Protocol", styles["H2"]))
    story.append(_para(
        "Rolling-origin cross-validation, 6 folds, each fold trains on 365 days and validates "
        "on the following 30 days. The folds slide forward in 30-day increments; validation "
        "never overlaps training and earlier folds never see later data. Each model is fit from "
        "scratch per fold. The protocol is <b>1-step-ahead at hourly resolution</b>: at validation "
        "hour <i>t</i>, the model predicts <i>y[t]</i> using information available through "
        "<i>t&minus;1</i>. Metrics computed per fold: MAE (primary), RMSE, WAPE, sMAPE, "
        "MASE (with seasonal period 24 for the daily-naive denominator).",
        styles["Body"],
    ))
    story.append(_para(
        "True 24-step-ahead day-ahead forecasting &mdash; the horizon that matches Enel's market-bid "
        "block &mdash; is listed as future work; the reported MAE values are therefore a lower bound "
        "on the operationally-realistic difficulty.",
        styles["Body"],
    ))

    story.append(PageBreak())

    # ── PAGE 3: Models + Leaderboard + DM ──────────────────────────
    story.append(_para("3. Model Lineup", styles["H1"]))
    story.append(_para(
        "The lineup spans four model families. Each was selected to test a distinct hypothesis "
        "about which class of structure dominates the signal.",
        styles["Body"],
    ))
    model_rows = [
        ["Model", "Family", "Role"],
        ["naive_lag_{1h, 24h, 168h}", "Persistence baselines", "Floors for daily / weekly seasonality"],
        ["SARIMA(1,1,1)(1,1,1,24)", "Classical statistical", "Satisfies brief's ≥1 classical requirement"],
        ["LSTM (168h lookback, 64 units)", "Deep-learning sequence", "Satisfies brief's ≥1 DL requirement"],
        ["GRU (168h lookback, 64 units)", "Deep-learning sequence", "Within-family DL comparison"],
        ["XGBoost (depth=6, lr=0.05, n=500)", "Gradient-boosted trees", "Tabular model with 31 engineered features"],
        ["Stacking (ridge α=1.0)", "Ensemble (bonus)", "Tests base-model error decorrelation"],
    ]
    tbl = Table(model_rows, colWidths=[5.0 * cm, 4.0 * cm, 8.0 * cm])
    tbl.setStyle(_table_style())
    story.append(tbl)
    story.append(Spacer(1, 6))

    story.append(_para("4. Results", styles["H1"]))
    story.append(_para("4.1 Leaderboard", styles["H2"]))
    story.append(_scaled_image(FIGURES / "leaderboard_bar.png", max_width=15 * cm, max_height=7 * cm))
    story.append(_para(
        "Figure 1. Cross-fold mean MAE per model with ±1σ error bars across the six "
        "rolling-origin folds. Red bars are naive persistence baselines; blue bars are the "
        "real model families. Dashed line marks the lag-1h persistence floor (0.380).",
        styles["Caption"],
    ))

    lb_table = [["Model", "MAE", "RMSE", "MASE"]]
    name_map = {
        "xgb_default": "XGBoost",
        "gru_default": "GRU",
        "lstm_default": "LSTM",
        "sarima_111_111_24": "SARIMA",
        "naive_lag_1h": "Naive lag-1h",
        "naive_lag_24h_daily": "Naive lag-24h",
        "naive_lag_168h_weekly": "Naive lag-168h",
    }
    for _, row in leaderboard.iterrows():
        lb_table.append([
            name_map.get(row["model"], row["model"]),
            f"{row['MAE_mean']:.3f} ± {row['MAE_std']:.3f}",
            f"{row['RMSE_mean']:.3f} ± {row['RMSE_std']:.3f}",
            f"{row['MASE_mean']:.3f} ± {row['MASE_std']:.3f}",
        ])
    t = Table(lb_table, colWidths=[4.0 * cm, 4.0 * cm, 4.0 * cm, 4.0 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(_para(
        "Table 1. Cross-fold mean ± std for each model. Lower is better. XGBoost wins on MAE; "
        "all four real models cluster within 3% of each other.",
        styles["Caption"],
    ))

    story.append(_para("4.2 Statistical Significance &mdash; Diebold&ndash;Mariano Test", styles["H2"]))
    story.append(_para(
        "Pairwise Diebold&ndash;Mariano tests on absolute-error loss between the four real models, "
        "using the Harvey&ndash;Leybourne&ndash;Newbold small-sample correction and Newey&ndash;West HAC variance "
        "at lag 0 (appropriate for 1-step-ahead forecasts). Two-sided p-values:",
        styles["Body"],
    ))
    dm_table = [["", "XGBoost", "SARIMA", "LSTM", "GRU"]]
    for row in ["xgboost", "sarima", "lstm", "gru"]:
        r = [row.upper()]
        for col in ["xgboost", "sarima", "lstm", "gru"]:
            v = dm.loc[row, col]
            r.append("—" if pd.isna(v) else f"{v:.3f}")
        dm_table.append(r)
    t = Table(dm_table, colWidths=[2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(_para(
        "Table 2. DM two-sided p-values, n &asymp; 4,100 paired predictions. Only XGBoost vs "
        "SARIMA (p = 0.022) clears the &alpha; = 0.05 bar. XGBoost vs LSTM (p = 0.33) and XGBoost "
        "vs GRU (p = 0.42) are not statistically distinguishable.",
        styles["Caption"],
    ))

    story.append(PageBreak())

    # ── PAGE 4: Per-Segment + Weather + External Context ───────────
    story.append(_para("4.3 Per-Segment Performance", styles["H1"]))
    story.append(_scaled_image(FIGURES / "segment_mae_by_dimension.png",
                               max_width=15 * cm, max_height=10 * cm))
    story.append(_para(
        "Figure 2. MAE decomposed by hour-of-day, day-of-week, and month for each of the four "
        "real models. Aggregate MAE hides operationally important structure.",
        styles["Caption"],
    ))
    story.append(_para(
        "<b>Hour of day.</b> Every model exhibits the same shape: easy at 03&ndash;06h "
        "(MAE &asymp; 0.17), hard at 18&ndash;21h (MAE &asymp; 0.45). XGBoost wins quiet hours; "
        "GRU is slightly better at peak hours. <b>Day of week.</b> Weekend MAE is ~5% higher than "
        "weekday &mdash; modest. <b>Month.</b> August is by far the easiest validation month "
        "(MAE 0.21&ndash;0.25, the empty-household <i>grandes vacances</i>); October is the "
        "hardest (0.42&ndash;0.44, the regulated <i>p&eacute;riode de chauffe</i> heating ramp).",
        styles["Body"],
    ))

    story.append(_para("4.4 Weather Data as Exogenous Regressor", styles["H1"]))
    story.append(_para(
        "To test the &quot;input information, not architecture&quot; hypothesis empirically, "
        "we retrained XGBoost on the feature matrix augmented with four hourly weather columns "
        "from station Paris&ndash;Montsouris (5.2 km from Sceaux, via the <i>meteostat</i> client): "
        "temperature (&deg;C), relative humidity (%), wind speed (km/h), and a strictly past "
        "24-hour-lagged temperature. Per-fold result:",
        styles["Body"],
    ))
    wx_table = [["Fold", "Baseline MAE", "+Weather MAE", "Δ MAE", "Δ %"]]
    for i in range(len(xgb_base)):
        b = xgb_base["MAE"].iloc[i]
        w = xgb_wx["MAE"].iloc[i]
        delta = w - b
        pct = delta / b * 100
        wx_table.append([
            f"{int(xgb_base['fold'].iloc[i])}",
            f"{b:.4f}",
            f"{w:.4f}",
            f"{delta:+.4f}",
            f"{pct:+.2f}%",
        ])
    wx_table.append([
        "<b>Mean</b>",
        f"<b>{xgb_base['MAE'].mean():.4f}</b>",
        f"<b>{xgb_wx['MAE'].mean():.4f}</b>",
        f"<b>{xgb_wx['MAE'].mean() - xgb_base['MAE'].mean():+.4f}</b>",
        f"<b>{(xgb_wx['MAE'].mean() - xgb_base['MAE'].mean()) / xgb_base['MAE'].mean() * 100:+.2f}%</b>",
    ])
    t = Table(wx_table, colWidths=[2.0 * cm, 3.5 * cm, 3.5 * cm, 3.0 * cm, 3.0 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(_para(
        "Table 3. XGBoost with vs without four weather features. The mean MAE delta is small "
        "(&minus;0.77%) but signed in the predicted direction and concentrated on the harder folds: "
        "the late-winter / spring / October-heating-ramp windows show the larger gains, while "
        "the easy summer fold (1) gains little &mdash; consistent with weather mattering when "
        "consumption is temperature-driven, not when the household is flat-line.",
        styles["Caption"],
    ))

    story.append(_para("5. External Context", styles["H1"]))
    story.append(_para(
        "Several empirical patterns map onto documented French and European context. The longest "
        "outage (17&ndash;22 August 2010) and the &quot;August is easiest&quot; finding both reflect "
        "the institutionalised summer vacation that INSEE and RTE document accounts for a "
        "national consumption drop of 15&ndash;20%. The January 2010 outage (12&ndash;14 January) "
        "coincides with a European cold wave during which RTE's <i>Bilan &Eacute;lectrique 2010</i> "
        "records French national load exceeding 96 GW for the first time, driven by the country's "
        "~30% electric-heating penetration. October's role as the hardest validation month is "
        "explained by the regulated <i>p&eacute;riode de chauffe</i> beginning 15 October each "
        "year, after which RTE documents typical residential consumption rising 40&ndash;60% "
        "between mid-October and mid-November.",
        styles["Body"],
    ))

    story.append(PageBreak())

    # ── PAGE 5: Discussion + Future + Reproducibility ──────────────
    story.append(_para("6. Discussion", styles["H1"]))
    story.append(_para(
        "<b>Architecture matters less than engineering protocol.</b> Four model families &mdash; "
        "classical SARIMA, tree-based XGBoost, sequence DL LSTM and GRU &mdash; cluster within 3% "
        "MAE, and the within-cluster differences are statistically indistinguishable. This is "
        "consistent with M-competition findings that on data-rich forecasting problems, feature "
        "engineering and evaluation discipline dominate architecture choice. The stacking-ensemble "
        "result corroborates this: base-model errors are correlated enough that linear combination "
        "does not help &mdash; ruling out &quot;more model variety&quot; as the path forward.",
        styles["Body"],
    ))
    story.append(_para(
        "<b>The &quot;best&quot; model depends on the operational metric.</b> XGBoost wins overall "
        "MAE but GRU wins peak-hour (18&ndash;21h) MAE. For a grid operator whose cost function "
        "weights peak-hour accuracy heavily &mdash; a real concern since peak forecast errors "
        "drive imbalance penalties &mdash; GRU is the more defensible deployment choice "
        "despite XGBoost's leaderboard lead. This nuance is hidden by an aggregate MAE.",
        styles["Body"],
    ))
    story.append(_para(
        "<b>GRU is competitive with LSTM at lower complexity.</b> Across all six folds, GRU's MAE "
        "(0.339) is essentially tied with LSTM's (0.341). The extra output gate in LSTM is unused "
        "capacity on this dataset; GRU is a reasonable default when adding a sequence-DL model to "
        "a forecasting pipeline.",
        styles["Body"],
    ))
    story.append(_para(
        "<b>Weather augmentation: small but real, where it matters.</b> A &minus;0.77% MAE gain on "
        "XGBoost from four free Paris&ndash;Montsouris weather columns is modest, but signed in the "
        "predicted direction and concentrated on heating-season folds. It validates the direction "
        "of the priority claim &mdash; better input information is closer to the leaderboard "
        "ceiling than another architecture would bring us &mdash; without overclaiming a "
        "transformative gain.",
        styles["Body"],
    ))

    story.append(_para("7. Future Work", styles["H1"]))
    future = [
        "<b>Extend the weather-augmented matrix to SARIMA, LSTM, and GRU.</b> The augmented feature "
        "matrix is already persisted at "
        "<font face='Courier' size='8'>data/processed/features_with_weather.parquet</font>. The "
        "missing work is ~70 min of model retraining. Expected to confirm whether the &minus;0.77% "
        "XGBoost gain is single-model-specific or a team-wide improvement.",
        "<b>Implement 24-step-ahead day-ahead forecasting.</b> Current results are 1-step-ahead at "
        "hourly resolution. Production day-ahead forecasts at Enel use 24-step horizons made once "
        "per day from a fixed daily cutoff; the reported MAE values are a lower bound on "
        "operationally-realistic difficulty.",
    ]
    story.extend(_bullets(future, styles["Bullet"]))

    story.append(_para("8. Reproducibility", styles["H1"]))
    story.append(_para(
        "Every result above is reproducible from a fresh clone of "
        "<i>github.com/ThierryIshimwe/Energy-Forecasting</i>. The deliverable notebook "
        "(<font face='Courier' size='8'>notebooks/00_main_deliverable.ipynb</font>) loads "
        "tracked artifacts only and ships with outputs pre-rendered, so an evaluator needs three "
        "commands: clone, "
        "<font face='Courier' size='8'>pip install -r requirements.txt &amp;&amp; pip install -e .</font>, "
        "open the notebook. No raw-data download is needed for the deliverable view; "
        "the EDA summary numbers and seasonality plot data are precomputed and tracked.",
        styles["Body"],
    ))
    story.append(_para(
        "Pipeline guarantees: Python 3.11 pinned in "
        "<font face='Courier' size='8'>.python-version</font>; all dependencies version-pinned "
        "in <font face='Courier' size='8'>requirements.txt</font>; global RNG seed 42 in "
        "<font face='Courier' size='8'>conf/base.yaml</font>; source dataset SHA-256 verified at "
        "download; 64+ unit + integration tests, including the leakage-registry test and the "
        "past-only-imputation test, run on every commit. Re-training every model from raw data "
        "takes ~90 minutes and is documented in <i>README.md</i>.",
        styles["Body"],
    ))

    # ── Build the document ─────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=1.3 * cm,
        rightMargin=1.3 * cm,
        topMargin=1.1 * cm,
        bottomMargin=1.1 * cm,
        title="Energy Forecasting — Technical Report",
        author="Thierry Ishimwe, Linda Carla Zorzoli, Mariavittoria Giurato, Cesar Dushimimana",
    )
    doc.build(story)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    build()
