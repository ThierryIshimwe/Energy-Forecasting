"""Build the technical-report PDF in the structure required by the project rules.

Structure (per the rules document):
    Title page (1 page)
    Section 1: Introduction
    Section 2: Methods (with subsections)
    Section 3: Results and Discussion (technical + business-value)
    Section 4: Conclusions
    Appendix A: Code Description (max 1 page, pseudocode/flowchart)
    Appendix B: Author Contribution + GenAI Statement (max 1 page)

Output: reports/output/technical_report.pdf
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
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
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"
OUT_PDF = ROOT / "reports" / "output" / "technical_report.pdf"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = 9
    return {
        "TitleBig": ParagraphStyle("TitleBig", parent=base["Title"],
            fontSize=22, leading=26, alignment=TA_CENTER, spaceAfter=20,
            textColor=colors.HexColor("#1a3552")),
        "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"],
            fontSize=12, leading=14, alignment=TA_CENTER, spaceAfter=30,
            textColor=colors.HexColor("#444")),
        "Authors": ParagraphStyle("Authors", parent=base["Normal"],
            fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=4),
        "Affil": ParagraphStyle("Affil", parent=base["Normal"],
            fontSize=10, leading=12, alignment=TA_CENTER, spaceAfter=4,
            textColor=colors.HexColor("#444")),
        "H1": ParagraphStyle("H1", parent=base["Heading1"],
            fontSize=12, leading=14, spaceBefore=10, spaceAfter=4,
            textColor=colors.HexColor("#1a3552")),
        "H2": ParagraphStyle("H2", parent=base["Heading2"],
            fontSize=10.5, leading=12, spaceBefore=5, spaceAfter=2,
            textColor=colors.HexColor("#1a3552")),
        "Body": ParagraphStyle("Body", parent=base["Normal"],
            fontSize=body, leading=11.5, alignment=TA_JUSTIFY, spaceAfter=4),
        "Caption": ParagraphStyle("Caption", parent=base["Normal"],
            fontSize=7.5, leading=9, alignment=TA_CENTER,
            textColor=colors.HexColor("#666"), spaceAfter=6),
        "Bullet": ParagraphStyle("Bullet", parent=base["Normal"],
            fontSize=body, leading=11.5, leftIndent=10, spaceAfter=2,
            alignment=TA_JUSTIFY),
        "Mono": ParagraphStyle("Mono", parent=base["Normal"],
            fontName="Courier", fontSize=8, leading=10, spaceAfter=4),
        "Draft": ParagraphStyle("Draft", parent=base["Normal"],
            fontSize=8.5, leading=11, alignment=TA_LEFT,
            textColor=colors.HexColor("#a04040"), spaceAfter=4,
            fontName="Helvetica-Oblique"),
    }


def _table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3552")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f3f6fa")]),
    ])


def _para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _bullets(items: list[str], style: ParagraphStyle) -> list[Paragraph]:
    return [Paragraph(f"<bullet>&bull;</bullet>&nbsp;{it}", style) for it in items]


def _scaled(path: Path, max_w: float, max_h: float) -> Image:
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    scale = min(max_w / iw, max_h / ih)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    return img


def build() -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    s = _styles()
    story: list[Any] = []

    lb = pd.read_csv(RESULTS / "leaderboard.csv")
    dm = pd.read_csv(RESULTS / "dm_test_pvalues.csv", index_col=0)
    base = pd.read_csv(RESULTS / "xgboost_fold_results.csv")
    wx = pd.read_csv(RESULTS / "xgboost_with_weather_fold_results.csv")

    # ── TITLE PAGE ─────────────────────────────────────────────────
    story.append(Spacer(1, 4 * cm))
    story.append(_para("Forecasting Hourly Household Electricity Consumption", s["TitleBig"]))
    story.append(_para("A Leakage-Safe Comparison of Classical, Tree-Based, and Sequence Models", s["Subtitle"]))
    story.append(Spacer(1, 1.5 * cm))
    story.append(_para("<b>Group members</b>", s["Affil"]))
    story.append(_para(
        "Thierry Ishimwe &nbsp;&middot;&nbsp; "
        "Linda Carla Zorzoli &nbsp;&middot;&nbsp; "
        "Mariavittoria Giurato &nbsp;&middot;&nbsp; "
        "Cesar Dushimimana",
        s["Authors"],
    ))
    story.append(Spacer(1, 1 * cm))
    story.append(_para("<b>Course</b><br/>Artificial Intelligence Techniques &mdash; LUISS Guido Carli",
                        s["Affil"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(_para("<b>Industry partner</b><br/>Enel Global ICT", s["Affil"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(_para("May 2026", s["Affil"]))
    story.append(PageBreak())

    # ── SECTION 1: INTRODUCTION ────────────────────────────────────
    story.append(_para("1. Introduction", s["H1"]))
    story.append(_para(
        "This project develops an hourly electricity-consumption forecasting "
        "pipeline for a single household and uses it to compare four model "
        "families on a uniform protocol. The case study is the UCI <i>Individual "
        "Household Electric Power Consumption</i> dataset: 2,075,259 minute-level "
        "active-power measurements from a household in Sceaux (Paris area), France, "
        "between December 2006 and November 2010, resampled to hourly mean for "
        "modelling (34,000 hourly observations). The goal is to forecast next-hour "
        "<i>Global_active_power</i> in a leakage-safe way and to evaluate four model "
        "families &mdash; SARIMA, XGBoost, LSTM, GRU &mdash; against three naive "
        "persistence baselines on six rolling-origin folds.",
        s["Body"]))
    story.append(_para(
        "The headline finding is that all four real model families cluster within "
        "a 3% MAE band (0.336&ndash;0.345), the top three (XGBoost, GRU, LSTM) are "
        "statistically indistinguishable under the Diebold&ndash;Mariano test, and "
        "augmenting the leaderboard winner with four free Paris&ndash;Montsouris "
        "weather columns reduces cross-fold MAE by 0.77% &mdash; concentrated on "
        "the heating-season folds where forecasting hurts most. For Enel, the "
        "implication is that on residential consumption at this resolution, the "
        "next gain comes from <i>better input information</i>, not from a different "
        "architecture.",
        s["Body"]))

    # ── SECTION 2: METHODS ─────────────────────────────────────────
    story.append(_para("2. Methods", s["H1"]))

    story.append(_para("2.1 Data", s["H2"]))
    story.append(_para(
        "The primary source is the UCI dataset cited above, SHA-256-pinned in the "
        "project configuration and integrity-verified on download. Weather data was "
        "collected separately from station <b>Paris&ndash;Montsouris</b> (NOAA / "
        "M&eacute;t&eacute;o-France via the <i>meteostat</i> client), 5.2 km from "
        "Sceaux &mdash; the closest publicly available station with continuous "
        "hourly records spanning the full study period. The original 1-minute "
        "series is resampled to hourly mean so the modelling frequency aligns with "
        "the 1-hour bidding blocks used in European electricity markets.",
        s["Body"]))

    story.append(_para("2.2 Pre-processing", s["H2"]))
    story.append(_para(
        "Raw missingness is 1.25% (25,979 minute rows across 71 distinct gap "
        "segments), bimodally distributed: 54 short gaps (&le;3 min, 72 minutes "
        "total) and 17 long outages (&gt;3 min, 25,907 minutes total). The longest "
        "single outage spans 5 days in August 2010, coinciding with the French "
        "summer-vacation period. <b>Short gaps are forward-filled</b> from the last "
        "past observation; <b>long outages are preserved as NaN</b> with explicit "
        "<i>is_outage_gap</i> indicators so models can see the outage rather than "
        "be silently fed an imputed value. Forward-only fill is essential: a "
        "bilateral interpolation would pull from future timestamps and leak "
        "information through downstream lag and rolling features.",
        s["Body"]))

    story.append(_para("2.3 Feature engineering", s["H2"]))
    story.append(_para(
        "The feature matrix is <b>leakage-safe by construction</b>. Every feature "
        "is registered with a <i>leakage_safe</i> flag and a written rationale; the "
        "pipeline refuses to emit any column flagged unsafe. Six UCI columns "
        "(<i>Voltage</i>, <i>Global_intensity</i>, <i>Global_reactive_power</i>, "
        "<i>Sub_metering_1/2/3</i>) are mechanically excluded because they are "
        "same-timestamp physical components of the target (<i>P = V&middot;I</i>; "
        "sub-meters sum into the total active power) &mdash; using them would let "
        "the model reconstruct the target rather than forecast it. The matrix has "
        "<b>31 leakage-safe features</b> in five families: target lags at 1, 24, "
        "168 h; rolling statistics (mean / std / min / max) over 3, 24, 168 h "
        "windows computed as <i>shift(1).rolling(W)</i> so the window covers "
        "strictly past values; calendar features (hour, day-of-week, month, "
        "<i>is_weekend</i>, <i>is_french_holiday</i>); cyclical sine/cosine "
        "encodings of the calendar dimensions; and outage indicators.",
        s["Body"]))

    story.append(_para("2.4 Models considered", s["H2"]))
    story.append(_para(
        "Eight models span four families. Each is selected to test a distinct "
        "hypothesis about which class of structure dominates the signal.",
        s["Body"]))
    model_rows = [
        ["Model", "Family", "Role"],
        ["3 naive baselines (lag-1h, -24h, -168h)", "Persistence", "Brief: baselines for comparison"],
        ["SARIMA(1,1,1)(1,1,1,24)", "Classical statistical", "Brief: ≥1 classical model"],
        ["LSTM (168h lookback, 64 units)", "Deep-learning sequence", "Brief: ≥1 DL model"],
        ["GRU (168h lookback, 64 units)", "Deep-learning sequence", "Additional: within-DL comparison"],
        ["XGBoost (depth=6, lr=0.05, n=500)", "Gradient-boosted trees", "Additional: 4th family, transparent feature importance"],
        ["Stacking (ridge α=1.0)", "Ensemble", "Additional: tests base-model error decorrelation"],
        ["XGBoost + 4 weather features", "Tree + exogenous", "Additional: tests input-information bottleneck"],
    ]
    t = Table(model_rows, colWidths=[5.6 * cm, 3.6 * cm, 7.4 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(_para("Table 1. Model lineup with role tags. <i>Brief</i> items satisfy the "
                       "project requirements directly; <i>Additional</i> items go beyond the "
                       "minimum with the value they add stated inline.", s["Caption"]))

    story.append(_para("2.5 Hyperparameter choices", s["H2"]))
    story.append(_para(
        "Hyperparameters are documented defaults consistent with the published "
        "literature for each model family: SARIMA order chosen from ADF / KPSS "
        "stationarity tests (<i>d=1</i>) plus STL decomposition (<i>m=24</i>); "
        "XGBoost depth 6, learning rate 0.05, 500 estimators; LSTM and GRU with a "
        "single recurrent layer of 64 hidden units over a 168-hour lookback, Adam "
        "optimizer, 25 epochs. Each fold trains a fresh model with these settings "
        "&mdash; no global tuning loop runs across folds (which would introduce "
        "cross-fold information leakage). Within-fold validation is implicit in "
        "the 365-day training window.",
        s["Body"]))

    story.append(_para("2.6 Evaluation protocol", s["H2"]))
    story.append(_para(
        "<b>Rolling-origin cross-validation</b>, 6 folds, each fold trains on 365 "
        "days and validates on the following 30 days. Folds slide forward in 30-day "
        "increments; validation never overlaps training and earlier folds never see "
        "later data. Each model is fit from scratch per fold. The protocol is "
        "<b>1-step-ahead at hourly resolution</b>: at validation hour <i>t</i>, the "
        "model predicts <i>y[t]</i> using information available through <i>t&minus;1</i>. "
        "Metrics: MAE (primary, robust to the right-skewed target), RMSE (peak-error "
        "guardrail), WAPE, sMAPE, and MASE with seasonal period 24 (a MASE below 1 "
        "means the model beats the in-sample daily-naive baseline).",
        s["Body"]))

    story.append(_para("2.7 Baselines", s["H2"]))
    story.append(_para(
        "Three persistence baselines anchor the leaderboard. <i>naive_lag_1h</i> "
        "predicts <i>y[t] = y[t&minus;1]</i> (hour-to-hour persistence); "
        "<i>naive_lag_24h</i> predicts yesterday-at-this-hour; <i>naive_lag_168h</i> "
        "predicts last-week-at-this-hour. They establish the floor any real model "
        "must beat to deserve a place in the report.",
        s["Body"]))
    story.append(PageBreak())

    # ── SECTION 3: RESULTS AND DISCUSSION ──────────────────────────
    story.append(_para("3. Results and Discussion", s["H1"]))

    story.append(_para("3.1 Technical results", s["H2"]))
    story.append(_scaled(FIGURES / "leaderboard_bar.png", 15 * cm, 6.5 * cm))
    story.append(_para("Figure 1. Cross-fold mean MAE per model with &plusmn;1&sigma; error "
                       "bars across the six rolling-origin folds. Red bars are persistence "
                       "baselines; blue bars are the real model families.", s["Caption"]))

    lb_rows = [["Model", "MAE", "RMSE", "MASE"]]
    name = {"xgb_default": "XGBoost", "gru_default": "GRU", "lstm_default": "LSTM",
            "sarima_111_111_24": "SARIMA",
            "naive_lag_1h": "Naive lag-1h",
            "naive_lag_24h_daily": "Naive lag-24h",
            "naive_lag_168h_weekly": "Naive lag-168h"}
    for _, r in lb.iterrows():
        lb_rows.append([name.get(r["model"], r["model"]),
                        f"{r['MAE_mean']:.3f} ± {r['MAE_std']:.3f}",
                        f"{r['RMSE_mean']:.3f} ± {r['RMSE_std']:.3f}",
                        f"{r['MASE_mean']:.3f} ± {r['MASE_std']:.3f}"])
    t = Table(lb_rows, colWidths=[4 * cm, 4 * cm, 4 * cm, 4 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(_para("Table 2. Leaderboard, cross-fold mean &plusmn; std. The four real "
                       "model families cluster within 3% MAE.", s["Caption"]))

    story.append(_para(
        "<i>(Additional: Diebold&ndash;Mariano significance test.)</i> A pairwise "
        "Diebold&ndash;Mariano test with the Harvey&ndash;Leybourne&ndash;Newbold "
        "small-sample correction was run between the four real model families on "
        "<i>n</i> &asymp; 4,100 paired predictions. Only <b>XGBoost vs SARIMA "
        "(p = 0.022)</b> clears the &alpha; = 0.05 bar. XGBoost vs LSTM (p = 0.33) "
        "and XGBoost vs GRU (p = 0.42) are not statistically distinguishable &mdash; "
        "the top three are a statistical tie. The added value of the DM test is "
        "that it answers a question MAE alone cannot: whether the leaderboard order "
        "is real or within noise.",
        s["Body"]))

    story.append(_para(
        "<i>(Additional: per-segment analysis.)</i> MAE decomposed by hour-of-day, "
        "day-of-week, and validation month surfaces operationally important "
        "structure that aggregate MAE hides. Every model shares the same hour-of-day "
        "shape (easy at 03&ndash;06h with MAE &asymp; 0.17, hard at 18&ndash;21h "
        "with MAE &asymp; 0.45); XGBoost wins quiet hours and GRU wins peak hours. "
        "The monthly pattern is dominated by the <i>grandes vacances</i> (August, "
        "MAE 0.21&ndash;0.25, empty household) and the regulated <i>p&eacute;riode "
        "de chauffe</i> (October, MAE 0.42&ndash;0.44, heating ramp).",
        s["Body"]))
    story.append(_scaled(FIGURES / "segment_mae_by_dimension.png", 15 * cm, 9 * cm))
    story.append(_para("Figure 2. Per-segment MAE by model.", s["Caption"]))

    story.append(_para(
        "<i>(Additional: weather augmentation.)</i> XGBoost was retrained on the "
        "feature matrix augmented with four hourly Paris&ndash;Montsouris weather "
        "columns: temperature, relative humidity, wind speed, and a strictly past "
        "24-hour-lagged temperature. Per-fold result:",
        s["Body"]))
    wx_rows = [["Fold", "Baseline MAE", "+Weather MAE", "Δ MAE", "Δ %"]]
    for i in range(len(base)):
        b = base["MAE"].iloc[i]; w = wx["MAE"].iloc[i]
        d = w - b; p = d / b * 100
        wx_rows.append([f"{int(base['fold'].iloc[i])}", f"{b:.4f}", f"{w:.4f}",
                        f"{d:+.4f}", f"{p:+.2f}%"])
    wx_rows.append(["<b>Mean</b>", f"<b>{base['MAE'].mean():.4f}</b>",
                    f"<b>{wx['MAE'].mean():.4f}</b>",
                    f"<b>{wx['MAE'].mean() - base['MAE'].mean():+.4f}</b>",
                    f"<b>{(wx['MAE'].mean() - base['MAE'].mean()) / base['MAE'].mean() * 100:+.2f}%</b>"])
    t = Table(wx_rows, colWidths=[2 * cm, 3.5 * cm, 3.5 * cm, 3 * cm, 3 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(_para("Table 3. Weather-augmented XGBoost vs baseline XGBoost. The "
                       "&minus;0.77% mean is small but signed in the predicted direction; "
                       "the improvement concentrates on heating-season folds (2, 3, 4, 5, 6).",
                       s["Caption"]))

    story.append(_para("3.2 Project-oriented (business-value) implications", s["H2"]))
    story.append(_para(
        "For Enel Global ICT, three implications follow from the technical results:",
        s["Body"]))
    impls = [
        "<b>Architecture choice is a near-decision on this dataset.</b> XGBoost / "
        "GRU / LSTM are statistically tied. Choosing among them should be driven "
        "by non-MAE factors &mdash; inference latency, retraining cost, "
        "interpretability, and peak-hour accuracy (a real driver of imbalance-market "
        "penalties for grid operators).",
        "<b>Peak-hour accuracy and overall MAE rank models differently.</b> XGBoost "
        "minimises overall MAE but GRU minimises 18&ndash;21h MAE. A deployment "
        "weighted toward peak-hour cost should prefer GRU; one weighted toward "
        "overall accuracy can keep XGBoost. The aggregate leaderboard hides this "
        "trade-off &mdash; per-segment analysis surfaces it.",
        "<b>Free weather data yields a real but modest improvement.</b> Four "
        "Paris&ndash;Montsouris columns reduce XGBoost MAE by 0.77% with the "
        "improvement concentrated on heating-season folds. For a fleet-level "
        "residential forecast this is a non-trivial reduction in imbalance-market "
        "exposure during winter. The deeper implication is that <i>better input "
        "information</i> &mdash; richer weather features, occupancy proxies, "
        "calendar-aware regressors for SARIMA &mdash; is where additional gains live, "
        "not in trying new model architectures.",
    ]
    story.extend(_bullets(impls, s["Bullet"]))
    story.append(PageBreak())

    # ── SECTION 4: CONCLUSIONS ─────────────────────────────────────
    story.append(_para("4. Conclusions", s["H1"]))
    takeaways = [
        "<b>The pipeline is leakage-safe by construction.</b> A feature registry "
        "mechanically excludes same-timestamp components of the target; short-gap "
        "imputation is past-only; two regression tests guard both policies on every "
        "commit.",
        "<b>XGBoost wins the leaderboard at cross-fold MAE 0.336</b>, beating SARIMA "
        "significantly (Diebold&ndash;Mariano p = 0.022) but not beating LSTM "
        "(p = 0.33) or GRU (p = 0.42). The top three are statistically tied.",
        "<b>GRU is competitive with LSTM at lower complexity.</b> Across all six "
        "folds, GRU's MAE (0.339) is essentially tied with LSTM's (0.341). The "
        "simpler cell captures the relevant sequential structure as well.",
        "<b>Stacking confirms the bottleneck is input information, not model "
        "variety.</b> A ridge ensemble over the four base models does not break the "
        "3% cross-fold cluster &mdash; base-model errors are too correlated.",
        "<b>Weather augmentation validates the &quot;input information&quot; "
        "hypothesis empirically.</b> Adding four weather columns to XGBoost reduces "
        "cross-fold MAE by 0.77%, concentrated on the heating-season folds.",
        "<b>Operationally, the best model depends on the metric.</b> XGBoost wins "
        "overall MAE; GRU wins peak-hour MAE. A grid operator weighting peak-hour "
        "accuracy heavily should prefer GRU.",
    ]
    story.extend(_bullets(takeaways, s["Bullet"]))

    # ── APPENDIX A: CODE DESCRIPTION ───────────────────────────────
    story.append(PageBreak())
    story.append(_para("Appendix A &mdash; Code Description", s["H1"]))
    story.append(_para(
        "The source code in <b>src.zip</b> is organised as numbered scripts run in "
        "order, with a supporting Python package. Pseudocode of the end-to-end "
        "pipeline:",
        s["Body"]))
    pseudo = """
01_build_features.py
    raw = load_uci('data/raw/data.txt')
    short_gaps, long_gaps = classify_missing(raw)
    cleaned = forward_fill(raw, max_gap=3 min)        # past-only
    hourly  = resample(cleaned, '1h', mean)
    features = build(target=hourly.Global_active_power,
                     lags=[1,24,168],
                     rolling=[3,24,168],
                     calendar=['hour','dow','month','is_weekend','is_french_holiday'],
                     cyclical=['hour','dow','month'])  # sin/cos
    save(features, 'data/features.parquet')

02_fetch_weather.py
    weather = meteostat.Hourly('07156', start, end).fetch()  # Paris-Montsouris
    save(weather[['temp','rhum','wspd']] + lag24(temp),
         'data/paris_montsouris_weather.parquet')

04..08  run_<model>.py
    for fold in rolling_origin_splits(features.index, n=6, valid_days=30):
        Xt, yt = features.loc[fold.train], target.loc[fold.train]
        Xv, yv = features.loc[fold.valid], target.loc[fold.valid]
        model = <FAMILY>Forecaster(**hyperparams)
        model.fit(Xt, yt)
        preds = model.predict_one_step_ahead(Xv)
        save(preds, f'results/<model>_predictions_fold{fold.id}.csv')
        save(metrics(yv, preds), f'results/<model>_fold_results.csv')

09_run_stacking.py
    base = load_predictions(['xgboost','sarima','lstm','gru'], folds=1..5)
    meta = Ridge(alpha=1.0).fit(base.X, base.y)
    eval(meta, predictions(folds=6))

10_run_dm_test_and_segments.py
    p_values = pairwise_dm_test(predictions_folds_1_6)
    segment_mae = group_by(['hour','dow','month'], MAE)
    save(p_values, 'results/dm_test_pvalues.csv')

11_aggregate_leaderboard.py
    combine fold_results CSVs -> mean +- std per model -> results/leaderboard.csv

12_build_technical_report.py / 13_build_presentation.py
    load tracked artifacts, render PDF / PPTX
"""
    for line in pseudo.strip().split("\n"):
        story.append(_para(line.replace(" ", "&nbsp;"), s["Mono"]))

    # ── APPENDIX B: CONTRIBUTION + GENAI ───────────────────────────
    story.append(PageBreak())
    story.append(_para("Appendix B &mdash; Author Contribution and Generative AI Statement",
                        s["H1"]))

    story.append(_para("B.1 Author contribution (CRediT) &mdash; <i>DRAFT</i>", s["H2"]))
    story.append(_para(
        "<i>Draft.</i> Using the CRediT taxonomy, contributions across the four "
        "authors are as follows:",
        s["Draft"]))
    credit = [
        "<b>Linda Carla Zorzoli &amp; Mariavittoria Giurato</b> &mdash; "
        "<i>Investigation</i> (exploratory data analysis), <i>Resources</i> "
        "(research on documented French / European events explaining the dataset "
        "outages), <i>Validation</i> (review of the submission materials against "
        "the project rules).",
        "<b>Thierry Ishimwe &amp; Cesar Dushimimana</b> &mdash; "
        "<i>Methodology</i> and <i>Software</i> (feature engineering, modelling, "
        "evaluation pipeline, reporting tooling); <i>Writing &mdash; original draft</i> "
        "and <i>Writing &mdash; review &amp; editing</i>.",
        "<b>Conceptualization</b> and <b>Project administration</b> were shared "
        "across all four authors.",
    ]
    story.extend(_bullets(credit, s["Bullet"]))

    story.append(_para("B.2 Generative AI Statement &mdash; <i>DRAFT</i>", s["H2"]))
    story.append(_para(
        "<i>Draft.</i> We used Anthropic&apos;s Claude during development for: "
        "<b>(a) extensive leakage checking</b> &mdash; auditing the feature "
        "pipeline for same-timestamp target components, verifying that lag and "
        "rolling features are strictly past-only, and surfacing the bilateral-"
        "interpolation leak in the missing-value policy which we then fixed; "
        "<b>(b)</b> brainstorming the registry-based leakage-prevention design and "
        "the past-only-fill regression test; <b>(c)</b> reviewing wording across "
        "the notebook, this technical report, and the presentation. All "
        "AI-generated code was reviewed, executed, and validated against tests. "
        "All design decisions and final wording are owned and understood by the "
        "authors.",
        s["Draft"]))

    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4,
        leftMargin=1.3 * cm, rightMargin=1.3 * cm,
        topMargin=1.1 * cm, bottomMargin=1.1 * cm,
        title="Energy Forecasting Technical Report",
        author="Thierry Ishimwe, Linda Carla Zorzoli, Mariavittoria Giurato, Cesar Dushimimana",
    )
    doc.build(story)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    build()
