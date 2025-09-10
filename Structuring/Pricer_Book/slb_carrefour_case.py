from __future__ import annotations

from .core import (
    build_schedule_fixed,
    build_schedule_variable,
    present_value,
    price_from_yield,
    yield_from_price,
    macaulay_duration_convexity,
    truncate_with_redemption,
)

# DCM_Project/Pricer_Book/slb_carrefour_case.py

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt  # <-- ajouté pour le graphique SLB créatif
import math

# Core utilities shared in your project
from .core import (
    build_schedule_variable,   # (notional, rates_annual[], freq, structure)
    price_from_yield,
    yield_from_price,
    macaulay_duration_convexity,
)

# Shared charts
from .visuals import (
    price_yield_chart,
    cashflow_breakdown_chart,
    amortization_chart,
    rate_path_chart,
)

# =========================================================
# Case data (from the press release / term-sheet snippets)
# =========================================================
CASE_NAME   = "Carrefour — Sustainability-Linked Bond (2025)"
ISSUER      = "Carrefour SA"
CASE_SIZE   = 650_000_000.0          # €650m
CASE_COUPON = 3.75                    # % p.a. base coupon
CASE_TENOR  = 7.9                     # years to May-2033 from mid-2025 (approx.)
CASE_FREQ   = 2                       # semi-annual
CASE_FMT    = "Fixed, bullet"         # format text only

# =========================================================
# Schedule builder for SLB with 2 KPIs and step adjustments
# =========================================================
def build_slb_schedule(
    notional: float,
    base_coupon_pct: float,           # in %
    freq: int,
    years: float,
    structure: str,
    # KPI 1 — GHG (Scopes 1&2)
    kpi1_met: bool,
    kpi1_obs_year: float,
    kpi1_step_bps_if_missed: float,
    # KPI 2 — Suppliers engaged
    kpi2_met: bool,
    kpi2_obs_year: float,
    kpi2_step_bps_if_missed: float,
):
    """
    Build per-period annual 'profit' rates reflecting SLB step-ups if KPIs are missed.
    (Educational simplification — we only model step-ups; no step-downs or make-wholes.)
    Returns (schedule_df, times_years (np.ndarray), rate_path_pct (np.ndarray))
    """
    n = max(1, int(round(freq * years)))
    per = 1 / freq
    times = np.arange(1, n + 1) * per

    # Start with flat base coupon (annual %)
    path_pct = np.full(n, base_coupon_pct, dtype=float)

    # Apply step-up from observation date onward if KPI missed
    if not kpi1_met:
        path_pct[times >= kpi1_obs_year] += kpi1_step_bps_if_missed / 100.0
    if not kpi2_met:
        path_pct[times >= kpi2_obs_year] += kpi2_step_bps_if_missed / 100.0

    # Convert to decimals for the schedule builder
    rates_annual = (path_pct / 100.0).tolist()
    df = build_schedule_variable(notional, rates_annual, freq, structure)
    df["Total CF"] = df["Coupon/Profit"] + df["Principal"]
    return df, times, path_pct


# =========================================================
# UI
# =========================================================
def render():
    st.subheader("Structuring Desk — SLB Case Study")

    st.markdown(
        f"""
**Case:** *{CASE_NAME}*  
**Issuer:** {ISSUER} · **Issue Size:** €{CASE_SIZE:,.0f} · **Format:** {CASE_FMT}  
**Base coupon:** **{CASE_COUPON:.2f}%** p.a. · **Tenor:** ~{CASE_TENOR:g}y · **Freq.:** Semi-annual
        """
    )

    slb_carrefour_case_ui = render

    # ------------------ Parameters (sliders & toggles only) ------------------
    st.markdown("### Case parameters & KPI performance")

    # Left box: capital structure / solve; Right box: KPI logic
    lcol, rcol = st.columns([1.1, 1.3])

    with lcol:
        # Notional (slider, keep wide range but start at case size)
        notional = st.slider(
            "Notional (€)", min_value=100_000_000, max_value=1_500_000_000,
            value=int(CASE_SIZE), step=50_000_000, help="For scenario sizing / PV scaling."
        )
        years = st.slider(
            "Maturity (years)", min_value=1.0, max_value=15.0,
            value=float(CASE_TENOR), step=0.1
        )
        freq_label = st.selectbox("Coupon frequency", ["Annual", "Semi-annual", "Quarterly"], index=1)
        freq = {"Annual": 1, "Semi-annual": 2, "Quarterly": 4}[freq_label]
        structure = st.selectbox("Structure", ["bullet", "equal_principal"], index=0)

        # Solve mode
        solve_mode = st.radio("Solve for", ["Price (given Yield)", "Yield (given Clean Price)"], horizontal=True)
        if solve_mode == "Price (given Yield)":
            ytm_pct = st.slider("Yield to Maturity (%)", 0.00, 12.00, 5.00, 0.05)
        else:
            clean_target = st.slider("Target clean price (per 100)", 50.0, 130.0, 100.0, 0.5)

        grid_bps = st.slider("Price–Yield grid width (bps)", 50, 500, 200, 25)

    with rcol:
        st.markdown("#### KPI performance & step-up rules (interactive)")
        base_coupon_pct = st.slider("Base coupon (%)", 0.00, 10.00, float(CASE_COUPON), 0.05)

        # KPI 1 — GHG (Scopes 1&2)
        st.markdown("**KPI #1 — GHG (Scopes 1&2)**")
        kpi1_met = st.toggle("Target met?", value=False, help="If unticked = missed → step-up applies from the observation date.")
        kpi1_obs_year = st.slider("Observation year (KPI #1)", 0.5, years, min(3.0, years), 0.5)
        kpi1_step_bps = st.slider("Step-up if missed (bp) — KPI #1", 0.0, 50.0, 25.0, 5.0)

        # KPI 2 — Suppliers engaged in climate strategy
        st.markdown("**KPI #2 — Suppliers engaged**")
        kpi2_met = st.toggle("Target met?  ", value=True, help="If unticked = missed → step-up applies from the observation date.")
        kpi2_obs_year = st.slider("Observation year (KPI #2)", 0.5, years, min(5.0, years), 0.5)
        kpi2_step_bps = st.slider("Step-up if missed (bp) — KPI #2", 0.0, 50.0, 25.0, 5.0)

    # ------------------ Build schedule from KPI selections ------------------
    schedule, times, path_pct = build_slb_schedule(
        notional,
        base_coupon_pct,
        freq,
        years,
        structure,
        kpi1_met, kpi1_obs_year, kpi1_step_bps,
        kpi2_met, kpi2_obs_year, kpi2_step_bps,
    )

    # ------------------ Solve & core risk -----------------------------------
    if solve_mode == "Price (given Yield)":
        ytm = ytm_pct / 100.0
        clean, accrued, dirty = price_from_yield(schedule, ytm, freq, notional, accrued_frac=0.0)
        _, mac_dur, mod_dur, conv = macaulay_duration_convexity(schedule, ytm, freq)
    else:
        ytm = yield_from_price(schedule, clean_target, freq, notional)
        clean, accrued, dirty = price_from_yield(schedule, ytm, freq, notional, accrued_frac=0.0)
        _, mac_dur, mod_dur, conv = macaulay_duration_convexity(schedule, ytm, freq)

    # KPIs summary
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clean Price (per 100)", f"{clean:,.4f}")
    k2.metric("Dirty Price (per 100)", f"{dirty:,.4f}")
    k3.metric("YTM (%)", f"{ytm*100:,.4f}")
    k4.metric("Modified Duration (yrs)", f"{mod_dur:,.4f}")
    k5, k6, k7 = st.columns(3)
    k5.metric("Macaulay Duration (yrs)", f"{mac_dur:,.4f}")
    k6.metric("Convexity (yrs²)", f"{conv:,.4f}")
    # simple text badges
    st.caption(
        f"KPI #1 (GHG) — **{'Met' if kpi1_met else 'Missed'}** at {kpi1_obs_year:g}y · "
        f"KPI #2 (Suppliers) — **{'Met' if kpi2_met else 'Missed'}** at {kpi2_obs_year:g}y."
    )

    # ------------------ Charts (interactive) --------------------------------
    st.markdown("### Results & Charts")
    left, right = st.columns(2)

    # Price–Yield curve around current YTM (interactive)
    bps = grid_bps / 10_000.0
    y_grid = np.linspace(max(-0.99, ytm - bps), ytm + bps, 41)
    prices = np.array([price_from_yield(schedule, y, freq, notional, 0.0)[0] for y in y_grid])
    with left:
        st.altair_chart(
            price_yield_chart(y_grid * 100, prices, title="Price–Yield (with KPI-driven steps)").interactive(),
            use_container_width=True
        )

    with right:
        st.altair_chart(cashflow_breakdown_chart(schedule).interactive(), use_container_width=True)

    r1, r2 = st.columns(2)

    # Remplacement de l'amortization chart par un graphique SLB créatif
    with r1:
        # KPI Step-Up Timeline: ligne du coupon effectif, ligne base (pointillée),
        # et règles/points aux dates d'observation avec statut Met/Missed.
        df_coupon = pd.DataFrame({
            "Time (years)": times,
            "Coupon (%)": path_pct,
            "Base coupon (%)": np.full_like(path_pct, base_coupon_pct, dtype=float),
        })

        marks = pd.DataFrame({
            "Time (years)": [kpi1_obs_year, kpi2_obs_year],
            "KPI": ["GHG", "Suppliers"],
            "Status": ["Met" if kpi1_met else "Missed", "Met" if kpi2_met else "Missed"],
            "Step (bp)": [0.0 if kpi1_met else kpi1_step_bps, 0.0 if kpi2_met else kpi2_step_bps],
        })

        line_effective = (
            alt.Chart(df_coupon)
            .mark_line()
            .encode(
                x=alt.X("Time (years):Q", title="Time (years)"),
                y=alt.Y("Coupon (%):Q", title="Coupon (%)"),
                tooltip=["Time (years)", "Coupon (%)"],
            )
        )

        line_base = (
            alt.Chart(df_coupon)
            .mark_line(strokeDash=[6, 4])
            .encode(
                x="Time (years):Q",
                y="Base coupon (%):Q",
                tooltip=["Time (years)", "Base coupon (%)"],
            )
        )

        rules = (
            alt.Chart(marks)
            .mark_rule(size=1)
            .encode(
                x="Time (years):Q",
                color=alt.Color("Status:N", scale=alt.Scale(domain=["Missed","Met"], range=["#d62728", "#2ca02c"])),
                tooltip=["KPI","Status","Step (bp)","Time (years)"],
            )
        )

        points = (
            alt.Chart(marks)
            .mark_point(filled=True, size=80)
            .encode(
                x="Time (years):Q",
                y=alt.Y("y:Q", title=None),
                color=alt.Color("Status:N", scale=alt.Scale(domain=["Missed","Met"], range=["#d62728", "#2ca02c"])),
                shape="KPI:N",
                tooltip=["KPI","Status","Step (bp)","Time (years)"],
            )
            .transform_calculate(
                # place points at corresponding coupon level after observation (approx = current coupon at that time)
                y=str(base_coupon_pct)
            )
        )

        chart_slb = alt.layer(line_effective, line_base, rules, points).properties(
            title="KPI Step-Up Timeline (Coupon path vs. Base)"
        )

        st.altair_chart(chart_slb.interactive(), use_container_width=True)

    with r2:
        # Suppression du graphique "Coupon/Profit rate path (%)" (demandé)
        # (la colonne est laissée vide pour conserver la structure visuelle)
        pass

    # ------------------ Cash-flow table & export -----------------------------
    st.markdown("### Cash flow table")
    table = schedule.copy()
    for c in ["Outstanding (begin)", "Coupon/Profit", "Principal", "Total CF", "Outstanding (end)"]:
        table[c] = table[c].map(lambda x: round(float(x), 6))
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "Download cash flow table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="slb_carrefour_cash_flows.csv",
        mime="text/csv",
    )

    # ------------------ Case Study: definitions & learn more -----------------
    st.markdown("---")
    st.markdown("## What is a Sustainability-Linked Bond (SLB)?")
    st.markdown(
        """
- **Use of proceeds is general** (not ring-fenced) but the bond is **contractually linked to KPIs/targets**.  
- If KPIs are **missed** at observation dates, a **coupon step-up** (e.g., +25 bp) typically applies **from then to maturity**.  
- If KPIs are **met/over-achieved**, many frameworks keep the base coupon (some structures allow step-downs).  
- The mechanism is documented via **Sustainability-Linked Financing Framework** and external **SPO**.
        """
    )

    # Learn more (modal/expander)
    if hasattr(st, "dialog"):
        @st.dialog("Learn more — SLB vs. Green Bond")
        def _more():
            _render_learn_more()
        if st.button("Learn More"):
            _more()
    else:
        with st.expander("Learn more — SLB vs. Green Bond"):
            _render_learn_more()

    # ------------------ Example PDF (Library) --------------------------------
    st.markdown("### Example — Download")
    pdf_path = Path(__file__).resolve().parent.parent.parent / "Library" / "SLB Example - Carrefour (2025).pdf"

    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download: SLB Example — Carrefour (2025) (PDF)",
                data=f.read(),
                file_name=pdf_path.name,
                mime="application/pdf",
            )
    else:
        st.info(f"Place the case PDF at **{pdf_path}** (filename must match exactly).")

def _render_learn_more():
    st.markdown(
        """
### SLB vs. Green Bond (quick primer)
- **Green Bond (GB):** use-of-proceeds is **earmarked** for eligible green projects; financial terms are **not** KPI-linked.  
- **Sustainability-Linked Bond (SLB):** proceeds are **general corporate**; **financial characteristics** (e.g., coupon) vary with
  **KPI performance** measured against **SBTi-aligned** or similar targets, with **observation dates** and **step-up amounts**.

### How we calculate in this case study
1) Build a per-period rate path: start from **base coupon** and add **step-ups (bp)** from each observation date **if the KPI is missed**.  
2) Convert the per-period annual rates to cash flows (profit + principal) using the selected **frequency** and **structure**.  
3) **Price (given Yield):** discount all cash flows at the input **YTM** (same engine as plain bonds).  
4) **Yield (given Clean Price):** numerically solve for the YTM matching the target **clean price**.  
5) Charts update as you toggle KPI outcomes and move sliders (observation years, step-up sizes, base coupon, etc.).

> Notes: Real documentation can include **multiple check-ins**, step-up caps, **step-downs**, make-wholes, and other nuances.
  This module keeps the mechanics **transparent for education** while matching the spirit of the Carrefour SLB.
        """
    )


# =========================
# Sources (kept as comments)
# =========================
# Case press release snippet: “Success of a €650m 7.9-year Sustainability-Linked Bond … coupon 3.75%,
# indexed to two objectives: (1) GHG Scope 1&2 reduction; (2) suppliers engaged in climate strategy.”
# (PDF placed in Library as "SLB Example - Carrefour (2025).pdf")
