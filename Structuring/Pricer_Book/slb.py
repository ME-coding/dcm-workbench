# slb.py — Sustainability-Linked Financing (Saint-Gobain) case study
from __future__ import annotations

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# Case constants (Saint-Gobain Aug-2022 issuance & framework)
# Two KPIs assessed against 2030 targets (2017 baseline):
#   • KPI #1: Reduce absolute Scope 1 & 2 CO₂ emissions by 33% by 2030
#   • KPI #2: Achieve an 80% reduction of non-recovered production waste by 2030
# If a KPI is NOT met, the 2032 coupon steps up by +37.5 bps per missed KPI (max +75 bps).
# Only the 2032 coupon is affected by the step-up; earlier coupons remain at 2.625%.
# Notional (deal size) is FIXED for this case study: €500,000,000.
# ============================================================

BASE_COUPON = 0.02625        # 2.625%
STEP_UP_PER_MISS = 0.00375   # 37.5 bps per KPI missed
PAY_YEAR = 2032              # Step-up applies to coupon paid in 2032
ISSUE_YR = 2023              # start of coupon schedule (adjust if you prefer)
TENOR_Y = 10
YEARS = list(range(ISSUE_YR, ISSUE_YR + TENOR_Y))  # e.g., 2023..2032
NOTIONAL = 500_000_000       # Fixed deal size

# Traffic-light colors
COLOR_GOOD = "#2E7D32"    # green
COLOR_NEUTRAL = "#FB8C00" # orange
COLOR_BAD = "#C62828"     # red

@dataclass
class Scenario:
    name: str
    step_up_bps: float
    color: str

SCENARIOS = [
    Scenario("Good (0 bps)", 0.0, COLOR_GOOD),
    Scenario("Neutral (+37.5 bps)", 37.5, COLOR_NEUTRAL),
    Scenario("Bad (+75 bps)", 75.0, COLOR_BAD),
]

# ---------
# KPI helper (no boxes, centered, large text, robust colors)
# ---------
KPI_CSS = """
<style>
.kpi {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  margin: 2px 0 6px 0;
  width: 100%;
}
.kpi * { margin-left: auto; margin-right: auto; }
.kpi-label { font-size: .95rem; opacity: .85; margin-bottom: 2px; }
.kpi-value { font-size: 2.1rem; font-weight: 700; line-height: 1.1; }
</style>
"""
def kpi(label: str, value_html: str, color: str | None = None):
    style = f"style='color:{color}'" if color else ""
    st.markdown(
        KPI_CSS + f"""
<div class='kpi'>
  <div class='kpi-label'>{label}</div>
  <div class='kpi-value' {style}>{value_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================
# Streamlit page rendering
# =========================
def render():
    st.header("Sustainability-Linked Bond")

    # ----------------------------------------------------------
    # Deal at a glance — context
    # ----------------------------------------------------------
    with st.expander("Deal at a glance (context) — read this first", expanded=True):
        st.markdown(
            """
**What was issued?**  
A **€500 million 10-year fixed-rate bond** (base coupon **2.625%**) whose **2032 coupon** can step up based on two sustainability targets.

**Targets tied to the bond (measured vs 2017 baseline):**  
- **KPI #1 — Scope 1 & 2 CO₂:** *Reduce absolute emissions by **33% by 2030***.  
- **KPI #2 — Non-recovered production waste:** *Achieve an **80% reduction by 2030***.

**Mechanics of the step-up (only 2032):**  
- Miss **one** KPI → **+37.5 bps** on the 2032 coupon.  
- Miss **both** KPIs → **+75 bps** on the 2032 coupon.  
- Hit both → No change; all coupons stay at **2.625%**.
            """
        )

        lib_candidates = [
            Path("Library") / "SLB Example - Saint Gobain (2022).pdf",
            Path("Library") / "Saint-Gobain Sustainability-Linked Financing Framework.pdf",
        ]
        existings = [p for p in lib_candidates if p.exists()]
        if existings:
            # 1/ libellé avec deux-points
            st.markdown("**Source of the Case Study:**")
            for p in existings:
                with open(p, "rb") as f:
                    st.download_button(f"{p.name}", data=f.read(), file_name=p.name, mime="application/pdf")

    # ===================================
    # Two-panel header: parameters (left) vs metrics (right)
    # ===================================
    left, right = st.columns([1.1, 1.1])

    # ---- PARAMETERS (left)
    with left:
        st.markdown("### Simulation controls")

        # Texte + slider pour Scope 1 & 2
        st.markdown(
            "Reduction of the absolute Scope 1 and 2 CO₂ emissions "
            "(<u><b>objective: >33%</b></u>)",
            unsafe_allow_html=True
        )
        co2_red = st.slider("", 0.0, 100.0, 30.0, 0.5)

        # Texte + slider pour waste
        st.markdown(
            "Reduction of non-recovered production waste "
            "(<u><b>objective: >80%</b></u>)",
            unsafe_allow_html=True
        )
        waste_red = st.slider("", 0.0, 100.0, 75.0, 0.5)

    # Determine KPI misses
    miss_co2   = 0 if co2_red > 33.0 else 1
    miss_waste = 0 if waste_red > 80.0 else 1
    misses = miss_co2 + miss_waste
    step_up_bps_current = misses * 37.5
    step_up_rate_current = step_up_bps_current / 10_000.0

    # Discrete outcomes
    coupon_2032 = (BASE_COUPON + step_up_rate_current) * 100.0     # 2.625 / 3.000 / 3.375
    extra_2032_eur_current = step_up_rate_current * NOTIONAL        # 0 / 1,875,000 / 3,750,000

    # Colors driven by the step-up (robust)
    if step_up_bps_current == 0.0:
        coupon_color = COLOR_GOOD
        cost_color = COLOR_GOOD
    elif step_up_bps_current == 37.5:
        coupon_color = COLOR_NEUTRAL
        cost_color = COLOR_NEUTRAL
    else:
        coupon_color = COLOR_BAD
        cost_color = COLOR_BAD

    # ---- METRICS (right) — centrés
    with right:
        st.subheader("Key metrics")
        col_a, col_b = st.columns(2)
        with col_a:
            kpi("2032 coupon", f"{coupon_2032:.3f}%", coupon_color)
        with col_b:
            kpi("Expected 2032 incremental cost to the issuer", f"€{extra_2032_eur_current:,.0f}", cost_color)

    # ----------------------------------------------------------
    # Chart — grouped bars ordered Good / Neutral / Bad (barres fines)
    # ----------------------------------------------------------
    st.markdown("### Coupon evolution under sustainability scenarios")

    # Build schedules (base coupons each year; 2032 adjusted)
    rows = []
    for sc in SCENARIOS:
        for y in YEARS:
            coupon_rate = BASE_COUPON
            if y == PAY_YEAR:
                coupon_rate = BASE_COUPON + (sc.step_up_bps / 10_000.0)
            rows.append({
                "Year": y,
                "Annual coupon (%)": coupon_rate * 100.0,
                "Scenario": sc.name
            })
    df_sched = pd.DataFrame(rows)

    # Order: Good -> Neutral -> Bad
    domain_order = [SCENARIOS[0].name, SCENARIOS[1].name, SCENARIOS[2].name]
    color_range = [SCENARIOS[0].color, SCENARIOS[1].color, SCENARIOS[2].color]

    chart = (
        alt.Chart(df_sched)
        .mark_bar(size=10)  # barres fines
        .encode(
            # 2/ étiquettes des années horizontales
            x=alt.X("Year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Scenario:N", scale=alt.Scale(domain=domain_order)),
            y=alt.Y("Annual coupon (%):Q", title="Annual coupon (%)"),
            color=alt.Color(
                "Scenario:N",
                scale=alt.Scale(domain=domain_order, range=color_range),
                legend=alt.Legend(title="Scenario"),
            ),
            tooltip=[
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Year:O"),
                alt.Tooltip("Annual coupon (%):Q", format=".3f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)

# For multi-page apps that import this module:
def run():
    render()

if __name__ == "__main__":
    render()
