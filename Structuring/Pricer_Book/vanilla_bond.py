# vanilla_bond.py — Streamlit sub-page (Vanilla Bonds)
# Everything wrapped in render(), no page_config here.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


# ---------- Model & utilities ----------
@dataclass
class BondSpec:
    face: float = 100.0
    coupon_rate: float = 0.05     # annual coupon rate (e.g., 0.05 = 5%)
    freq: int = 2                 # payments per year (1=annual, 2=semi, 4=quarterly, 12=monthly)
    years_to_maturity: float = 5.0
    clean_price: float | None = None


def _cashflow_schedule(spec: BondSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Return times (in years) and cash flows per period."""
    n = int(round(spec.years_to_maturity * spec.freq))
    times = np.arange(1, n + 1) / spec.freq
    cpn = spec.face * spec.coupon_rate / spec.freq
    cfs = np.full(n, cpn, dtype=float)
    cfs[-1] += spec.face
    return times, cfs


def price_from_yield(spec: BondSpec, ytm: float) -> float:
    """Clean price given an annual yield (compounded at 'freq')."""
    times, cfs = _cashflow_schedule(spec)
    per = ytm / spec.freq
    disc = (1.0 + per) ** (spec.freq * times)
    return float(np.sum(cfs / disc))


def yield_from_price(spec: BondSpec, target_price: float, guess: float = 0.05) -> float:
    """Solve annualized YTM that matches a clean price (Newton with finite-diff slope)."""
    r = max(1e-8, guess)
    for _ in range(60):
        f0 = price_from_yield(spec, r) - target_price
        if abs(f0) < 1e-10:
            break
        h = max(1e-6, 1e-3 * max(1.0, r))
        f1 = price_from_yield(spec, r + h) - target_price
        deriv = (f1 - f0) / h
        if deriv == 0:
            break
        r = r - f0 / deriv
        # sensible bounds
        r = min(max(r, -0.99), 5.0)
    return float(max(-0.99, r))


def macaulay_duration(spec: BondSpec, ytm: float) -> float:
    """Macaulay duration (years)."""
    times, cfs = _cashflow_schedule(spec)
    per = ytm / spec.freq
    disc = (1.0 + per) ** (spec.freq * times)
    pv = cfs / disc
    price = float(np.sum(pv))
    weights = pv / price
    return float(np.sum(times * weights))


def modified_duration(spec: BondSpec, ytm: float) -> float:
    mac = macaulay_duration(spec, ytm)
    return float(mac / (1.0 + ytm / spec.freq))


def convexity(spec: BondSpec, ytm: float) -> float:
    """Convexity (years^2, discrete compounding approximation)."""
    times, cfs = _cashflow_schedule(spec)
    per = ytm / spec.freq
    disc = (1.0 + per) ** (spec.freq * times)
    pv = cfs / disc
    price = float(np.sum(pv))
    t_per = (spec.freq * times)  # in coupon periods
    num = np.sum((t_per * (t_per + 1.0)) * cfs / (1.0 + per) ** (t_per + 2.0))
    conv_per = num / (price * (spec.freq ** 2))
    return float(conv_per)


def _relative_value_demo(spec: BondSpec, df: pd.DataFrame):
    """
    Simple relative-value demo:
      • compute peer YTMs from Price (if Yield missing)
      • take median peer yield as 'fair'
      • derive fair price for the subject bond and a spread vs peers
    """
    peers = df.copy()

    if 'Yield' not in peers.columns and 'Price' in peers.columns:
        yields = []
        for _, r in peers.iterrows():
            ps = BondSpec(
                face=r.get('Face', spec.face),
                coupon_rate=float(r['Coupon']),
                freq=int(r.get('Freq', spec.freq)),
                years_to_maturity=float(r['MaturityYears']),
            )
            y = yield_from_price(ps, float(r['Price']), guess=spec.coupon_rate)
            yields.append(y)
        peers['Yield'] = yields

    fair_y = float(peers['Yield'].median())
    fair_p = price_from_yield(spec, fair_y)
    our_y = yield_from_price(spec, fair_p, guess=fair_y)
    spread_bps = (our_y - fair_y) * 1e4
    return fair_y, fair_p, spread_bps, peers


# ---------- UI entry point ----------
def render():
    st.markdown("## Vanilla Bond")

    # Global CSS: paragraphs justified + subtle grey utility
    st.markdown(
        """
        <style>
          .subtle-grey { color:#9aa0a6; }
          /* justify most narrative text */
          .stMarkdown p, .stMarkdown li, .stMarkdown div { text-align: justify; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Small grey subtitle
    st.markdown(
        '<div class="subtle-grey" style="margin-top:-6px;margin-bottom:10px;">'
        'Plain-vanilla fixed-rate bonds: no embedded options, no special features — just coupons and final redemption.'
        '</div>',
        unsafe_allow_html=True
    )

    # 2×2 matrix
    colTL, colTR = st.columns([1.05, 1.0])
    colBL, colBR = st.columns([1.2, 0.8])

    # --- Top-Left: Parameters ---
    with colTL:
        st.subheader("Parameters")

        method = st.radio(
            "Pricing method",
            ["Yield-to-Maturity", "Discounted Cash Flow", "Relative Value (demo)"],
        )

        face = st.number_input("Face value", value=100.0, step=1.0, format="%.2f")
        coupon_rate = st.number_input("Annual coupon (%)", value=5.0, step=0.25, format="%.2f") / 100.0

        # Frequency in words
        freq_label_to_val = {
            "Annual (1x)": 1,
            "Semiannual (2x)": 2,
            "Quarterly (4x)": 4,
            "Monthly (12x)": 12,
        }
        freq_label = st.selectbox("Coupon frequency", list(freq_label_to_val.keys()), index=1)
        freq = freq_label_to_val[freq_label]

        # Maturity slider 1–30 years
        years = st.slider("Maturity (years)", min_value=1, max_value=30, value=5, step=1)
        years = float(years)

        if method == "Yield-to-Maturity":
            mode = st.selectbox("Mode", ["Given YTM → Price", "Given Price → Solve YTM"], index=0)
            if mode == "Given YTM → Price":
                ytm_in = st.number_input("Yield-to-Maturity (annual, %)", value=5.5, step=0.25, format="%.2f") / 100.0
                px_in = None
            else:
                px_in = st.number_input("Observed clean price", value=98.50, step=0.25, format="%.2f")
                ytm_in = None

        elif method == "Discounted Cash Flow":
            disc_rate = st.number_input("Discount rate (annual, %)", value=6.0, step=0.25, format="%.2f") / 100.0

        else:  # Relative Value demo
            st.caption("Upload or edit a small peer table (demo).")
            default_peers = pd.DataFrame({
                "Coupon": [0.04, 0.05, 0.055, 0.06],
                "MaturityYears": [years, years, years, years],
                "Price": [99.2, 100.5, 98.0, 97.3],
                "Face": [face] * 4,
                "Freq": [freq] * 4,
            })
            peer_df = st.data_editor(default_peers, use_container_width=True, num_rows="dynamic")

    # --- Top-Right: Method overview (fixed text) ---
    with colTR:
        st.subheader("Method overview")
        st.markdown(
            "- **Yield-to-Maturity (YTM) Method:** This method involves calculating the present value of all future cash flows from the bond, assuming the bond is held until maturity, and discounting them using the bond's yield-to-maturity. The YTM is the interest rate that makes the present value of the bond's cash flows equal to its current market price."
        )
        st.markdown(
            "- **Discounted Cash Flow (DCF) Method:** This method involves projecting the bond's cash flows over its remaining life, discounting them back to their present value using an appropriate discount rate, and summing them to arrive at the bond's price."
        )
        st.markdown(
            "- **Relative Value Method:** This method involves comparing the bond's yield to the yields of other bonds with similar characteristics, such as credit rating, maturity, and coupon rate. This can help determine whether the bond is undervalued or overvalued relative to its peers."
        )

        # ---- Example PDF download (below the overview text) ----
        st.markdown("### Example — Download")
        pdf_path = Path(__file__).resolve().parent.parent.parent / "Library" / "Vanilla Bond Example - Legrand (2025).pdf"
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download: Vanilla Bond Example — Legrand (2025) (PDF)",
                    data=f.read(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                )
        else:
            st.info(f"Place the example PDF at **{pdf_path}** (filename must match exactly).")

    # --- Compute results based on inputs ---
    spec = BondSpec(face=face, coupon_rate=coupon_rate, freq=int(freq), years_to_maturity=years)

    if method == "Yield-to-Maturity":
        if mode == "Given YTM → Price":
            price = price_from_yield(spec, ytm_in)
            ytm_used = float(ytm_in)
        else:
            ytm_used = yield_from_price(spec, px_in, guess=max(1e-4, coupon_rate))
            price = float(px_in)

    elif method == "Discounted Cash Flow":
        # Pricing with a chosen discount rate is equivalent to YTM pricing with that rate
        ytm_used = float(disc_rate)
        price = price_from_yield(spec, disc_rate)

    else:  # Relative Value (demo)
        if 'peer_df' not in locals() or peer_df is None or peer_df.empty:
            peer_df = default_peers
        y_fair, p_fair, spr_bps, peers = _relative_value_demo(spec, peer_df)
        ytm_used, price = y_fair, p_fair

    macD = macaulay_duration(spec, ytm_used)
    modD = modified_duration(spec, ytm_used)
    convx = convexity(spec, ytm_used)

    # --- Bottom-Left: SINGLE dual-axis chart (cash-flow bars + effective yield line), interactive ---
    with colBL:
        st.subheader("Charts: Cash Flows Schedule & Effective Yield Curve")

        t, cf = _cashflow_schedule(spec)
        df_cf = pd.DataFrame({"Year": t, "CashFlow": cf})
        eff_yield = (1.0 + (ytm_used / spec.freq)) ** (spec.freq * t) - 1.0
        df_y = pd.DataFrame({"Year": t, "EffectiveYieldPct": eff_yield * 100.0})

        bars = (
            alt.Chart(df_cf)
              .mark_bar()
              .encode(
                  x=alt.X("Year:Q", title="Year"),
                  y=alt.Y("CashFlow:Q", title="Cash flow"),
                  tooltip=[alt.Tooltip("Year:Q"), alt.Tooltip("CashFlow:Q", format=",.2f")]
              )
        )

        line = (
            alt.Chart(df_y)
              .mark_line(point=True)
              .encode(
                  x="Year:Q",
                  y=alt.Y("EffectiveYieldPct:Q", title="Yield (%)"),
                  tooltip=[alt.Tooltip("Year:Q"), alt.Tooltip("EffectiveYieldPct:Q", format=",.2f")]
              )
        )

        dual_axis = alt.layer(bars, line).resolve_scale(y='independent').properties(height=420).interactive(bind_y=False)
        st.altair_chart(dual_axis, use_container_width=True)

        # --- Grey explanation block below the chart (clarified "Line") ---
        st.markdown(
            """
            <div class="subtle-grey" style="font-size:0.95rem; margin-top:.35rem;">
              <div><strong>What you’re seeing:</strong></div>
              <ul style="margin-top:.25rem; margin-bottom:.5rem;">
                <li><em>Bars</em>: the bond’s cash flows over time (coupons and final principal).</li>
                <li><em>Line</em>: for each future date <em>t</em>, the <u>effective annual rate</u> obtained by compounding the chosen YTM up to that horizon:
                    <code>(1 + YTM/freq)^(freq·t) − 1</code>. It shows how a single, constant YTM translates into higher annualized growth over longer horizons due to compounding. It is <u>not</u> a forecast of future market rates.</li>
              </ul>
              <div>
                <strong>How to read it:</strong> the YTM is the discount rate that equates today’s price with the present value of future cash flows.
                Higher YTM ➜ steeper line (money grows faster per year when held longer) and heavier discounting (distant cash flows contribute less to price).
                Lower YTM ➜ flatter line and milder discounting (distant cash flows retain more present value).
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

       # --- Bottom-Right: KPIs — two columns × three rows, big numbers, rounded to 2 decimals ---
    with colBR:
        st.subheader("Key metrics")

        # Custom CSS pour centrer les metrics
        st.markdown(
            """
            <style>
            [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
                text-align: center !important;
                justify-content: center !important;
                align-items: center !important;
                display: flex !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        price_to_par = 100.0 * (price / spec.face)

        left, right = st.columns(2)
        with left:
            st.metric("Price", f"{price:,.2f}")
            st.metric("Macaulay duration (yrs)", f"{macD:,.2f}")
            st.metric("Convexity (yrs²)", f"{convx:,.2f}")
        with right:
            st.metric("Yield (annual, %)", f"{ytm_used * 100:,.2f}%")
            st.metric("Modified duration (yrs)", f"{modD:,.2f}")
            st.metric("Price-to-Par (%)", f"{price_to_par:,.2f}%")

        # ---- Collapsible "Learn More" just under the KPIs (right column area) ----
        with st.expander("Learn More"):
            st.markdown(
                """
                <div class="subtle-grey" style="font-size:0.95rem;">
                <div style="font-weight:600; margin-bottom:.15rem;">Price</div>
                <div>The clean price is the present value of coupons and principal discounted at the chosen rate (YTM or discount rate). It excludes accrued interest.</div>

                <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Yield (annual, %)</div>
                <div>The single annualized rate that makes the present value of future cash flows equal to today’s clean price. Think of it as the bond’s IRR if held to maturity and coupons are reinvested at the same rate.</div>

                <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Macaulay duration (yrs)</div>
                <div>Time-weighted average maturity of the bond’s cash flows (in years). Interpreted as the investment’s effective “center of mass” in time.</div>

                <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Modified duration (yrs)</div>
                <div>Price sensitivity to small, parallel yield changes: approximately <em>%ΔPrice ≈ −ModDur × ΔYield</em> (yield in decimal). Useful for first-order risk.</div>

                <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Convexity (yrs²)</div>
                <div>Second-order curvature of the price–yield relation. Higher convexity means the duration estimate errs less for larger yield moves (price falls less when yields rise and rises more when yields fall).</div>

                <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Price-to-Par (%)</div>
                <div>Price expressed as a percentage of the face value. &gt;100% = premium bond (coupon &gt; yield), &lt;100% = discount bond (coupon &lt; yield).</div>
                </div>
                """,
                unsafe_allow_html=True
            )
