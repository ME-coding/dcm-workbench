from __future__ import annotations  # ✅ Doit être tout en haut

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import math

# Reuse shared valuation utilities
from .core import (
    build_schedule_fixed,
    build_schedule_variable,
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

# =======================================================
# Sukuk schedule builders (simplified, pricer-friendly)
# =======================================================

def build_sukuk_fixed(
    notional: float,
    profit_rate_annual: float,
    freq: int,
    years: float,
    structure: str = "bullet",
) -> pd.DataFrame:
    df = build_schedule_fixed(notional, profit_rate_annual, freq, years, structure)
    df["Total CF"] = df["Coupon/Profit"] + df["Principal"]
    return df


def build_sukuk_step(
    notional: float,
    base_profit_pct: float,
    freq: int,
    years: float,
    steps_bps: List[Tuple[float, float]],
    structure: str = "bullet",
) -> pd.DataFrame:
    n_periods = max(1, int(round(freq * years)))
    per = 1 / freq
    times = np.arange(1, n_periods + 1) * per
    adj_bps = np.zeros(n_periods)
    for start_year, step_bp in steps_bps:
        idx = np.where(times >= float(start_year))[0]
        if len(idx) > 0:
            adj_bps[idx] += float(step_bp)
    rates_annual = [(base_profit_pct + adj_bps[i] / 100.0) / 100.0 for i in range(n_periods)]
    df = build_schedule_variable(notional, rates_annual, freq, structure)
    df["Total CF"] = df["Coupon/Profit"] + df["Principal"]
    return df, times, np.array([(base_profit_pct + adj_bps[i] / 100.0) for i in range(n_periods)])


# =======================================================
# UI renderer
# =======================================================

def render():
    st.subheader("Structuring Desk — Sukuk Pricer")

    default_notional, default_years, default_profit, default_yield = 1_000_000.0, 5.0, 3.00, 5.00
    f_map = {"Annual": 1, "Semi-annual": 2, "Quarterly": 4}

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        notional = st.number_input("Notional", min_value=1_000.0, value=default_notional, step=1_000.0, format="%.2f")
    with r1c2:
        years = st.number_input("Maturity (years)", min_value=0.25, value=default_years, step=0.25, format="%.2f")
    with r1c3:
        freq_label = st.selectbox("Profit distribution frequency", sorted(f_map.keys()))
        freq = f_map[freq_label]
    with r1c4:
        accrued_frac = st.slider("Elapsed in current period (%)", 0, 100, 0,
                                 help="Accrued profit approximation as % of the current period.") / 100.0

    struct_options = sorted(["annuity", "bullet", "equal_principal"])
    structure = st.selectbox("Structure", struct_options, index=1)

    st.markdown("#### Profit schedule")
    mode = st.radio("Profit rate input", ["Flat profit rate", "Step table"], horizontal=True)

    profit_path_times = None
    profit_path_pct = None

    if mode == "Flat profit rate":
        profit_rate = st.number_input("Profit rate (%)", min_value=0.0, value=default_profit, step=0.10, format="%.4f") / 100.0
        schedule = build_sukuk_fixed(notional, profit_rate, freq, years, structure)
    else:
        if "sukuk_steps" not in st.session_state:
            st.session_state.sukuk_steps = pd.DataFrame({"From year (>=)": [3.0, 4.0], "Step (bp)": [25.0, 25.0]})
        steps_df = st.data_editor(
            st.session_state.sukuk_steps,
            num_rows="dynamic",
            use_container_width=True,
            key="sukuk_steps_editor",
            help="Each row applies from the specified year onward."
        )
        st.session_state.sukuk_steps = steps_df.copy()
        base_profit_pct = st.number_input("Base profit rate (%)", min_value=0.0, value=default_profit, step=0.10, format="%.4f")
        steps_bps = [(float(r["From year (>=)"]), float(r["Step (bp)"])) for _, r in steps_df.iterrows()]
        schedule, profit_path_times, profit_path_pct = build_sukuk_step(notional, base_profit_pct, freq, years, steps_bps, structure)

    st.markdown("#### Solve")
    s1, s2 = st.columns(2)
    solve_mode = st.radio("Solve for", ["Price (given Yield)", "Yield (given Clean Price)"], horizontal=True)
    if solve_mode == "Price (given Yield)":
        with s1:
            ytm = st.number_input("Yield to maturity (%)", min_value=-10.0, value=default_yield, step=0.25, format="%.4f") / 100.0
    else:
        with s1:
            clean_target = st.number_input("Target clean price (per 100)", min_value=0.0, value=100.0, step=0.5, format="%.4f")
    with s2:
        grid_bps = st.slider("Price–Yield grid width (bps)", 50, 500, 200, step=25)

    if solve_mode == "Price (given Yield)":
        clean, accrued, dirty = price_from_yield(schedule, ytm, freq, notional, accrued_frac=accrued_frac)
        _, mac_dur, mod_dur, conv = macaulay_duration_convexity(schedule, ytm, freq)
    else:
        ytm = yield_from_price(schedule, clean_target, freq, notional)
        clean, accrued, dirty = price_from_yield(schedule, ytm, freq, notional, accrued_frac=accrued_frac)
        _, mac_dur, mod_dur, conv = macaulay_duration_convexity(schedule, ytm, freq)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clean Price (per 100)", f"{clean:,.4f}")
    k2.metric("Accrued (per 100)", f"{accrued:,.4f}")
    k3.metric("Dirty Price (per 100)", f"{dirty:,.4f}")
    k4.metric("Yield to Maturity (%)", f"{ytm*100:,.4f}")

    k5, k6, k7 = st.columns(3)
    k5.metric("Macaulay Duration (yrs)", f"{mac_dur:,.4f}")
    k6.metric("Modified Duration (yrs)", f"{mod_dur:,.4f}")
    k7.metric("Convexity (yrs²)", f"{conv:,.4f}")

    st.markdown("### Results & Charts")
    left, right = st.columns(2)

    y0 = ytm
    bps = grid_bps / 10_000.0
    y_grid = np.linspace(max(-0.99, y0 - bps), y0 + bps, 41)
    prices = np.array([price_from_yield(schedule, y, freq, notional, 0.0)[0] for y in y_grid])
    with left:
        st.altair_chart(price_yield_chart(y_grid * 100, prices), use_container_width=True)

    with right:
        st.altair_chart(cashflow_breakdown_chart(schedule), use_container_width=True)

    e1, e2 = st.columns(2)
    with e1:
        st.altair_chart(amortization_chart(schedule), use_container_width=True)
    with e2:
        if profit_path_times is not None and profit_path_pct is not None:
            st.altair_chart(rate_path_chart(profit_path_times, profit_path_pct, title="Profit Rate Path (Step Mode)"), use_container_width=True)
        else:
            st.caption("Switch to **Step table** to visualize a profit-rate path.")

    st.markdown("### Cash flow table")
    table = schedule.copy()
    for c in ["Outstanding (begin)", "Coupon/Profit", "Principal", "Total CF", "Outstanding (end)"]:
        table[c] = table[c].map(lambda x: round(float(x), 6))
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "Download cash flow table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="sukuk_cash_flows.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("## Sukuk — Theory (Quick Primer)")

    st.markdown(
        """
**What is a Sukuk?**  
- *Sharia-compliant fixed-income–like instrument*: returns are structured as **profit** or **rent**, not interest.  
- *Ownership linkage*: investors hold certificates representing **undivided ownership** in underlying assets via an SPV.  
- *Common structures*: **Ijara** (lease/rental), **Murabaha** (cost-plus sale), **Wakala** (agency), **Mudaraba** (profit-sharing).  
- *Market behavior*: in practice, pricing of rated Sukuk is **highly correlated** with comparable conventional bonds in many markets.
        """
    )

    if hasattr(st, "dialog"):
        @st.dialog("Learn more about Sukuk")
        def _sukuk_more():
            _render_learn_more()
        if st.button("Learn More about Sukuk Bond"):
            _sukuk_more()
    else:
        with st.expander("Learn More about Sukuk Bond"):
            _render_learn_more()

    pdf_path = Path(__file__).resolve().parent.parent.parent / "Library" / "Sukuk Bond Example - Indonesia (2023).pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download: Sukuk Bond Example — Indonesia (2023) (PDF)",
                data=f.read(),
                file_name=pdf_path.name,
                mime="application/pdf",
            )
    else:
        st.info(
            f"Place the example PDF at **{pdf_path}** to enable download. "
            "Filename must match exactly."
        )


def _render_learn_more():
    st.markdown(
        """
**How it’s typically structured (schematic):**
1) **Issuer / Originator** sells or assigns beneficial interest in assets to an **SPV**.  
2) SPV issues **Sukuk certificates** to investors; proceeds are used to acquire the assets or rights.  
3) Assets generate **permitted cashflows** (e.g., rent in Ijara) which fund **periodic profit distributions**.  
4) At maturity (or early redemption), assets are **redeemed/returned** and principal is repaid.

**Pricing in this simplified pricer:**  
- We model cashflows like a fixed-rate instrument using a **profit rate** (not interest).  
- Clean/dirty pricing, duration and convexity are computed identically to conventional bonds.  
- This is appropriate for **education and quick intuition**, and aligns with empirical findings that many rated Sukuk price close to comparable bonds of the same issuer/sovereign and tenor.

**History & Rationale (why Sukuk exist):**
- **1960s–1980s**: Modern Islamic finance takes shape to provide financing compatible with **Shariah**, which prohibits *riba* (interest), *gharar* (excessive uncertainty), and *maysir* (speculation/gambling).  
- **1990s–2000s**: First sovereign and corporate Sukuk emerge (notably in **Malaysia** and the **GCC**), using asset-linked structures (e.g., Ijara) to deliver economic returns without paying interest.  
- **2010s–today**: Sukuk become a mainstream funding tool for **sovereigns, quasi-sovereigns, and corporates** (including non‑Muslim jurisdictions) to tap a broader investor base and support **ESG/real‑economy** financing (infrastructure, housing, green projects).  
- **Why not conventional bonds?** Conventional bonds pay **interest on money**, which is seen as *riba*. Sukuk aim to link returns to **ownership of assets, usufruct, or trade/partnership**, aligning with risk‑sharing principles and avoiding interest-based lending.

**Notes & caveats:**  
- Legal form matters: **asset-based** vs **asset-backed** Sukuk can carry different **recourse** profiles.  
- Documentation is subject to **Shariah governance** and local law; real deals may include purchase undertakings, servicing, tax, and accounting nuances that this pricer abstracts away.
        """
    )
