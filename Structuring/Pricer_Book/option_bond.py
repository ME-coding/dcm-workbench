# DCM_Project/Pricer_Book/option_bond.py
from __future__ import annotations  # ✅ must be the very first statement

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Core building blocks (already in your Pricer_Book/core.py)
from .core import (
    build_schedule_fixed,
    build_schedule_variable,
    present_value,
    price_from_yield,
    yield_from_price,
    macaulay_duration_convexity,
    truncate_with_redemption,
)

# Shared chart helpers (see Pricer_Book/visuals.py)
from .visuals import (
    price_yield_chart,
    cashflow_breakdown_chart,
    amortization_chart,
    rate_path_chart,
)
# --------------------------------------------------------------------------------------
# Call/Put pricing utilities (kept here so both products live in a single module)
# --------------------------------------------------------------------------------------

def price_callable_from_yield(schedule_base: pd.DataFrame, call_table: pd.DataFrame, ytm: float, freq: int) -> pd.DataFrame:
    """
    Price a callable bond across scenarios:
      - 'To Maturity'
      - Each call date at its call price (% of par)
    Uses schedule truncation + redemption at call price.
    """
    notional = float(schedule_base.loc[0, "Outstanding (begin)"]) if len(schedule_base) else 100.0
    rows = [("To Maturity", price_from_yield(schedule_base, ytm, freq, notional, 0.0)[0])]
    if call_table is not None and len(call_table) > 0:
        for _, r in call_table.iterrows():
            t, px = float(r.get("Call at (years)", 0)), float(r.get("Call Price (% of par)", 100.0))
            if t <= 0:
                continue
            df = truncate_with_redemption(schedule_base, t, freq, notional * (px / 100.0))
            if df.empty:
                continue
            rows.append((f"Called @ {t:g}y ({px:.2f}%)", price_from_yield(df, ytm, freq, notional, 0.0)[0]))
    out = pd.DataFrame(rows, columns=["Scenario", "Clean Price"]).sort_values("Clean Price").reset_index(drop=True)
    return out


def ytw_from_price(schedule_base: pd.DataFrame, call_table: pd.DataFrame, clean_px: float, freq: int) -> pd.DataFrame:
    """
    Yield-to-worst table for callable bond given a target clean price.
    """
    notional = float(schedule_base.loc[0, "Outstanding (begin)"]) if len(schedule_base) else 100.0
    rows = [("To Maturity", yield_from_price(schedule_base, clean_px, freq, notional) * 100)]
    if call_table is not None and len(call_table) > 0:
        for _, r in call_table.iterrows():
            t, px = float(r.get("Call at (years)", 0)), float(r.get("Call Price (% of par)", 100.0))
            if t <= 0:
                continue
            df = truncate_with_redemption(schedule_base, t, freq, notional * (px / 100.0))
            if df.empty:
                continue
            rows.append((f"Called @ {t:g}y ({px:.2f}%)", yield_from_price(df, clean_px, freq, notional) * 100))
    out = pd.DataFrame(rows, columns=["Scenario", "Yield (%)"])
    out["Is YTW"] = out["Yield (%)"] == out["Yield (%)"].min()
    return out.sort_values("Yield (%)").reset_index(drop=True)


def price_puttable_from_yield(schedule_base: pd.DataFrame, put_table: pd.DataFrame, ytm: float, freq: int) -> pd.DataFrame:
    """
    Price a puttable bond across scenarios:
      - 'To Maturity'
      - Each put date at its put price (% of par)
    Uses schedule truncation + redemption at put price.
    """
    notional = float(schedule_base.loc[0, "Outstanding (begin)"]) if len(schedule_base) else 100.0
    rows = [("To Maturity", price_from_yield(schedule_base, ytm, freq, notional, 0.0)[0])]
    if put_table is not None and len(put_table) > 0:
        for _, r in put_table.iterrows():
            t, px = float(r.get("Put at (years)", 0)), float(r.get("Put Price (% of par)", 100.0))
            if t <= 0:
                continue
            df = truncate_with_redemption(schedule_base, t, freq, notional * (px / 100.0))
            if df.empty:
                continue
            rows.append((f"Put @ {t:g}y ({px:.2f}%)", price_from_yield(df, ytm, freq, notional, 0.0)[0]))
    out = pd.DataFrame(rows, columns=["Scenario", "Clean Price"])
    out["Is Investor-Optimal"] = out["Clean Price"] == out["Clean Price"].max()
    return out.sort_values("Clean Price").reset_index(drop=True)


def ytb_from_price(schedule_base: pd.DataFrame, put_table: pd.DataFrame, clean_px: float, freq: int) -> pd.DataFrame:
    """
    Yield table for puttable bond given a target clean price (investor-best yield highlighted).
    """
    notional = float(schedule_base.loc[0, "Outstanding (begin)"]) if len(schedule_base) else 100.0
    rows = [("To Maturity", yield_from_price(schedule_base, clean_px, freq, notional) * 100)]
    if put_table is not None and len(put_table) > 0:
        for _, r in put_table.iterrows():
            t, px = float(r.get("Put at (years)", 0)), float(r.get("Put Price (% of par)", 100.0))
            if t <= 0:
                continue
            df = truncate_with_redemption(schedule_base, t, freq, notional * (px / 100.0))
            if df.empty:
                continue
            rows.append((f"Put @ {t:g}y ({px:.2f}%)", yield_from_price(df, clean_px, freq, notional) * 100))
    out = pd.DataFrame(rows, columns=["Scenario", "Yield (%)"])
    out["Is Investor-Best"] = out["Yield (%)"] == out["Yield (%)"].max()
    return out.sort_values("Yield (%)").reset_index(drop=True)


# Simple scenario bar chart (kept local so the module is self-contained)
def _scenario_bar_chart(df: pd.DataFrame, value_col: str, title: str):
    # Display smallest/lowest at top by default
    base = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{value_col}:Q", title=value_col),
        y=alt.Y("Scenario:N", sort="-x", title="Scenario"),
        tooltip=["Scenario", value_col],
        color=alt.condition(
            alt.datum.get("Is YTW", False) | alt.datum.get("Is Investor-Optimal", False) | alt.datum.get("Is Investor-Best", False),
            alt.value("#6c9cef"),
            alt.value("#a9b1c7"),
        ),
    ).properties(title=title, height=260)
    return base


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

def render():
    st.subheader("Structuring Desk — Option Bonds (Callable / Puttable)")

    # ----- Base schedule inputs -----
    default_notional, default_years, default_coupon, default_yield = 1_000_000.0, 5.0, 5.00, 5.00
    f_map = {"Annual": 1, "Semi-annual": 2, "Quarterly": 4}

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        notional = st.number_input("Notional", min_value=1_000.0, value=default_notional, step=1_000.0, format="%.2f")
    with r1c2:
        years = st.number_input("Maturity (years)", min_value=0.25, value=default_years, step=0.25, format="%.2f")
    with r1c3:
        freq_label = st.selectbox("Coupon frequency", sorted(f_map.keys()))
        freq = f_map[freq_label]
    with r1c4:
        accrued_frac = st.slider("Elapsed in current period (%)", 0, 100, 0,
                                 help="Accrued interest approximation as % of the current coupon period.") / 100.0

    structure = st.selectbox("Structure", ["bullet", "equal_principal", "annuity"], index=0)
    coupon_rate = st.number_input("Coupon rate (%)", min_value=0.0, value=default_coupon, step=0.25, format="%.4f") / 100.0

    base_schedule = build_schedule_fixed(notional, coupon_rate, freq, years, structure)
    base_schedule["Total CF"] = base_schedule["Coupon/Profit"] + base_schedule["Principal"]

    option_bond_ui = render

    # ----- Option selector & schedules -----
    st.markdown("#### Embedded option")
    opt_type = st.radio("Select", ["Callable Bond", "Puttable Bond"], horizontal=True)

    if opt_type == "Callable Bond":
        if "call_table" not in st.session_state:
            st.session_state.call_table = pd.DataFrame({
                "Call at (years)": [2.0, 3.0, 4.0],
                "Call Price (% of par)": [101.0, 100.5, 100.0],
            })
        call_table = st.data_editor(
            st.session_state.call_table, num_rows="dynamic", use_container_width=True, key="call_editor",
            help="Add, edit, or remove call dates and prices (as % of par)."
        )
        st.session_state.call_table = call_table.copy()
        put_table = None
    else:
        if "put_table" not in st.session_state:
            st.session_state.put_table = pd.DataFrame({
                "Put at (years)": [2.0, 4.0],
                "Put Price (% of par)": [100.0, 100.0],
            })
        put_table = st.data_editor(
            st.session_state.put_table, num_rows="dynamic", use_container_width=True, key="put_editor",
            help="Investor put options: redemption at given dates/prices (as % of par)."
        )
        st.session_state.put_table = put_table.copy()
        call_table = None

    # ----- Solve block -----
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

    # ----- Compute KPIs -----
    if solve_mode == "Price (given Yield)":
        clean, accrued, dirty = price_from_yield(base_schedule, ytm, freq, notional, accrued_frac=accrued_frac)
        _, mac_dur, mod_dur, conv = macaulay_duration_convexity(base_schedule, ytm, freq)

        scenario_df = None
        if opt_type == "Callable Bond":
            scenario_df = price_callable_from_yield(base_schedule, call_table, ytm, freq)
            # Worst for investor is the minimum price
            worst = scenario_df.iloc[0]
            st.caption(f"**Yield-to-Worst perspective (price view)** — worst scenario for the investor: "
                       f"{worst['Scenario']} → {worst['Clean Price']:.4f} per 100.")
        else:
            scenario_df = price_puttable_from_yield(base_schedule, put_table, ytm, freq)
            best = scenario_df.loc[scenario_df['Clean Price'].idxmax()]
            st.caption(f"**Investor-optimal price** — best scenario for the investor: "
                       f"{best['Scenario']} → {best['Clean Price']:.4f} per 100.")

    else:
        ytm = yield_from_price(base_schedule, clean_target, freq, notional)
        clean, accrued, dirty = price_from_yield(base_schedule, ytm, freq, notional, accrued_frac=accrued_frac)
        _, mac_dur, mod_dur, conv = macaulay_duration_convexity(base_schedule, ytm, freq)

        scenario_df = None
        if opt_type == "Callable Bond":
            scenario_df = ytw_from_price(base_schedule, call_table, clean, freq)
            ytw_row = scenario_df.iloc[0]
            st.caption(f"**Yield-to-Worst (issuer option)**: {ytw_row['Scenario']} → {ytw_row['Yield (%)']:.4f}%.")
        else:
            scenario_df = ytb_from_price(base_schedule, put_table, clean, freq)
            best = scenario_df.iloc[-1]
            st.caption(f"**Investor-best yield**: {best['Scenario']} → {best['Yield (%)']:.4f}%.")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clean Price (per 100)", f"{clean:,.4f}")
    k2.metric("Accrued (per 100)", f"{accrued:,.4f}")
    k3.metric("Dirty Price (per 100)", f"{dirty:,.4f}")
    k4.metric("Yield to Maturity (%)", f"{ytm*100:,.4f}")

    k5, k6, k7 = st.columns(3)
    k5.metric("Macaulay Duration (yrs)", f"{mac_dur:,.4f}")
    k6.metric("Modified Duration (yrs)", f"{mod_dur:,.4f}")
    k7.metric("Convexity (yrs²)", f"{conv:,.4f}")

    # ----- Charts -----
    st.markdown("### Results & Charts")

    left, right = st.columns(2)

    # Price–Yield
    y0 = ytm
    bps = grid_bps / 10_000.0
    y_grid = np.linspace(max(-0.99, y0 - bps), y0 + bps, 41)
    prices = np.array([price_from_yield(base_schedule, y, freq, notional, 0.0)[0] for y in y_grid])
    with left:
        st.altair_chart(price_yield_chart(y_grid * 100, prices), use_container_width=True)

    # Cash Flow Breakdown
    with right:
        st.altair_chart(cashflow_breakdown_chart(base_schedule), use_container_width=True)

    e1, e2 = st.columns(2)
    with e1:
        st.altair_chart(amortization_chart(base_schedule), use_container_width=True)
    with e2:
        if scenario_df is not None:
            if "Clean Price" in scenario_df.columns:
                st.altair_chart(_scenario_bar_chart(scenario_df, "Clean Price", "Scenario Prices"), use_container_width=True)
            else:
                st.altair_chart(_scenario_bar_chart(scenario_df, "Yield (%)", "Scenario Yields"), use_container_width=True)
        else:
            st.caption("Add at least one call/put date to display scenario bars.")

    # ----- Cash flow table & export -----
    st.markdown("### Cash flow table")
    table = base_schedule.copy()
    for c in ["Outstanding (begin)", "Coupon/Profit", "Principal", "Total CF", "Outstanding (end)"]:
        table[c] = table[c].map(lambda x: round(float(x), 6))
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "Download cash flow table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="option_bond_cash_flows.csv",
        mime="text/csv",
    )

    # ----------------------------------------------------------------------------------
    # Theory & Learn More
    # ----------------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## Callable & Puttable Bonds — Theory (Quick Primer)")

    st.markdown(
        """
**What are they?**  
- **Callable bond**: the **issuer** holds an option to **redeem early** at specified dates/prices → upside to issuer if rates fall.  
- **Puttable bond**: the **investor** holds an option to **redeem early** at specified dates/prices → downside protection if rates rise or spreads widen.

**How we compute (this pricer):**  
1) Build the **base schedule** (coupon + principal) from notional, coupon rate, frequency and structure.  
2) For each call/put date **t**, **truncate** the schedule at *t* and replace the last cash flow with the **redemption price** (% of par).  
3) **Pricing mode**  
   - *Price (given Yield)*: discount each scenario’s cash flows at the input **YTM** to get a **clean price** per 100.  
   - *Yield (given Clean Price)*: solve for the **YTM** that matches the target clean price (root‑finding), for each scenario.  
4) Highlight the relevant scenario:  
   - Callable → **Yield‑to‑Worst** (minimum yield for investor).  
   - Puttable → **Investor‑optimal price / best yield**.

**Why this is reasonable:**  
- Under standard assumptions (deterministic discounting / no stochastic exercise boundary), the value of an embedded option can be represented by comparing **path‑independent cash‑flow truncations** at contractual exercise dates.  
- This mirrors textbook treatments of **callable/puttable bonds** and gives the right intuition for **price–yield** behavior and **YTW** reporting in primary/secondary DCM workflows.

**Notes & caveats:**  
- Real issuance embeds **notice periods, make‑whole provisions, soft calls, step schedules**, or business‑day adjustments that we abstract.  
- In rigorous sell‑side pricing, callable/puttable bonds are often valued via **short‑rate/interest‑rate trees** (e.g., Hull–White) with **optimal exercise**; our truncation approach is a clean educational simplification aligned with DCM term‑sheet economics.
        """
    )

    # Learn More (modal/expander)
    if hasattr(st, "dialog"):
        @st.dialog("Learn more about callable/puttable bonds")
        def _learn_more():
            _render_learn_more()
        if st.button("Learn More"):
            _learn_more()
    else:
        with st.expander("Learn More"):
            _render_learn_more()

    # Example PDF links (place files in DCM_Project/Library/)
    st.markdown("### Examples — Open")

    c_pdf = Path(__file__).resolve().parent.parent / "Library" / "Callable Bond Example - UniCredit (2024).pdf"
    p_pdf = Path(__file__).resolve().parent.parent / "Library" / "Puttable Bond Example - Legrand (2025).pdf"

    cols = st.columns(2)
    with cols[0]:
        if c_pdf.exists():
            st.markdown(
                f"""
                <a href="./Library/{c_pdf.name}" target="_blank">
                    📖 Open: Callable Bond — UniCredit (2024) (PDF)
                </a>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(f"Place the callable example at **{c_pdf}**.")

    with cols[1]:
        if p_pdf.exists():
            st.markdown(
            f"""
                <a href="./Library/{p_pdf.name}" target="_blank">
                    📖 Open: Puttable Bond — Legrand (2025) (PDF)
                </a>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(f"Place the puttable example at **{p_pdf}**.")

def _render_learn_more():
    st.markdown(
        """
### Deeper dive — embedded options math (simplified)
Let \\( CF_t \\) denote the cash flow at time \\( t \\) (in periods) and \\( y \\) the annual YTM with frequency \\( f \\).
We discount per period at \\( y/f \\). The **present value** is:

\\[
P(y) = \\sum_{i=1}^{N} \\frac{CF_i}{\\left(1 + \\frac{y}{f}\\right)^i}
\\]

**Callable scenario @ T\_c, price x% of par**  
- Replace all cash flows beyond \\( T_c \\) with a single redemption \\( x\\% \\times \\text{par} \\) at \\( T_c \\).  
- Price is \\( P\\_{call}(y) \\) from the truncated series.  
- For **YTW**, we compute yields from a **target clean price** \\( \\tilde{P} \\) by solving \\( P\\_{scenario}(y)=\\tilde{P} \\).

**Puttable scenario @ T\_p, price x% of par**  
- Same truncation logic; investor can redeem.  
- The **investor‑optimal** price is the **maximum** across scenarios (and “best yield” is the maximum yield when solving from a price).

> This matches the “cash‑flow replacement at exercise date” approach and is consistent with DCM term‑sheet economics.  
> For advanced models, replace discounting with lattice/short‑rate models and compute optimal exercise by backward induction.

**Visualization tips**  
- *Price–Yield curve* shows convexity; callable bonds typically exhibit **negative convexity** around call region.  
- *Scenario bars* give an immediate view of **worst/best** cases across the option schedule.  
- *Amortization* helps explain **equal‑principal** vs **bullet** shapes.

**References (kept lightweight):**
- Hull, *Options, Futures, and Other Derivatives* — callable/puttable valuation by trees and yield conventions.  
- Your attached GitHub project for option pricing scaffolding (binomial/MC ideas for future extensions).
        """
    )


# --------------------------------------------------------------------------------------
# Sources (left as code comments at user request)
# --------------------------------------------------------------------------------------
# Hull, J. (2014) Options, Futures and Other Derivatives (Callable/Puttable bond treatment)
# Internal inspiration from user's uploaded GitHub project structure (binomial tree, MC)
