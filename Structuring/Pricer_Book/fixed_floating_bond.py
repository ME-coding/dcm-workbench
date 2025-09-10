from __future__ import annotations

# en tête du fichier
from .core import (
    build_schedule_fixed,
    build_schedule_variable,
    present_value,
    price_from_yield,
    yield_from_price,
    macaulay_duration_convexity,
    truncate_with_redemption,
)
# DCM_Project/Pricer_Book/fixed_floating_bond.py

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from .visuals import (
    price_yield_chart,
    cashflow_breakdown_chart,
    amortization_chart,
    rate_path_chart,
)
import math

# -----------------------------
# Small CSS (same look & feel)
# -----------------------------
st.markdown(
    """
    <style>
    .inline-help { margin-top: 28px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Core utilities (replicated from the main pricer)
# =========================================================
def _build_schedule(notional: float, coupon_rate: float, freq: int, years: float,
                    structure: str = "bullet") -> pd.DataFrame:
    n = max(1, int(round(freq * years)))
    per = 1 / freq
    r_per = coupon_rate / freq
    data, outstanding = [], notional

    if structure == "bullet":
        coupon_amount = notional * r_per
        for i in range(1, n + 1):
            principal_payment = notional if i == n else 0.0
            coupon_payment = coupon_amount
            total = coupon_payment + principal_payment
            outstanding_end = outstanding - principal_payment
            data.append([i, i * per, outstanding, coupon_payment, principal_payment, total, outstanding_end])
            outstanding = outstanding_end

    elif structure == "equal_principal":
        principal_payment_const = notional / n
        for i in range(1, n + 1):
            interest_payment = outstanding * r_per
            principal_payment = principal_payment_const
            total = interest_payment + principal_payment
            outstanding_end = outstanding - principal_payment
            data.append([i, i * per, outstanding, interest_payment, principal_payment, total, outstanding_end])
            outstanding = outstanding_end

    elif structure == "annuity":
        payment = notional / n if r_per == 0 else notional * (r_per) / (1 - (1 + r_per) ** (-n))
        for i in range(1, n + 1):
            interest_payment = outstanding * r_per
            principal_payment = payment - interest_payment
            total = payment
            outstanding_end = outstanding - principal_payment
            data.append([i, i * per, outstanding, interest_payment, principal_payment, total, outstanding_end])
            outstanding = outstanding_end

    df = pd.DataFrame(
        data,
        columns=["Period","Time (years)","Outstanding (begin)","Coupon/Profit","Principal","Total CF","Outstanding (end)"],
    )
    return df


def _build_schedule_variable(notional: float, rates_annual: List[float], freq: int,
                             structure: str = "bullet") -> pd.DataFrame:
    n = max(1, len(rates_annual))
    per = 1 / freq
    data, outstanding = [], notional

    if structure == "bullet":
        for i in range(1, n + 1):
            r_per = rates_annual[i - 1] / freq
            coupon_payment = outstanding * r_per
            principal_payment = notional if i == n else 0.0
            total = coupon_payment + principal_payment
            outstanding_end = outstanding - principal_payment
            data.append([i, i * per, outstanding, coupon_payment, principal_payment, total, outstanding_end])
            outstanding = outstanding_end
    elif structure == "equal_principal":
        principal_payment_const = notional / n
        for i in range(1, n + 1):
            r_per = rates_annual[i - 1] / freq
            interest_payment = outstanding * r_per
            principal_payment = principal_payment_const
            total = interest_payment + principal_payment
            outstanding_end = outstanding - principal_payment
            data.append([i, i * per, outstanding, interest_payment, principal_payment, total, outstanding_end])
            outstanding = outstanding_end

    df = pd.DataFrame(
        data,
        columns=["Period","Time (years)","Outstanding (begin)","Coupon/Profit","Principal","Total CF","Outstanding (end)"],
    )
    return df


def _present_value(cashflows: pd.Series, ytm_annual: float, freq: int) -> float:
    y = max(1e-12, ytm_annual / freq)
    disc = np.array([(1 + y) ** i for i in range(1, len(cashflows) + 1)])
    return float(np.sum(cashflows.values / disc))


def price_from_yield(schedule: pd.DataFrame, ytm_annual: float, freq: int,
                     notional: float, accrued_frac: float = 0.0) -> Tuple[float,float,float]:
    pv = _present_value(schedule["Total CF"], ytm_annual, freq)
    scale = 100.0 / notional
    clean_price = pv * scale
    try:
        r_per_inferred = (schedule.loc[0, "Coupon/Profit"] / schedule.loc[0, "Outstanding (begin)"]) if schedule.loc[0, "Outstanding (begin)"] > 0 else 0.0
    except Exception:
        r_per_inferred = 0.0
    current_outstanding = schedule.loc[0, "Outstanding (begin)"] if len(schedule) > 0 else notional
    accrued = (current_outstanding * r_per_inferred * accrued_frac) * scale
    return float(clean_price), float(accrued), float(clean_price + accrued)


def yield_from_price(schedule: pd.DataFrame, target_clean_price_per_100: float, freq: int,
                     notional: float, tol: float = 1e-8, max_iter: int = 200) -> float:
    target_pv = target_clean_price_per_100 * notional / 100.0
    low, high = -0.99, 1.5
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        pv = _present_value(schedule["Total CF"], mid, freq)
        if abs(pv - target_pv) < tol:
            return mid
        if pv > target_pv:
            low = mid
        else:
            high = mid
    return mid


def macaulay_duration_convexity(schedule: pd.DataFrame, ytm_annual: float, freq: int):
    y = max(1e-12, ytm_annual / freq)
    cfs = schedule["Total CF"].values
    disc = np.array([(1 + y) ** i for i in range(1, len(cfs) + 1)])
    pv_cf = cfs / disc
    price = float(np.sum(pv_cf))
    t = np.arange(1, len(cfs) + 1)
    mac_years = (np.sum(t * pv_cf) / price) / freq if price > 0 else 0.0
    mod_duration = mac_years / (1 + y)
    conv_years2 = (np.sum(cfs * t * (t + 1) / ((1 + y) ** (t + 2))) / price) / (freq ** 2) if price > 0 else 0.0
    return price, mac_years, mod_duration, conv_years2


# =========================================================
# Helper charts (Altair)
# =========================================================
def _chart_price_yield(y_grid_pct: np.ndarray, prices: np.ndarray, title: str = "Price–Yield Curve"):
    df = pd.DataFrame({"Yield (%)": y_grid_pct, "Clean Price (per 100)": prices})
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("Yield (%):Q", title="Yield (%)"),
            y=alt.Y("Clean Price (per 100):Q", title="Clean Price (per 100)"),
            tooltip=["Yield (%)","Clean Price (per 100)"],
        ).properties(title=title, height=260)
    )

def _chart_cashflows(schedule: pd.DataFrame):
    cf_df = schedule[["Time (years)","Coupon/Profit","Principal"]].melt(
        id_vars=["Time (years)"], var_name="Component", value_name="Amount"
    )
    return (
        alt.Chart(cf_df)
        .mark_bar()
        .encode(
            x=alt.X("Time (years):Q"),
            y=alt.Y("Amount:Q", title="Cash Flow"),
            color="Component:N",
            tooltip=["Time (years)","Component","Amount"],
        ).properties(title="Cash Flow Breakdown", height=260)
    )

def _chart_rate_path(times: np.ndarray, rates_pct: np.ndarray, title: str = "Rate path (%)"):
    df = pd.DataFrame({"Time (years)": times, "Rate (%)": rates_pct})
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(x="Time (years):Q", y=alt.Y("Rate (%):Q"), tooltip=["Time (years)","Rate (%)"])
        .properties(title=title, height=260)
    )

# =========================================================
# Builders for FRN & Fixed-to-Floating
# =========================================================
def build_fixed_schedule(notional: float, coupon_pct: float, freq: int, years: float, structure: str):
    df = _build_schedule(notional, coupon_pct/100.0, freq, years, structure)
    df["Total CF"] = df["Coupon/Profit"] + df["Principal"]
    return df

def build_frn_schedule(notional: float, base_ref_pct: float, spread_bps: float,
                       freq: int, years: float, structure: str,
                       per_period_refs_pct: List[float] | None = None):
    n = max(1, int(round(freq * years)))
    if per_period_refs_pct is None:
        rates_annual = [ (base_ref_pct + spread_bps/100.0)/100.0 ] * n
        path_pct = np.array([base_ref_pct + spread_bps/100.0] * n)
    else:
        # pad/truncate to n
        ref_list = (per_period_refs_pct + [per_period_refs_pct[-1]] * (n - len(per_period_refs_pct)))[:n]
        rates_annual = [ (r + spread_bps/100.0)/100.0 for r in ref_list ]
        path_pct = np.array([ r + spread_bps/100.0 for r in ref_list ])
    df = _build_schedule_variable(notional, rates_annual, freq, structure)
    df["Total CF"] = df["Coupon/Profit"] + df["Principal"]
    times = np.arange(1, n+1) * (1/freq)
    return df, times, path_pct

def build_fix_to_float_schedule(notional: float, fixed_coupon_pct: float, years_fixed: float,
                                float_ref_pct: float, float_spread_bps: float,
                                freq: int, years_total: float, structure: str,
                                per_period_refs_pct: List[float] | None = None):
    n = max(1, int(round(freq * years_total)))
    per = 1/freq
    times = np.arange(1, n+1)*per
    rates_pct = np.zeros(n)
    k_fix = min(n, int(round(freq * years_fixed)))
    # fixed phase
    rates_pct[:k_fix] = fixed_coupon_pct
    # floating phase
    if k_fix < n:
        if per_period_refs_pct is None:
            rates_pct[k_fix:] = float_ref_pct + float_spread_bps/100.0
        else:
            # align provided refs to floating segment length
            refs = (per_period_refs_pct + [per_period_refs_pct[-1]] * (n-k_fix - len(per_period_refs_pct)))[:(n-k_fix)]
            rates_pct[k_fix:] = np.array(refs) + float_spread_bps/100.0
    rates_annual = (rates_pct/100.0).tolist()
    df = _build_schedule_variable(notional, rates_annual, freq, structure)
    df["Total CF"] = df["Coupon/Profit"] + df["Principal"]
    return df, times, rates_pct

# =========================================================
# UI
# =========================================================
def render():
    st.subheader("Structuring Desk — Fixed / Floating / Fixed‑to‑Floating")

    # ---- Product choice
    product = st.radio("Select product", ["Fixed Rate Bond","Floating Rate Note (FRN)","Fixed‑to‑Floating"], horizontal=True)

    # ---- Common inputs
    default_notional, default_years, default_coupon, default_yield = 1_000_000.0, 5.0, 4.00, 5.00
    f_map = {"Annual": 1, "Semi‑annual": 2, "Quarterly": 4}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        notional = st.number_input("Notional", min_value=1_000.0, value=default_notional, step=1_000.0, format="%.2f")
    with c2:
        years = st.number_input("Maturity (years)", min_value=0.25, value=default_years, step=0.25, format="%.2f")
    with c3:
        freq = f_map[st.selectbox("Coupon frequency", list(f_map.keys()), index=1)]
    with c4:
        structure = st.selectbox("Structure", ["bullet","equal_principal","annuity"], index=0)

    # ---- Product-specific parameters
    rate_path_times = None
    rate_path_pct = None

    if product == "Fixed Rate Bond":
        coupon_pct = st.slider("Coupon rate (%)", 0.00, 12.00, default_coupon, 0.05)
        schedule = build_fixed_schedule(notional, coupon_pct, freq, years, structure)

    elif product == "Floating Rate Note (FRN)":
        st.markdown("#### Reference & Spread")
        mode = st.radio("Reference input", ["Flat rate","Per‑period table"], horizontal=True)
        spread_bps = st.slider("Spread (bp)", 0.0, 600.0, 120.0, 5.0)
        if mode == "Flat rate":
            ref_pct = st.slider("Flat reference rate (%)", 0.00, 10.00, 3.00, 0.05)
            schedule, rate_path_times, rate_path_pct = build_frn_schedule(
                notional, ref_pct, spread_bps, freq, years, structure, None
            )
        else:
            n = max(1, int(round(freq * years)))
            if "frn_table_ff" not in st.session_state:
                st.session_state.frn_table_ff = pd.DataFrame({"Ref rate (%)": [3.00]*n})
            frn_table = st.data_editor(
                st.session_state.frn_table_ff, num_rows="dynamic", use_container_width=True,
                help="Provide per‑period reference rates in %."
            )
            st.session_state.frn_table_ff = frn_table.copy()
            refs = frn_table["Ref rate (%)"].tolist()
            schedule, rate_path_times, rate_path_pct = build_frn_schedule(
                notional, 0.0, spread_bps, freq, years, structure, refs
            )

    else:  # Fixed‑to‑Floating
        st.markdown("#### Fixed → Floating structure")
        years_fixed = st.slider("Fixed phase length (years)", 0.5, years, min(years, 3.0), 0.5)
        fixed_coupon_pct = st.slider("Fixed coupon (%)", 0.00, 12.00, 4.00, 0.05)

        mode = st.radio("Floating reference", ["Flat rate","Per‑period table"], horizontal=True)
        float_spread_bps = st.slider("Floating spread (bp)", 0.0, 600.0, 200.0, 5.0)
        if mode == "Flat rate":
            float_ref_pct = st.slider("Flat reference rate (%)", 0.00, 10.00, 3.00, 0.05)
            schedule, rate_path_times, rate_path_pct = build_fix_to_float_schedule(
                notional, fixed_coupon_pct, years_fixed, float_ref_pct, float_spread_bps,
                freq, years, structure, None
            )
        else:
            n_float = max(0, int(round(freq * (years - years_fixed))))
            if "ftf_table_ff" not in st.session_state:
                st.session_state.ftf_table_ff = pd.DataFrame({"Ref rate (%)": [3.00]*max(1, n_float)})
            ftf_table = st.data_editor(
                st.session_state.ftf_table_ff, num_rows="dynamic", use_container_width=True,
                help="Provide per‑period reference rates (floating phase only) in %."
            )
            st.session_state.ftf_table_ff = ftf_table.copy()
            refs = ftf_table["Ref rate (%)"].tolist()
            schedule, rate_path_times, rate_path_pct = build_fix_to_float_schedule(
                notional, fixed_coupon_pct, years_fixed, 0.0, float_spread_bps,
                freq, years, structure, refs
            )

    # ---- Solve block
    st.markdown("#### Solve")
    s1, s2 = st.columns(2)
    solve_mode = st.radio("Solve for", ["Price (given Yield)","Yield (given Clean Price)"], horizontal=True)
    if solve_mode == "Price (given Yield)":
        with s1:
            ytm = st.slider("Yield to Maturity (%)", -5.00, 20.00, 5.00, 0.05) / 100.0
    else:
        with s1:
            clean_target = st.slider("Target clean price (per 100)", 50.0, 150.0, 100.0, 0.5)
    with s2:
        grid_bps = st.slider("Price–Yield grid width (bps)", 50, 500, 200, 25)

    # ---- Pricing & risk
    if solve_mode == "Price (given Yield)":
        clean, accrued, dirty = price_from_yield(schedule, ytm, freq, notional, accrued_frac=0.0)
        _, mac_dur, mod_dur, conv = macaulay_duration_convexity(schedule, ytm, freq)
    else:
        ytm = yield_from_price(schedule, clean_target, freq, notional)
        clean, accrued, dirty = price_from_yield(schedule, ytm, freq, notional, accrued_frac=0.0)
        _, mac_dur, mod_dur, conv = macaulay_duration_convexity(schedule, ytm, freq)

    # ---- KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clean Price (per 100)", f"{clean:,.4f}")
    k2.metric("Accrued (per 100)", f"{accrued:,.4f}")
    k3.metric("Dirty Price (per 100)", f"{dirty:,.4f}")
    k4.metric("Yield to Maturity (%)", f"{ytm*100:,.4f}")
    k5, k6, k7 = st.columns(3)
    k5.metric("Macaulay Duration (yrs)", f"{mac_dur:,.4f}")
    k6.metric("Modified Duration (yrs)", f"{mod_dur:,.4f}")
    k7.metric("Convexity (yrs²)", f"{conv:,.4f}")

    fixed_floating_bond_ui = render

    # =========================================================
    # Charts
    # =========================================================
    st.markdown("### Results & Charts")
    left, right = st.columns(2)

    # Price–Yield
    #y0 = ytm
    #bps = grid_bps / 10_000.0
    #y_grid = np.linspace(max(-0.99, y0 - bps), y0 + bps, 41)
    #prices = np.array([price_from_yield(schedule, y, freq, notional, 0.0)[0] for y in y_grid])
    #with left:
        #st.altair_chart(_chart_price_yield(y_grid*100, prices), use_container_width=True)


    # --- Price–Yield interactif ---
    with left:
        st.subheader("Price–Yield")

        # 1) Contrôle de l'axe X (autour du YTM en ± bp)
        #    Valeur par défaut = l'ancien grid_bps si dispo, sinon 200 bps
        default_bps = int(grid_bps) if "grid_bps" in locals() else 200
        x_half_range_bps = st.slider("Écart sur l’axe X (± basis points)", 25, 1000, default_bps, step=25)

        # 2) Recalcule la grille de rendements selon le slider
        y0 = ytm
        bps = x_half_range_bps / 10_000.0
        y_grid = np.linspace(max(-0.99, y0 - bps), y0 + bps, 41)

        # 3) Calcule les prix
        prices = np.array([price_from_yield(schedule, y, freq, notional, 0.0)[0] for y in y_grid])

        # 4) Contrôle de l’axe Y
        auto_y = st.checkbox("Axe Y automatique", value=True)
        if auto_y:
            y_domain = None
        else:
            p_min, p_max = float(prices.min()), float(prices.max())
            # marge +/- 5% autour des valeurs observées
            low_default  = max(0.0, p_min * 0.95)
            high_default = p_max * 1.05
            y_min, y_max = st.slider(
                "Plage de prix (axe Y)",
                min_value=float(max(0.0, p_min * 0.8)),
                max_value=float(p_max * 1.2),
                value=(float(low_default), float(high_default)),
            )
            y_domain = [y_min, y_max]

        # 5) Construit le DataFrame
        df_plot = pd.DataFrame({
            "Yield (%)": y_grid * 100.0,
            "Price": prices
        })

        # 6) Chart Altair interactif (pan/zoom + hover)
        x_domain = [df_plot["Yield (%)"].min(), df_plot["Yield (%)"].max()]
        hover = alt.selection_point(fields=["Yield (%)"], nearest=True, on="mouseover", empty="none")

        base = alt.Chart(df_plot)

        line = base.mark_line().encode(
            x=alt.X("Yield (%):Q", scale=alt.Scale(domain=x_domain), title="Yield (%)"),
            y=alt.Y("Price:Q", scale=None if y_domain is None else alt.Scale(domain=y_domain), title="Price"),
            tooltip=[alt.Tooltip("Yield (%):Q", format=".2f"), alt.Tooltip("Price:Q", format=",.2f")]
        )

        points = base.mark_circle(size=60).encode(
            x="Yield (%):Q",
            y="Price:Q",
            opacity=alt.condition(hover, alt.value(1), alt.value(0))
        ).add_params(hover)

        chart = (line + points).interactive()  # active pan/zoom

        st.altair_chart(chart, use_container_width=True)

    # Cash‑flows
    with right:
        st.altair_chart(_chart_cashflows(schedule), use_container_width=True)

    # Only show the rate path chart (Outstanding over time removed)
    if rate_path_times is not None and rate_path_pct is not None:
        title = "Coupon/Profit rate path (%)"
        st.altair_chart(_chart_rate_path(rate_path_times, rate_path_pct, title), use_container_width=True)
    else:
        st.info("Rate path chart appears when the product has variable rates (FRN / Fixed-to-Floating).")


    # ---- Cash flow table & export
    st.markdown("### Cash flow table")
    table = schedule.copy()
    for c in ["Outstanding (begin)","Coupon/Profit","Principal","Total CF","Outstanding (end)"]:
        table[c] = table[c].map(lambda x: round(float(x), 6))
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "Download cash flow table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{product.lower().replace(' ','_')}_cash_flows.csv",
        mime="text/csv",
    )

    # =========================================================
    # Theory / Learn more / Example
    # =========================================================
    st.markdown("---")
    st.markdown("## Definitions")

    st.markdown(
        """
**Fixed Rate Bond** — coupon **constant** through life (e.g., 4% p.a.).  
**Floating Rate Note (FRN)** — coupon **index‑linked** (e.g., Euribor 3m + 120 bp).  
**Fixed‑to‑Floating Notes** — **hybrid**: start **fixed**, then switch to **floating** after a given date (often the first call).

### Why Fixed‑to‑Floating exists?
- **Issuer (esp. banks, Tier 2 / AT1)**: fixed funding cost initially; later coupons float with market rates, often closer to asset yields.  
- **Investor**: stable income in phase 1; **rate‑rise protection** in phase 2 via floating coupons.
        """
    )

    # Learn more (modal or expander)
    if hasattr(st, "dialog"):
        @st.dialog("Learn more — Fixed vs Floating vs Fixed‑to‑Floating")
        def _more():
            _render_learn_more()
        if st.button("Learn More"):
            _more()
    else:
        with st.expander("Learn more — Fixed vs Floating vs Fixed‑to‑Floating"):
            _render_learn_more()

    # Example PDF
    st.markdown("### Example — Download")
    pdf_path = Path(__file__).resolve().parent.parent.parent / "Library" / "Fixed to Floating Rate Notes Example - J.P. Morgan Chase & Co (2021).pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download: Fixed‑to‑Floating Notes — J.P. Morgan Chase & Co (2021) (PDF)",
                data=f.read(),
                file_name=pdf_path.name,
                mime="application/pdf",
            )
    else:
        st.info(f"Place the example PDF at **{pdf_path}** (filename must match exactly).")

def _render_learn_more():
    st.markdown(
        """
### How we price here (simplified, desk‑style)
1) **Build schedule** of cashflows from per‑period annual **rate path**:  
   - Fixed: constant rate.  
   - FRN: `reference + spread` (flat or per‑period).  
   - Fixed‑to‑Floating: fixed coupon for *Y* years, then floating (`ref + spread`).  
2) **Discount** with an input **Yield to Maturity** (solve for price) or **solve YTM** from a target clean price (bisection).  
3) Compute **duration & convexity** from discounted CFs.  
4) Charts: **Price–Yield** (local sensitivity), **Cash‑flows**, **Outstanding**, and **Rate path** where relevant.

> Notes: Production pricers use a **curve** (OIS/credit) & day‑count. Here we keep it **clean & educational**.
        """
    )

# =========================
# Sources / Notes (comments)
# =========================
# Internal logic mirrors DCM_Project/Page1/pricer.py utilities for consistency.
# Explanations inspired by the user brief on Fixed / Floating / Fixed‑to‑Floating.
# Example PDF expected at: Library/"Fixed to Floating Rate Notes Example - J.P. Morgan Chase & Co (2021)".pdf
