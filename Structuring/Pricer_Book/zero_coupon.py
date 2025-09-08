# DCM_Project/Pricer_Book/zero_coupon.py
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

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---- Core (mêmes conventions que les autres modules) ----
from .core import (
    build_schedule_fixed,         # (notional, rate_annual, freq, years, structure)
    price_from_yield,
    yield_from_price,
    macaulay_duration_convexity,
)

# ---- Shared charts ----
from .visuals import (
    price_yield_chart,
    cashflow_breakdown_chart,
    amortization_chart,
    rate_path_chart,
)
import math

# =========================================================
# Utilities — schedule & Monte Carlo (CIR) for ZCB
# =========================================================

def build_zcb_schedule(notional: float, freq: int, years: float) -> pd.DataFrame:
    """
    Zero-coupon schedule = bullet principal at maturity, no coupons.
    We reuse build_schedule_fixed with 0% rate.
    """
    df = build_schedule_fixed(notional, 0.0, freq, years, "bullet")
    df["Total CF"] = df["Coupon/Profit"] + df["Principal"]
    return df


def cir_generate_paths(r0: float, T: float, M: int, I: int,
                       kappa: float, theta: float, sigma: float,
                       exact: bool = True) -> np.ndarray:
    """
    CIR(85) short-rate paths, shape (M+1, I).
    Vectorized version aligned with user's ZCB_CIR.py (no deprecated np.float).
    """
    dt = T / M
    r = np.zeros((M + 1, I), dtype=np.float64)
    r[0, :] = r0

    if exact:
        # Exact sampling via non-central chi-square decomposition
        d = 4.0 * kappa * theta / (sigma ** 2)          # degrees parameter
        c = (sigma ** 2) * (1.0 - np.exp(-kappa * dt)) / (4.0 * kappa)
        chi = np.random.standard_normal((M + 1, I))     # for the (Z + sqrt(l))^2 term

        for t in range(1, M + 1):
            l = r[t - 1, :] * np.exp(-kappa * dt) / c   # non-centrality /2
            if d > 1.0:
                # (Z + sqrt(l))^2 + chi2(d-1)
                chi2 = np.random.chisquare(d - 1.0, size=I)
                r[t, :] = c * ((chi[t, :] + np.sqrt(l)) ** 2 + chi2)
            else:
                # Poisson mixture for small d
                N = np.random.poisson(l / 2.0, size=I)
                chi2 = np.random.chisquare(d + 2 * N, size=I)
                r[t, :] = c * chi2
    else:
        # Euler full truncation (low bias)
        z = np.random.standard_normal((M + 1, I))
        for t in range(1, M + 1):
            r_prev = np.maximum(r[t - 1, :], 0.0)
            drift = kappa * (theta - r_prev) * dt
            diff = sigma * np.sqrt(np.maximum(r_prev, 0.0)) * np.sqrt(dt) * z[t, :]
            r[t, :] = np.maximum(r_prev + drift + diff, 0.0)

    return r


def zcb_price_mc_cir(r_paths: np.ndarray, T: float, M: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Monte Carlo estimator for ZCB price under CIR:
    B(0,T) = E[ exp(-∫_0^T r_s ds) ]  with trapezoidal discretization.
    Returns (price, avg_rate_path, price_path_mean).
    """
    dt = T / M
    # trapezoid integral per path
    # int r_s ds ≈ sum_{t=1..M} 0.5*(r_t + r_{t-1})*dt
    integral = 0.5 * dt * (r_paths[0:-1, :] + r_paths[1:, :]).sum(axis=0)
    discounts = np.exp(-integral)                       # one value per path
    price = float(discounts.mean())

    # For charts
    avg_rate_path = r_paths.mean(axis=1)                # length M+1
    # rolling product of per-step discount factors to show a "mean path" of price
    per_step_disc = np.exp(-0.5 * dt * (r_paths[0:-1, :] + r_paths[1:, :]))
    price_path = np.vstack([np.ones((1, r_paths.shape[1])), per_step_disc]).cumprod(axis=0)
    price_path_mean = price_path.mean(axis=1)           # length M+1

    return price, avg_rate_path, price_path_mean


# =========================================================
# UI
# =========================================================

def render():
    st.subheader("Structuring Desk — Zero‑Coupon Bond Pricer")

    # ---- Inputs (cohérents avec le pricer global) ----
    default_notional, default_years, default_yield = 1_000_000.0, 5.0, 5.00
    f_map = {"Annual": 1, "Semi-annual": 2, "Quarterly": 4}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        notional = st.number_input("Notional", min_value=1_000.0, value=default_notional, step=1_000.0, format="%.2f")
    with c2:
        years = st.number_input("Maturity (years)", min_value=0.25, value=default_years, step=0.25, format="%.2f")
    with c3:
        freq_label = st.selectbox("Accrual frequency (for duration math)", sorted(f_map.keys()))
        freq = f_map[freq_label]
    with c4:
        accrued_frac = st.slider("Elapsed in current period (%)", 0, 100, 0,
                                 help="Accrued (approx.) as % of the current period — zero for pure ZCB, kept for UI symmetry.") / 100.0

    schedule = build_zcb_schedule(notional, freq, years)

    zero_coupon_ui = render

    # ---- Solve block ----
    st.markdown("#### Solve")
    s1, s2 = st.columns(2)
    solve_mode = st.radio("Solve for", ["Price (given Yield)", "Yield (given Clean Price)"], horizontal=True)
    if solve_mode == "Price (given Yield)":
        with s1:
            ytm = st.number_input("Yield to maturity (%)", min_value=-10.0, value=default_yield, step=0.25, format="%.4f") / 100.0
    else:
        with s1:
            clean_target = st.number_input("Target clean price (per 100)", min_value=0.0, value=90.0, step=0.5, format="%.4f")
    with s2:
        grid_bps = st.slider("Price–Yield grid width (bps)", 50, 500, 200, step=25)

    # ---- Deterministic pricing (discounting with YTM) ----
    if solve_mode == "Price (given Yield)":
        clean, accrued, dirty = price_from_yield(schedule, ytm, freq, notional, accrued_frac=accrued_frac)
        _, mac_dur, mod_dur, conv = macaulay_duration_convexity(schedule, ytm, freq)
    else:
        ytm = yield_from_price(schedule, clean_target, freq, notional)
        clean, accrued, dirty = price_from_yield(schedule, ytm, freq, notional, accrued_frac=accrued_frac)
        _, mac_dur, mod_dur, conv = macaulay_duration_convexity(schedule, ytm, freq)

    # ---- KPIs ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clean Price (per 100)", f"{clean:,.4f}")
    k2.metric("Accrued (per 100)", f"{accrued:,.4f}")
    k3.metric("Dirty Price (per 100)", f"{dirty:,.4f}")
    k4.metric("Yield to Maturity (%)", f"{ytm*100:,.4f}")

    k5, k6, k7 = st.columns(3)
    k5.metric("Macaulay Duration (yrs)", f"{mac_dur:,.4f}")
    k6.metric("Modified Duration (yrs)", f"{mod_dur:,.4f}")
    k7.metric("Convexity (yrs²)", f"{conv:,.4f}")

    # =========================================================
    # Charts (deterministic)
    # =========================================================
    st.markdown("### Results & Charts")

    left, right = st.columns(2)
    # Price–Yield curve around current YTM
    y0 = ytm
    bps = grid_bps / 10_000.0
    y_grid = np.linspace(max(-0.99, y0 - bps), y0 + bps, 41)
    prices = np.array([price_from_yield(schedule, y, freq, notional, 0.0)[0] for y in y_grid])
    with left:
        st.altair_chart(price_yield_chart(y_grid * 100, prices), use_container_width=True)

    # Cash‑flow breakdown (only principal at T)
    with right:
        st.altair_chart(cashflow_breakdown_chart(schedule), use_container_width=True)

    a1, a2 = st.columns(2)
    with a1:
        st.altair_chart(amortization_chart(schedule), use_container_width=True)

    # =========================================================
    # Monte Carlo (CIR) section
    # =========================================================
    st.markdown("### Monte Carlo (CIR) — Optional")
    with st.expander("Simulate short‑rate (CIR) and estimate B(0,T)"):
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            r0 = st.number_input("r0 (initial short rate)", min_value=0.0, value=0.01, step=0.005, format="%.4f")
        with mc2:
            kappa = st.number_input("κ (mean reversion)", min_value=0.001, value=0.10, step=0.01, format="%.4f")
        with mc3:
            theta = st.number_input("θ (long‑run mean)", min_value=0.0, value=0.03, step=0.005, format="%.4f")
        with mc4:
            sigma = st.number_input("σ (vol of short rate)", min_value=0.001, value=0.20, step=0.01, format="%.4f")

        mc5, mc6, mc7 = st.columns(3)
        with mc5:
            steps = st.number_input("Time steps (M)", min_value=10, value=50, step=5)
        with mc6:
            paths = st.number_input("Paths (I)", min_value=100, value=10_000, step=1_000)
        with mc7:
            exact = st.checkbox("Use exact sampling", value=True, help="If unticked, uses Euler full‑truncation.")

        if st.button("Run Monte Carlo (CIR)"):
            r_paths = cir_generate_paths(r0, years, int(steps), int(paths), kappa, theta, sigma, exact=exact)
            mc_price, avg_rate_path, price_path_mean = zcb_price_mc_cir(r_paths, years, int(steps))

            st.success(f"Monte Carlo ZCB estimate (per 1 of notional): {mc_price:,.6f}")
            # Two charts: short-rate path (mean) and price path (mean)
            df_rate = pd.DataFrame({"Step": np.arange(len(avg_rate_path)), "Short rate": avg_rate_path})
            df_price = pd.DataFrame({"Step": np.arange(len(price_path_mean)), "Price (per 1)": price_path_mean})

            ch_rate = (
                alt.Chart(df_rate)
                .mark_line()
                .encode(x="Step:Q", y=alt.Y("Short rate:Q", title="short rate"), tooltip=["Step","Short rate"])
                .properties(title="Short‑rate path (mean of simulations)", height=260)
            )
            ch_price = (
                alt.Chart(df_price)
                .mark_line()
                .encode(x="Step:Q", y=alt.Y("Price (per 1):Q", title="price"), tooltip=["Step","Price (per 1)"])
                .properties(title="Price path (mean of simulations)", height=260)
            )

            ccols = st.columns(2)
            with ccols[0]:
                st.altair_chart(ch_rate, use_container_width=True)
            with ccols[1]:
                st.altair_chart(ch_price, use_container_width=True)

            st.caption("Monte Carlo estimate discounts simulated short‑rate paths: "
                       "B(0,T) = E[exp(−∫ r_s ds)]. Mean paths are for visualization; pricing uses all paths.")

    # ---- Cash flow table & export ----
    st.markdown("### Cash flow table")
    table = schedule.copy()
    for c in ["Outstanding (begin)", "Coupon/Profit", "Principal", "Total CF", "Outstanding (end)"]:
        table[c] = table[c].map(lambda x: round(float(x), 6))
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "Download cash flow table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="zcb_cash_flows.csv",
        mime="text/csv",
    )

    # =========================================================
    # Theory / Learn more / How we calculate it / Example
    # =========================================================
    st.markdown("---")
    st.markdown("## Zero‑Coupon Bond — Definitions")

    st.markdown(
        """
**What is a Zero‑Coupon Bond?**  
- Pays **no periodic coupons**; a single **lump‑sum** redemption at maturity.  
- Issued at a **deep discount** to par; investor return = difference between **purchase price** and **face value**.  
- Duration equals **time to maturity** for a pure ZCB; highly sensitive to rates.
        """
    )

    # Learn more (modal/expander)
    if hasattr(st, "dialog"):
        @st.dialog("Learn more about Zero‑Coupon Bonds")
        def _more():
            _render_learn_more()
        if st.button("Learn More"):
            _more()
    else:
        with st.expander("Learn More"):
            _render_learn_more()

    # Example PDF
    st.markdown("### Example — Download")
    pdf_path = Path(__file__).resolve().parent.parent.parent / "Library" / "Zero Coupon Notes Example - HSBC (2023).pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download: Zero Coupon Notes — HSBC (2023) (PDF)",
                data=f.read(),
                file_name=pdf_path.name,
                mime="application/pdf",
            )
    else:
        st.info(f"Place the example PDF at **{pdf_path}** (filename must match exactly).")


def _render_learn_more():
    st.markdown(
        r"""
### How do we calculate it?

**Deterministic (Yield-based):**  
For a notional \(N\) with maturity \(T\) and annual yield \(y\) with frequency \(f\):
\[
\text{Clean Price per 100} = 100 \times \left(1+\frac{y}{f}\right)^{-fT}
\]
In our implementation, we build a one-line cash‑flow table (principal at \(T\)) and reuse the same
pricing/duration/convexity engine as other products.

**Monte Carlo (CIR short rate):**  
Model the short rate \(r_t\) under **CIR(85)**:
\[
dr_t = \kappa(\theta - r_t)\,dt + \sigma \sqrt{r_t}\,dW_t
\]
The ZCB price is the risk‑neutral expectation of the **discount factor**:
\[
B(0,T) = \mathbb{E}\!\left[\exp\!\left(-\int_0^T r_s\,ds\right)\right].
\]
We simulate many short‑rate paths using the **exact** scheme (or Euler full‑truncation), approximate the integral with
a trapezoid, compute \(\exp(-\int r\,ds)\) per path, then average across paths.

**When to use which?**  
- Yield‑based discounting is **fast** and consistent with desk conventions.  
- Monte Carlo (CIR) is useful to **illustrate term‑structure risk** and stress parameter changes; it’s educational and
aligns with the course/code you provided.

**Caveats:**  
- Real‑world pricing typically uses a **fitted discount curve** (OIS/treasury) with day‑count; we abstract those details.  
- CIR parameter choice impacts level/vol of rates; calibrate to market if using beyond education.
        """
    )

# =========================
# Notes / Sources (kept as comments)
# =========================
# Zenodo-FiZeroBond.pdf – class notes on ZCB pricing and discount factor intuition.
# ZCB_CIR.py – user-provided Monte Carlo implementation for CIR with exact/Euler schemes.
# “DERIVATIVES ANALYTICS WITH PYTHON” – reference for CIR simulation and ZCB expectation.
