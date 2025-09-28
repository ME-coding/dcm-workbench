# zero_coupon.py — Streamlit sub-page (Zero-Coupon Bonds)
# Everything wrapped in render(), no page_config here.

from __future__ import annotations

# ============================================================
# Imports
# ============================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple, Optional

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


# ============================================================
# Model & Utilities
# ============================================================
@dataclass
class ZCSpec:
    face: float = 100.0
    years_to_maturity: float = 5.0
    compounding: str = "Continuous"  # "Continuous" or "Annual"
    # Monte Carlo parameters:
    r0: float = 0.03      # initial short rate (annual, in decimals)
    a: float = 0.10       # mean reversion speed
    vol: float = 0.01     # short-rate volatility
    theta: float = 0.02   # long-run level (constant drift term)
    dt: float = 1 / 252   # time step in years (e.g., daily ~ 1/252)
    n_paths: int = 5_000  # number of Monte Carlo paths
    seed: Optional[int] = 32


# ---------- Spot curve helpers (Analytical pricing) ----------
def _flat_zero_rate_fn(r: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function r(t) that is flat at level r."""
    def r_of_t(t: np.ndarray) -> np.ndarray:
        return np.full_like(t, r, dtype=float)
    return r_of_t


def _interp_zero_rate_fn(tenors: np.ndarray, rates: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Piecewise-linear zero curve r(t) on [0, max(tenor)]. Extrapolate flat beyond bounds."""
    ten = np.asarray(tenors, dtype=float)
    rts = np.asarray(rates, dtype=float)

    ten[0] = max(1e-6, ten[0])  # avoid 0 exactly for interpolation stability
    order = np.argsort(ten)
    ten, rts = ten[order], rts[order]

    def r_of_t(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        r = np.interp(t, ten, rts, left=rts[0], right=rts[-1])
        return r
    return r_of_t


def _continuous_discount_from_curve(
    r_of_t: Callable[[np.ndarray], np.ndarray],
    T: float,
    n_steps: int = 1000
) -> float:
    """Compute exp(-∫_0^T r(u) du) via trapezoidal rule."""
    grid = np.linspace(0.0, T, n_steps + 1)
    r_vals = r_of_t(grid)
    integral = np.trapz(r_vals, grid)
    return float(np.exp(-integral))


def analytic_price_zero(
    spec: ZCSpec,
    r_of_t: Callable[[np.ndarray], np.ndarray],
    T: float,
    compounding: str = "Continuous"
) -> Tuple[float, float]:
    """
    Price a zero-coupon analytically from a spot curve r(t).
    Returns (price, yield_display) where yield_display uses the chosen compounding for display.
    """
    df_cont = _continuous_discount_from_curve(r_of_t, T)
    price_cont = spec.face * df_cont

    if compounding == "Annual":
        y = (spec.face / price_cont) ** (1.0 / T) - 1.0  # annual comp display
    else:
        y = -np.log(price_cont / spec.face) / T          # continuous comp display

    return float(price_cont), float(y)


# ---------- Durations/Convexity for ZC (pedagogical definitions) ----------
def macaulay_duration_zc(T: float) -> float:
    """For a pure zero-coupon, Macaulay duration = T (years)."""
    return float(T)


def modified_duration_zc(T: float, y: float, compounding: str) -> float:
    """Continuous: ModDur = T. Annual: ModDur = T / (1 + y)."""
    if compounding == "Annual":
        return float(T / max(1e-12, (1.0 + y)))
    return float(T)


def convexity_zc(T: float, y: float, compounding: str) -> float:
    """Continuous: T^2 ; Annual (discrete) ≈ T*(T+1)/(1+y)^2."""
    if compounding == "Annual":
        return float(T * (T + 1.0) / max(1e-12, (1.0 + y) ** 2))
    return float(T ** 2)


# ---------- Monte Carlo short-rate (Ornstein–Uhlenbeck / Hull–White style) ----------
def simulate_short_rate_paths(spec: ZCSpec, T: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate r_t via: dr_t = a*(theta - r_t) dt + vol dW_t  (Vasicek/Hull-White with constant theta).
    Returns (time_grid, paths) with paths.shape = (n_paths, n_steps+1).
    """
    if spec.seed is not None:
        np.random.seed(spec.seed)

    n_steps = int(np.ceil(T / spec.dt))
    dt = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1)

    r = np.empty((spec.n_paths, n_steps + 1), dtype=float)
    r[:, 0] = spec.r0

    # Euler–Maruyama
    drift_coef = spec.a
    vol = spec.vol
    th = spec.theta

    std_norm = np.random.normal
    for k in range(n_steps):
        z = std_norm(0.0, 1.0, size=spec.n_paths)
        r[:, k + 1] = r[:, k] + drift_coef * (th - r[:, k]) * dt + vol * np.sqrt(dt) * z

    return t_grid, r


def mc_price_zero(spec: ZCSpec, T: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Monte Carlo estimator of a zero-coupon price:
    average the discounted payoff across simulated paths.
    Returns (price, equivalent_cont_yield, t_grid, r_paths_sampled_for_plot).
    """
    t_grid, paths = simulate_short_rate_paths(spec, T)
    dt = t_grid[1] - t_grid[0]
    # ∫ r ds ≈ Σ 0.5*(r_k + r_{k+1})*Δt
    integral = 0.5 * dt * (paths[:, 0:-1] + paths[:, 1:]).sum(axis=1)
    disc = np.exp(-integral)
    price = spec.face * float(np.mean(disc))
    y_cont = -np.log(price / spec.face) / T  # equivalent continuous-comp yield

    # Keep a small sample of paths for plotting (e.g., 50)
    n_keep = min(50, spec.n_paths)
    idx = np.random.choice(spec.n_paths, size=n_keep, replace=False)
    return float(price), float(y_cont), t_grid, paths[idx, :]


# ============================================================
# UI Entry Point
# ============================================================
def render():
    st.markdown("## Zero-Coupon Bond")

    # --- Global CSS: justify narrative + center st.metric values
    st.markdown(
        """
        <style>
          .subtle-grey { color:#9aa0a6; }
          .stMarkdown p, .stMarkdown li, .stMarkdown div { text-align: justify; }
          [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
              text-align: center !important; justify-content: center !important;
              align-items: center !important; display: flex !important;
          }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtle-grey" style="margin-top:-6px;margin-bottom:10px;">'
        'A single cash flow at maturity. Two ways to value it: (i) discounting with a spot curve (analytical), '
        '(ii) simulating the short rate and averaging discounted payoffs (Monte Carlo).'
        '</div>',
        unsafe_allow_html=True
    )

    # --- 2×2 layout (top parameters / overview; bottom charts / metrics)
    colTL, colTR = st.columns([1.05, 1.0])
    colBL, colBR = st.columns([1.2, 0.8])

    # Unique widget key namespace for this sub-page
    W = "zc_"

    # Small helper: grey inline explanations next to titles
    def _label_with_help(title: str, help_text: str):
        st.markdown(
            f'<div style="margin-bottom:4px;"><span style="font-weight:600;">{title}</span> '
            f'<span class="subtle-grey">— {help_text}</span></div>',
            unsafe_allow_html=True
        )

    # ------------------------------
    # Top-Left: Parameters
    # ------------------------------
    with colTL:
        st.subheader("Parameters")

        method = st.radio(
            "Pricing method",
            ["Analytical (Spot Curve)", "Monte Carlo (Short-Rate)"],
            index=0,
            key=W + "method",
        )

        # Display yield as — concise helper
        _label_with_help(
            "Display yield as",
            "Continuous = compounding at every instant; Annual = once per year."
        )
        compounding = st.selectbox(
            "Display yield as",
            ["Continuous", "Annual"],
            index=0,
            key=W + "comp",
            label_visibility="collapsed"
        )

        face = st.number_input("Face value", value=100.0, step=1.0, format="%.2f", key=W + "face")
        years = float(st.slider("Maturity (years)", min_value=0, max_value=50, value=5, step=1, key=W + "mat"))

        if method == "Analytical (Spot Curve)":
            curve_mode = st.selectbox(
                "Spot curve mode", ["Flat rate", "Custom zero curve (table)"],
                index=0, key=W + "curve_mode"
            )
            if curve_mode == "Flat rate":
                flat_z = st.number_input(
                    "Flat zero rate (annual, %)", value=3.00, step=0.25, format="%.2f", key=W + "flat_z"
                ) / 100.0
                r_of_t = _flat_zero_rate_fn(flat_z)
                curve_df = None
            else:
                st.caption("Provide tenors (years) and corresponding zero rates (annual, %). Piecewise-linear interpolation.")
                default_curve = pd.DataFrame({
                    "TenorYears": [1, 2, 3, 5, 7, 10],
                    "ZeroRatePct": [2.0, 2.3, 2.6, 3.0, 3.1, 3.2],
                })
                curve_df = st.data_editor(
                    default_curve, use_container_width=True,
                    num_rows="dynamic", key=W + "curve_tbl"
                )
                if curve_df is None or curve_df.empty:
                    curve_df = default_curve
                r_of_t = _interp_zero_rate_fn(
                    curve_df["TenorYears"].to_numpy(dtype=float),
                    (curve_df["ZeroRatePct"].to_numpy(dtype=float) / 100.0),
                )
        else:
            # Monte Carlo short-rate parameters (no formula caption here)
            _label_with_help("Initial short rate r₀ (%, annual)", "Starting instantaneous risk-free rate at t = 0.")
            r0 = st.number_input(
                "Initial short rate r₀ (%, annual)", value=3.00, step=0.25, format="%.2f",
                key=W + "r0", label_visibility="collapsed"
            ) / 100.0

            _label_with_help("Mean reversion a", "Speed at which r(t) is pulled back toward the long-run level θ.")
            a = st.number_input(
                "Mean reversion a", value=0.10, step=0.05, format="%.2f",
                key=W + "a", label_visibility="collapsed"
            )

            _label_with_help("Volatility σ (%, annual)", "Size of random yearly rate moves; higher σ → wider rate dispersion.")
            vol = st.number_input(
                "Volatility σ (%, annual)", value=1.00, step=0.10, format="%.2f",
                key=W + "vol", label_visibility="collapsed"
            ) / 100.0

            _label_with_help("Long-run level θ (%, annual)", "Rate it tends to over time (mean-reversion).")
            theta = st.number_input(
                "Long-run level θ (%, annual)", value=2.00, step=0.25, format="%.2f",
                key=W + "theta", label_visibility="collapsed"
            ) / 100.0

            _label_with_help("Time step", "Simulation granularity; smaller steps approximate the integral more accurately.")
            dt = st.selectbox(
                "Time step", ["Daily (~1/252)", "Weekly (~1/52)", "Monthly (~1/12)"],
                index=0, key=W + "dt", label_visibility="collapsed"
            )
            dt_val = {"Daily (~1/252)": 1/252, "Weekly (~1/52)": 1/52, "Monthly (~1/12)": 1/12}[dt]

            _label_with_help("Number of paths", "Monte Carlo scenarios to average over; more paths reduce statistical noise.")
            n_paths = int(st.number_input(
                "Number of paths", value=5000, step=1000,
                key=W + "npaths", label_visibility="collapsed"
            ))

            _label_with_help("Random seed", "Locks randomness for repeatable results.")
            seed = int(st.number_input(
                "Random seed", value=42, step=1,
                key=W + "seed", label_visibility="collapsed"
            ))

        # Sticky spec
        if method == "Monte Carlo (Short-Rate)":
            spec = ZCSpec(
                face=face, years_to_maturity=years, compounding=compounding,
                r0=r0, a=a, vol=vol, theta=theta, dt=dt_val, n_paths=n_paths, seed=seed
            )
        else:
            spec = ZCSpec(face=face, years_to_maturity=years, compounding=compounding)

    # ------------------------------
    # Top-Right: Method overview
    # ------------------------------
    with colTR:
        st.subheader("Method overview")

        st.markdown(
            "- **Analytical (Spot Curve):** We treat the bond’s single payoff at maturity as being discounted by the market "
            "**term structure of zero rates**. If the zero curve is *flat* at level *z*, the price is "
            "`Face × exp(−z × T)` under continuous compounding. With a **custom curve**, we integrate the instantaneous "
            "spot rate over time (area under the curve) and exponentiate the negative of that integral. "
            "This is the cleanest way to value a pure discount bond in a deterministic-rate world."
        )
        st.markdown(
            "- **Monte Carlo (Short-Rate):** We simulate many future paths for the **instantaneous short rate** and average the "
            "discounted payoff. We use a simple **mean-reverting** process: the rate tends to drift back toward a long-run level θ "
            "at speed a, while random shocks with volatility σ push it around (often written as *dr = a(θ − r)dt + σ dW*)."
        )

        # Example PDF download
        st.markdown("### Example — Download")
        pdf_path = Path(__file__).resolve().parent.parent.parent / "Library" / "Zero Coupon Notes Example - HSBC (2023).pdf"
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download: Zero-Coupon Notes — HSBC (2023) (PDF)",
                    data=f.read(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=W + "dl_hsbc_2023",
                )
        else:
            st.info(f"Place the example PDF at **{pdf_path}** (filename must match exactly).")

    # ============================================================
    # Compute pricing
    # ============================================================
    if method == "Analytical (Spot Curve)":
        price, y_disp = analytic_price_zero(
            spec, r_of_t, spec.years_to_maturity, compounding=spec.compounding
        )
        # For charts below, create a one-bar CF at T and an “effective yield” line
        t = np.array([spec.years_to_maturity], dtype=float)
        cf = np.array([spec.face], dtype=float)
        t_line = np.linspace(0.5, max(0.5, spec.years_to_maturity), 40)
        eff_y = (np.exp(y_disp * t_line) - 1.0) if spec.compounding == "Continuous" else ((1.0 + y_disp) ** t_line - 1.0)
        y_label = "Yield (%)"
        mc_paths = None
        t_grid_paths = None
    else:
        # Monte Carlo price (yield displayed in chosen convention)
        price_mc, y_cont, t_grid_paths, mc_paths = mc_price_zero(spec, spec.years_to_maturity)
        price = price_mc
        y_disp = (np.exp(y_cont) - 1.0) if spec.compounding == "Annual" else y_cont
        # Chart data
        t = np.array([spec.years_to_maturity], dtype=float)
        cf = np.array([spec.face], dtype=float)
        t_line = np.linspace(0.5, max(0.5, spec.years_to_maturity), 40)
        eff_y = (np.exp(y_disp * t_line) - 1.0) if spec.compounding == "Continuous" else ((1.0 + y_disp) ** t_line - 1.0)
        y_label = "Yield (%)"

    # Risk metrics for a pure ZC
    macD = macaulay_duration_zc(spec.years_to_maturity)
    modD = modified_duration_zc(spec.years_to_maturity, y_disp, spec.compounding)
    convx = convexity_zc(spec.years_to_maturity, y_disp, spec.compounding)

    # ------------------------------
    # Bottom-Left: Chart (bar + line)
    # ------------------------------
    with colBL:
        st.subheader("Charts: Payout at Maturity & Effective Yield vs. Horizon")

        df_cf = pd.DataFrame({"Year": t, "CashFlow": cf})
        df_y = pd.DataFrame({"Year": t_line, "EffectiveYieldPct": eff_y * 100.0})

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
                  y=alt.Y("EffectiveYieldPct:Q", title=y_label),
                  tooltip=[alt.Tooltip("Year:Q"), alt.Tooltip("EffectiveYieldPct:Q", format=",.2f")]
              )
        )

        dual_axis = (
            alt.layer(bars, line)
               .resolve_scale(y="independent")
               .properties(height=380)
               .interactive(bind_y=False)
        )
        st.altair_chart(dual_axis, use_container_width=True)

        # Explanation block
        st.markdown(
            """
            <div class="subtle-grey" style="font-size:0.95rem; margin-top:.35rem;">
              <div><strong>What you’re seeing:</strong></div>
              <ul style="margin-top:.25rem; margin-bottom:.5rem;">
                <li><em>Bar</em>: the single redemption at maturity (no coupons).</li>
                <li><em>Line</em>: for each horizon <em>t</em>, the <u>effective annualized growth</u> implied by the single yield you selected for display:<br>
                    <code>Annual:</code> (1 + y)<sup>t</sup> − 1 &nbsp;&nbsp;|&nbsp;&nbsp;
                    <code>Continuous:</code> exp(y × t) − 1.</li>
              </ul>
              <div>
                <strong>How to read it:</strong> a zero’s value is all about discounting one payoff. Higher yield ⇒ steeper line and heavier discounting, hence a lower price; lower yield ⇒ flatter line and a higher present value.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ------------------------------
    # Bottom-Right: Key metrics
    # ------------------------------
    with colBR:
        st.subheader("Key metrics")

        price_to_par = 100.0 * (price / spec.face)

        left, right = st.columns(2)
        with left:
            st.metric("Price", f"{price:,.2f}")
            st.metric("Macaulay duration (yrs)", f"{macD:,.2f}")
            st.metric("Convexity (yrs²)", f"{convx:,.2f}")
        with right:
            if spec.compounding == "Annual":
                st.metric("Zero yield (annual, %)", f"{y_disp * 100:,.2f}%")
            else:
                st.metric("Zero yield (continuous, %)", f"{y_disp * 100:,.2f}%")
            st.metric("Modified duration (yrs)", f"{modD:,.2f}")
            st.metric("Price-to-Par (%)", f"{price_to_par:,.2f}%")

        # Collapsible "Learn More"
        with st.expander("Learn More"):
            st.markdown(
                """
                <div class="subtle-grey" style="font-size:0.95rem;">
                <div style="font-weight:600; margin-bottom:.15rem;">Price</div>
                <div>The clean price is the present value of coupons and principal discounted at the chosen rate (YTM or discount rate). It excludes accrued interest.</div>

                <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Zero yield</div>
                <div>The single annualized rate that discounts the redemption amount back to today. It is the term rate applicable to the maturity T.</div>

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

    # ------------------------------
    # Extra: Monte Carlo paths chart
    # ------------------------------
    if method == "Monte Carlo (Short-Rate)" and mc_paths is not None and t_grid_paths is not None:
        st.markdown("---")
        st.subheader("Monte Carlo: Sampled short-rate paths")

        # Long DataFrame for Altair
        df_paths = pd.DataFrame(mc_paths)
        df_paths["path_id"] = np.arange(df_paths.shape[0])
        df_long = df_paths.melt(id_vars=["path_id"], var_name="k", value_name="rate")

        # Map k → time using t_grid_paths
        df_long["Year"] = df_long["k"].map(dict(enumerate(t_grid_paths)))
        df_long["RatePct"] = df_long["rate"] * 100.0

        chart = (
            alt.Chart(df_long)
              .mark_line(opacity=0.85)
              .encode(
                  x=alt.X("Year:Q", title="Year"),
                  y=alt.Y("RatePct:Q", title="Short rate (%)"),
                  color=alt.Color("path_id:N", legend=None),
                  tooltip=[alt.Tooltip("Year:Q"), alt.Tooltip("RatePct:Q", format=",.2f")]
              )
              .properties(height=500)  # taller, fixed (non-interactive)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            """
            <div class="subtle-grey" style="font-size:0.95rem; margin-top:.35rem;">
              <strong>Interpretation:</strong> Each line is a simulated path of the instantaneous short rate. For every scenario, we discount the payoff using the whole path of rates to get a present value; the Monte Carlo price is the <em>average</em> of those scenario values.
              <br><br>
              <strong>Monte Carlo:</strong> A numerical method that approximates expected values by repeated random sampling. More paths reduce statistical noise; smaller time steps make the path integral more accurate. Use it to see how volatility, mean reversion and horizon change the price distribution.
            </div>
            """,
            unsafe_allow_html=True
        )
