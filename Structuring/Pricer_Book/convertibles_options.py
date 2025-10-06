# convertibles_options.py — Streamlit sub-page (Convertible & Options)
# Pricing a convertible (conversion + optional issuer call) via a binomial tree.
# Chart removed for now (placeholder).

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import math

import numpy as np
import streamlit as st


# =========================
#    Model & utilities
# =========================

# Fixed number of steps for the binomial tree (precision without being too heavy)
_STEPS_FIXED = 240  # ~monthly over 20 years; adjust if needed

@dataclass
class ConvSpec:
    # Bond
    face: float = 100.0
    coupon_rate: float = 0.02     # annual coupon in decimal
    freq: int = 2                 # coupons per year (1,2,4,12)
    years_to_maturity: float = 5.0

    # Equity & market
    s0: float = 50.0              # stock spot price
    div_yield: float = 0.00       # continuous dividend yield q
    vol: float = 0.30             # annual volatility
    r: float = 0.02               # continuous risk-free rate
    credit_spread: float = 0.01   # add-on for bond-leg discounting

    # Conversion
    conv_ratio: float = 1.5       # shares per bond

    # Embedded option (CALL only, optional)
    callable: bool = False
    call_price: float = 105.0
    call_start_year: float = 3.0  # earliest call date (in years)

    # Tree discretization (fixed, not exposed to UI)
    steps: int = _STEPS_FIXED


def _coupon_per_period(spec: ConvSpec) -> float:
    return spec.face * spec.coupon_rate / spec.freq


def _times_and_periods(spec: ConvSpec) -> Tuple[np.ndarray, int]:
    """Return coupon times in years and total coupon periods."""
    periods = int(round(spec.years_to_maturity * spec.freq))
    t_pay = np.arange(1, periods + 1) / spec.freq
    return t_pay, periods


def _straight_bond_price_floor(spec: ConvSpec) -> float:
    """Plain bond floor: discount coupons + principal at (r + spread) with discrete compounding at 'freq'."""
    t_pay, _ = _times_and_periods(spec)
    cpn = _coupon_per_period(spec)
    per = (spec.r + spec.credit_spread) / spec.freq
    disc = (1.0 + per) ** (spec.freq * t_pay)
    cfs = np.full_like(t_pay, cpn, dtype=float)
    cfs[-1] += spec.face
    return float(np.sum(cfs / disc))


def _binomial_params(spec: ConvSpec):
    """CRR tree parameters under risk-neutral measure with continuous dividend yield q."""
    T = spec.years_to_maturity
    N = int(spec.steps)
    dt = T / N
    u = math.exp(spec.vol * math.sqrt(dt))
    d = 1.0 / u
    a = math.exp((spec.r - spec.div_yield) * dt)  # growth factor under Q
    p = (a - d) / (u - d)
    p = min(max(p, 0.0), 1.0)  # numerical guard
    return N, dt, u, d, p


def _call_is_active(spec: ConvSpec, t_years: float) -> bool:
    return spec.callable and (t_years >= spec.call_start_year)


def _binomial_convertible_pricer(spec: ConvSpec) -> float:
    """
    Convertible bond pricing via backward induction on a CRR equity tree.

    Per-node logic:
      • Base investor decision: max(conversion value, continuation value)
      • If call is active: issuer may settle at min( node value , max(conversion, call price) )
        (issuer calls when cheaper than letting it continue; investor converts if conversion > call price)
      • Continuation discounted at (r + credit_spread) to reflect bond-leg credit risk
    """
    N, dt, u, d, p = _binomial_params(spec)

    # Stock prices at maturity
    S_T = np.array([spec.s0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)], dtype=float)

    # Coupon schedule mapped onto tree steps (coupon paid at end of period n)
    coupon_per = _coupon_per_period(spec)
    pay_coupon = np.zeros(N + 1, dtype=float)
    coupon_indices = set()
    for k in range(1, int(spec.years_to_maturity * spec.freq) + 1):
        t_k = k / spec.freq
        idx = int(round(t_k / dt)) - 1  # coupon paid at step idx (0-index)
        idx = min(max(idx, 0), N - 1)
        coupon_indices.add(idx)
    for idx in coupon_indices:
        pay_coupon[idx] = coupon_per

    # Terminal payoff (max of redemption vs immediate conversion)
    conv_val_T = spec.conv_ratio * S_T
    V = np.maximum(spec.face, conv_val_T)

    # Backward induction
    disc = math.exp(-(spec.r + spec.credit_spread) * dt)
    for n in range(N - 1, -1, -1):
        t_n1 = (n + 1) * dt
        j_idx = np.arange(0, n + 1)
        S_n = spec.s0 * (u ** j_idx) * (d ** (n - j_idx))

        # Continuation: discounted RN expectation + coupon if due at end of step n
        V_cont_next = p * V[1:n + 2] + (1.0 - p) * V[0:n + 1]
        V_cont = disc * V_cont_next + pay_coupon[n]

        # Immediate conversion
        V_conv = spec.conv_ratio * S_n

        # Base node value (investor choice: convert vs continue)
        V_node = np.maximum(V_conv, V_cont)

        # Issuer call (if active): issuer can settle at better of (call price, immediate conversion)
        if _call_is_active(spec, t_n1):
            call_settlement = np.maximum(V_conv, spec.call_price)
            V_node = np.minimum(V_node, call_settlement)

        V = V_node

    return float(V[0])


def _greek_delta_gamma(spec: ConvSpec) -> Tuple[float, float]:
    """Delta & Gamma via finite differences on S0."""
    s0 = spec.s0
    bump = max(0.01 * s0, 1e-3)
    spec_up = ConvSpec(**{**spec.__dict__, "s0": s0 + bump})
    spec_dn = ConvSpec(**{**spec.__dict__, "s0": s0 - bump if s0 > bump else 1e-6})
    p0 = _binomial_convertible_pricer(spec)
    p_up = _binomial_convertible_pricer(spec_up)
    p_dn = _binomial_convertible_pricer(spec_dn)
    delta = (p_up - p_dn) / (2.0 * bump)
    gamma = (p_up - 2.0 * p0 + p_dn) / (bump ** 2)
    return float(delta), float(gamma)


def _parity_and_conv_value(spec: ConvSpec) -> Tuple[float, float]:
    """Parity (a.k.a. conversion value today) = S0 × ratio."""
    conv_now = spec.conv_ratio * spec.s0
    return conv_now, conv_now


# =========================
#           UI
# =========================

def render():
    st.markdown("## Convertible & Options")

    # Global CSS: justify paragraphs + center st.metric
    st.markdown(
        """
        <style>
          .subtle-grey { color:#9aa0a6; }
          .stMarkdown p, .stMarkdown li, .stMarkdown div { text-align: justify; }
          [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
              text-align: center !important;
              justify-content: center !important;
              align-items: center !important;
              display: flex !important;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 2×2 matrix
    colTL, colTR = st.columns([1.05, 1.0])
    colBL, colBR = st.columns([1.2, 0.8])

    # -------------------
    # Top-Left: Parameters
    # -------------------
    with colTL:
        st.subheader("Parameters")

        # Compact layout: two columns (maturity stays a slider; others are inputs)
        p_left, p_right = st.columns(2)

        with p_left:
            face = st.number_input("Face value", value=100.0, step=1.0, format="%.2f", key="cv_face")
            coupon_rate = st.number_input("Annual coupon (%)", value=2.00, step=0.25, format="%.2f", key="cv_cpn") / 100.0

        with p_right:
            freq_label_to_val = {
                "Annual (1x)": 1,
                "Semiannual (2x)": 2,
                "Quarterly (4x)": 4,
                "Monthly (12x)": 12,
            }
            freq_label = st.selectbox("Coupon frequency", list(freq_label_to_val.keys()), index=1, key="cv_freq_lbl")
            freq = int(freq_label_to_val[freq_label])
            years = float(st.slider("Maturity (years)", min_value=1, max_value=30, value=5, step=1, key="cv_mat"))

        st.caption("Bond leg: coupons are paid at the chosen frequency; the face value is redeemed at maturity.")

        # ---- Equity & market (compact: two columns) ----
        st.markdown("##### Equity & market")
        em_left, em_right = st.columns(2)
        with em_left:
            s0 = st.number_input("Underlying stock price (S₀)", value=47.50, step=0.5, format="%.2f", key="cv_s0")
            div_y = st.number_input("Dividend yield (q, %)", value=0.00, step=0.25, format="%.2f", key="cv_div") / 100.0
            vol = st.number_input("Volatility (%, annual)", value=30.0, step=1.0, format="%.0f", key="cv_vol") / 100.0
        with em_right:
            r = st.number_input("Risk-free rate (%, cont.)", value=2.00, step=0.25, format="%.2f", key="cv_rf") / 100.0
            spr = st.number_input("Credit spread (%, add-on)", value=1.00, step=0.25, format="%.2f", key="cv_spr") / 100.0
            conv_ratio = st.number_input("Conversion ratio (shares per bond)", value=1.50, step=0.10, format="%.2f", key="cv_ratio")

        st.caption("Equity & market: stock dynamics drive the conversion option; coupons/continuation are discounted at (r + credit spread).")

        # ---- Embedded option (CALL only, optional) ----
        st.markdown("##### Embedded options")
        call_enabled = st.checkbox("Add issuer call option", value=False, key="cv_call_enable")
        if call_enabled:
            eo1, eo2 = st.columns(2)
            with eo1:
                call_price = st.number_input("Call price", value=105.0, step=0.5, format="%.2f", key="cv_call_px")
            with eo2:
                call_start = st.number_input("Callable from (year)", value=3.0, step=0.5, format="%.1f", key="cv_call_from")
        else:
            # Provide defaults (won't be used unless call_enabled True)
            call_price = 105.0
            call_start = 3.0

    # -------------------
    # Top-Right: Method overview + Examples — Download
    # -------------------
    with colTR:
        st.subheader("Method overview")

        # -- Explanation for non-experts
        st.markdown(
            """
            **What is a convertible bond?**  
            A convertible bond is a regular bond (coupons + redemption at par) that also gives the investor a **choice**:
            keep receiving coupons like a bond, **or** convert the bond into a fixed number of shares (the *conversion ratio*).
            Some issues also allow the **issuer** to redeem early (a *call*), which can cap the upside if the stock rallies.
            Because these choices can happen at different dates, we price the instrument by simulating many possible stock paths
            and solving the investor’s/issuer’s decisions **backward in time**.
            """
        )

        st.markdown(
            """
            **How we price it — step by step**  
            1) **Build an equity tree (CRR):** at each small time step, the stock can go *up* or *down*.
               In our model, the stock grows at the risk-free rate *r* minus the dividend yield *q* (dividends leave the stock price).  
            
            2) **We value the bond part of the convertible**: the expected future value is discounted at (risk-free + credit spread), and coupons are added on coupon dates.  
            
            3) **Conversion & call**: at each step, the investor chooses between holding or converting (value = shares × stock), while the issuer—if allowed—may call the bond (redeem at the call price, which in practice can force investors to convert if the stock is high), limiting the investor’s upside.
            
            4) **Roll back to today:** repeating this comparison from maturity to now yields the fair price and the risk metrics.
            """
        )

        with st.expander("What about the call option?"):
            st.markdown(
                """
                **What is the call in a CB?**  
                The *issuer call* is the issuer’s right to **redeem early** at a preset call price (sometimes only on specific dates
                or once the stock trades above a trigger — “soft call”). In practice, if the stock rallies and the convertible becomes
                very equity-like, the issuer may call to stop paying coupons or to push investors to convert.

                **How is it reflected in pricing?**  
                On call-eligible dates, after computing the investor’s node value (*max of continue vs convert*), the issuer may override
                that value by choosing the **cheapest** settlement for itself: either pay the **call price** in cash or **let investors convert**.
                Mathematically we replace the node value by  
                \\(\\min\\big(\\text{node value},\\; \\max(\\text{conversion},\\; \\text{call price})\\big)\\).  
                This *caps the upside* of the convertible when the stock is high, thus **reducing its value** compared with an identical
                non-callable structure. The effect is small when the bond is **bond-like**, and much larger when it’s **equity-like**.
                """
            )

        st.markdown("### Examples — Download")

        lib_dir = Path(__file__).resolve().parent.parent.parent / "Library"

        pdf1 = lib_dir / "Convertible Bond Example - Hoffmann-Green OCEANE (2024, by Portzamparc BNPP).pdf"
        if pdf1.exists():
            with open(pdf1, "rb") as f:
                st.download_button(
                    "Download: Hoffmann-Green OCEANE (2024) — Portzamparc BNPP (PDF)",
                    data=f.read(),
                    file_name=pdf1.name,
                    mime="application/pdf",
                    key="cv_dl_hgo"
                )
        else:
            st.info(f"Place the example PDF at **{pdf1}** (filename must match exactly).")

        pdf2 = lib_dir / "Convertible Bond Example - Vinci (2025).pdf"
        if pdf2.exists():
            with open(pdf2, "rb") as f:
                st.download_button(
                    "Download: Convertible Bond — Vinci (2025) (PDF)",
                    data=f.read(),
                    file_name=pdf2.name,
                    mime="application/pdf",
                    key="cv_dl_vinci"
                )
        else:
            st.info(f"Place the example PDF at **{pdf2}** (filename must match exactly).")

    # -------------------
    # Computation
    # -------------------
    spec = ConvSpec(
        face=face, coupon_rate=coupon_rate, freq=int(freq), years_to_maturity=years,
        s0=s0, div_yield=div_y, vol=vol, r=r, credit_spread=spr,
        conv_ratio=conv_ratio,
        callable=bool(call_enabled), call_price=call_price, call_start_year=call_start,
        steps=_STEPS_FIXED
    )

    price = _binomial_convertible_pricer(spec)
    bond_floor = _straight_bond_price_floor(spec)
    conv_val_now, parity = _parity_and_conv_value(spec)
    delta, gamma = _greek_delta_gamma(spec)

    # Conversion price (strike) derived from face and conversion ratio
    conversion_price = (spec.face / spec.conv_ratio) if spec.conv_ratio > 0 else float("nan")

    # -------------------
    # Bottom-Left: Chart — placeholder only
    # -------------------
    with colBL:
        st.subheader("Price vs Stock")
        st.info("To be updated.")

    # -------------------
    # Bottom-Right: Key metrics (rounded to 2 decimals)
    # -------------------
    with colBR:
        st.subheader("Key metrics")

        price_to_par = 100.0 * (price / spec.face)
        conv_premium_pct = 100.0 * (price / parity - 1.0) if parity > 0 else np.nan

        def f2(x: float) -> str:
            try:
                return f"{x:.2f}"
            except Exception:
                return "—"

        left, right = st.columns(2)
        with left:
            st.metric("Convertible price", f2(price))
            st.metric("Bond floor (no conversion)", f2(bond_floor))
            st.metric("Conversion price (strike)", f2(conversion_price))
        with right:
            st.metric("Delta (∂Price/∂S)", f2(delta))
            st.metric("Gamma (∂²Price/∂S²)", f2(gamma))
            st.metric("Price-to-Par (%)", f2(price_to_par))

        with st.expander("Learn More"):
            st.markdown(
                f"""
                <div class="subtle-grey" style="font-size:0.95rem;">
                  <div style="font-weight:600; margin-bottom:.15rem;">Convertible price</div>
                  Tree-based fair value reflecting coupons, conversion, and (if enabled) issuer call.

                  <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Bond floor</div>
                  Present value of coupons and principal discounted at <code>r + credit spread</code>.

                  <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Parity</div>
                  Immediate conversion value = <code>S₀ × ratio</code>.

                  <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Conversion price (strike)</div>
                  In a CB, the strike is the <em>conversion price</em> = <code>Face / Conversion ratio</code>.
                  It sets the equity level at which conversion starts to be attractive:
                  when the stock price <em>S</em> is <strong>above</strong> the conversion price, the immediate conversion value
                  (<code>ratio × S</code>) exceeds the bond’s par value (equity-like region); when <em>S</em> is <strong>below</strong>,
                  the bond behaves more like a straight bond (bond-like region). In practice, coupons, credit, and call features
                  can shift the exact conversion threshold, but the strike remains the key anchor for “moneyness”.

                  <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Delta</div>
                  First-order sensitivity to the stock price (holding other inputs constant).

                  <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Gamma</div>
                  Rate of change of Delta as the stock moves (second-order equity convexity).
                </div>
                """,
                unsafe_allow_html=True
            )
