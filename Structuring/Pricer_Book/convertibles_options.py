# convertibles_options.py — Streamlit sub-page (Convertible & Options)
# Pricing a convertible (conversion + optional issuer call) via a binomial tree.
# Chart mirrors classic CB payoff: CB price (blue), bond floor (yellow),
# parity/conversion value (green from origin), with regime bands.
# All Streamlit widget keys use a unique 'cb_' prefix to avoid collisions.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import math

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# =========================
#    Model & utilities
# =========================

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
    conv_ratio: float = 3         # shares per bond (default here; UI overrides)

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

    # Unique key helper for this page
    KPREFIX = "cb_"
    k = lambda name: f"{KPREFIX}{name}"

    # Global CSS
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

    # Inputs (two columns)
    colTL, colTR = st.columns([1.05, 1.0])

    # -------------------
    # Top-Left: Parameters
    # -------------------
    with colTL:
        st.subheader("Parameters")

        p_left, p_right = st.columns(2)

        with p_left:
            face = st.number_input("Face value", value=100.0, step=1.0, format="%.2f", key=k("face"))
            coupon_rate = st.number_input("Annual coupon (%)", value=2.00, step=0.25,
                                          format="%.2f", key=k("cpn")) / 100.0

        with p_right:
            freq_label_to_val = {
                "Annual (1x)": 1,
                "Semiannual (2x)": 2,
                "Quarterly (4x)": 4,
                "Monthly (12x)": 12,
            }
            freq_label = st.selectbox("Coupon frequency", list(freq_label_to_val.keys()),
                                      index=1, key=k("freq_lbl"))
            freq = int(freq_label_to_val[freq_label])
            years = float(st.slider("Maturity (years)", min_value=1, max_value=30,
                                    value=5, step=1, key=k("mat")))

        # ---- Equity & market ----
        st.markdown("##### Equity & market")
        em_left, em_right = st.columns(2)
        with em_left:
            s0 = st.number_input("Underlying stock price (S₀)", value=47.50, step=0.5,
                                 format="%.2f", key=k("s0"))
            div_y = st.number_input("Dividend yield (q, %)", value=0.00, step=0.25,
                                    format="%.2f", key=k("div")) / 100.0
            vol = st.number_input("Volatility (%, annual)", value=30.0, step=1.0,
                                  format="%.0f", key=k("vol")) / 100.0
        with em_right:
            r = st.number_input("Risk-free rate (%, cont.)", value=2.00, step=0.25,
                                format="%.2f", key=k("rf")) / 100.0
            spr = st.number_input("Credit spread (%, add-on)", value=1.00, step=0.25,
                                  format="%.2f", key=k("spr")) / 100.0
            conv_ratio = st.number_input("Conversion ratio (shares per bond)", value=1.50,
                                         step=0.10, format="%.2f", key=k("ratio"))

        # ---- Embedded option (CALL only, optional) ----
        st.markdown("##### Embedded options")
        call_enabled = st.checkbox("Add issuer call option", value=False, key=k("call_enable"))
        if call_enabled:
            eo1, eo2 = st.columns(2)
            with eo1:
                call_price = st.number_input("Call price", value=105.0, step=0.5,
                                             format="%.2f", key=k("call_px"))
            with eo2:
                call_start = st.number_input("Callable from (year)", value=3.0, step=0.5,
                                             format="%.1f", key=k("call_from"))
        else:
            call_price = 105.0
            call_start = 3.0

    # -------------------
    # Top-Right: Method overview + examples
    # -------------------
    with colTR:
        st.subheader("Method overview")
        st.markdown(
            """
            **What is a convertible bond?**  
            A convertible is a bond (coupons + redemption at par) with an embedded option to
            convert into a fixed number of shares (the *conversion ratio*). Some issues are callable
            by the issuer, which can cap upside when equity rallies. We price it on a CRR tree with
            backward induction of investor/issuer decisions.
            """
        )

        st.markdown(
            """
            **How we price it — step by step**  
            1) Build an equity tree under the risk-neutral measure.  
            2) Value the bond leg (discount at r + credit spread) and add coupons on dates.  
            3) At each node, investor chooses max(continue, convert); issuer (if callable) may
               settle at min(node value, max(call price, conversion value)).  
            4) Roll back to today to get price and Greeks.
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
                    key=k("dl_hgo")
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
                    key=k("dl_vinci")
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

    # Conversion price (strike)
    conversion_price = (spec.face / spec.conv_ratio) if spec.conv_ratio > 0 else float("nan")

    # -------------------
    # FULL-WIDTH: Key metrics BEFORE the chart
    # -------------------
    st.subheader("Key metrics")

    def f2(x: float) -> str:
        try:
            return f"{x:.2f}"
        except Exception:
            return "—"

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: st.metric("Convertible price", f2(price))
    with m2: st.metric("Bond floor (no conversion)", f2(bond_floor))
    with m3: st.metric("Conversion price (strike)", f2(conversion_price))
    with m4: st.metric("Parity now (ratio×S₀)", f2(parity))
    with m5: st.metric("Delta (∂Price/∂S)", f2(delta))
    with m6: st.metric("Gamma (∂²Price/∂S²)", f2(gamma))

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

    # -------------------
    # Chart — CB payoff with regime bands (SMALL + right-hand explanations)
    # -------------------
    st.subheader("Convertible Bond Price Evolution")

    # IMPORTANT: use same ratio as (colTL, colTR) so the right text aligns with "Method overview"
    colCL, colCR = st.columns([1.05, 1.0])

    with colCL:
        fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=120)
        plt.rcParams.update({"font.size": 8})

        # Grid of stock prices
        s_min = 0.0
        s_max = max(1.6 * spec.s0, 1.6 * conversion_price if math.isfinite(conversion_price) else 2.0 * spec.s0)
        s_grid = np.linspace(max(0.0, s_min), s_max, 45)

        # Curves
        cb_prices = np.empty_like(s_grid)
        for i, s in enumerate(s_grid):
            sp = ConvSpec(**{**spec.__dict__, "s0": float(s)})
            cb_prices[i] = _binomial_convertible_pricer(sp)

        floor_curve = np.full_like(s_grid, bond_floor, dtype=float)
        parity_curve = spec.conv_ratio * s_grid  # from origin

        # Regime bands
        if math.isfinite(conversion_price) and conversion_price > 0:
            b1, b2, b3 = 0.30*conversion_price, 0.80*conversion_price, 1.20*conversion_price
        else:
            b1, b2, b3 = 0.30*spec.s0, 0.80*spec.s0, 1.20*spec.s0

        ax.axvspan(0, b1, alpha=0.06); ax.axvspan(b1, b2, alpha=0.08)
        ax.axvspan(b2, b3, alpha=0.10); ax.axvspan(b3, s_max, alpha=0.06)
        for x in [b1, b2, b3]:
            ax.axvline(x, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)

        # Lines
        ax.plot(s_grid, cb_prices, label="Convertible price", linewidth=1.6, color="tab:blue")
        ax.plot(s_grid, floor_curve, label="Bond floor", linewidth=1.3, linestyle="--", color="gold")
        ax.plot(s_grid, parity_curve, label="Parity (conversion value)", linewidth=1.3, linestyle=":", color="green")

        ax.set_xlabel("Stock price", fontsize=9)
        ax.set_ylabel("Convertible bond price", fontsize=9)
        ax.grid(True, alpha=0.22)

        # Zone labels
        y_top = ax.get_ylim()[1] * 0.90
        ax.text(b1/2 if b1>0 else 0.02*s_max, y_top, "Junk", ha="center", va="top", fontsize=8)
        ax.text((b1+b2)/2, y_top, "Bond-like", ha="center", va="top", fontsize=8)
        ax.text((b2+b3)/2, y_top, "Balanced", ha="center", va="top", fontsize=8)
        ax.text((b3+s_max)/2, y_top, "Equity-like", ha="center", va="top", fontsize=8)

        # Conversion price: vertical line + HORIZONTAL label to the right
        if math.isfinite(conversion_price):
            ax.axvline(conversion_price, linestyle=":", linewidth=0.9, color="gray")
            mid_y = ax.get_ylim()[0] + 0.55 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(conversion_price + 0.02 * s_max, mid_y,
                    "Conversion price", rotation=0, va="top", ha="left",
                    color="gray", fontsize=8)

        # Conversion premium: vertical double arrow from bond floor up to CB curve at x_anno
        x_anno = b2 + 0.10 * (b3 - b2)  # inside the Balanced region
        if 0 < x_anno <= s_max:
            y_to = float(np.interp(x_anno, s_grid, cb_prices))   # CB price at x_anno
            y_from = float(bond_floor)                            # Bond floor (flat)
            if y_to > y_from:
                ax.annotate(
                    "", xy=(x_anno, y_to), xytext=(x_anno, y_from),
                    arrowprops=dict(arrowstyle="<->", lw=1.0, color="black")
                )
                ax.text(x_anno + 0.02 * s_max, (y_to + y_from) / 2.0,
                        "Conversion premium", fontsize=8, ha="left", va="center", color="black")

        # Legend under the plot
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=3,
            fontsize=8,
            frameon=False
        )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

 # RIGHT column (aligned with "Method overview")
    with colCR:
        st.markdown(
            """
            <div class="subtle-grey">
            <strong>What the chart shows</strong><br/>
            The figure plots the convertible bond price against the underlying stock price. The blue curve is the model value
            from the binomial tree; the yellow dashed line is the straight-bond floor (present value of coupons and principal
            discounted at r + credit spread); the green dotted line is the parity and the shaded bands illustrate typical regimes (from <em>junk/busted</em> to <em>equity-like</em>).
            <br/><br/>

            - For low stock levels, the CB behaves like a bond close to its floor.  
            - For high stock levels, it tends toward equity with a slope approximately equal to the conversion ratio.  
            - Between the two, optionality creates convexity.
             <br/>
            The convexity of a convertible bond comes from the asymmetry between downside risk and upside potential:
            - The bond floor limits losses when the stock falls, while the conversion option allows unlimited gains when it rises.
            - The curve bends upward: the slope flattens on the downside and steepens on the upside (protection and acceleration).
             <br/>

            <strong>Conversion premium</strong><br/>
            The difference between the convertible bond price and the parity at a given stock price — the excess paid for
            optionality, carry and credit over the pure equity value. On the chart it is shown by the vertical double arrow
            from the bond floor up to the blue curve in the balanced area.
            <br/>

            <strong>Conversion price (strike)</strong><br/>
            Face / Conversion ratio. It is the stock level where parity equals par (nominal). Above this level, conversion starts
            to be economically attractive, ignoring coupons, credit and call features. On the chart it is the vertical dotted line
            with a horizontal label.
            </div>
            """,
            unsafe_allow_html=True
        )
