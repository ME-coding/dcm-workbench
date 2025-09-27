# convertibles_options.py — Streamlit sub-page (Convertible & Options)
# Pricing a convertible (conversion + investor put + issuer call) via a binomial tree.
# UI layout mirrors your Vanilla Bond page.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import math

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


# =========================
#    Model & utilities
# =========================

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

    # Embedded options (ALWAYS ON)
    callable: bool = True
    call_price: float = 105.0
    call_start_year: float = 3.0  # earliest call date (in years)

    putable: bool = True
    put_price: float = 95.0
    put_start_year: float = 2.0   # earliest put date (in years)

    # Tree discretization
    steps: int = 200              # binomial steps


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
    a = math.exp((spec.r - spec.div_yield) * dt)  # growth factor under q
    p = (a - d) / (u - d)
    p = min(max(p, 0.0), 1.0)  # numerical guard
    return N, dt, u, d, p


def _call_is_active(spec: ConvSpec, t_years: float) -> bool:
    return spec.callable and (t_years >= spec.call_start_year)


def _put_is_active(spec: ConvSpec, t_years: float) -> bool:
    return spec.putable and (t_years >= spec.put_start_year)


def _binomial_convertible_pricer(spec: ConvSpec) -> float:
    """
    Convertible bond pricing via backward induction on a CRR tree.

    Per-node logic:
      • Base investor decision: max(conversion value, continuation value)
      • If put is active: max(put price, conversion, continuation)
      • If call is active: min(continuation, max(conversion, call price))
        (issuer calls when cheaper than letting it continue; investor would convert if conversion > call price)
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

    # Terminal payoff
    conv_val_T = spec.conv_ratio * S_T
    V = np.maximum(spec.face, conv_val_T)

    # Backward induction
    disc = math.exp(-(spec.r + spec.credit_spread) * dt)
    for n in range(N - 1, -1, -1):
        t_n1 = (n + 1) * dt
        j_idx = np.arange(0, n + 1)
        S_n = spec.s0 * (u ** j_idx) * (d ** (n - j_idx))

        # Continuation: discounted risk-neutral expectation + coupon if due at end of step n
        V_cont_next = p * V[1:n + 2] + (1.0 - p) * V[0:n + 1]
        V_cont = disc * V_cont_next + pay_coupon[n]

        # Immediate conversion
        V_conv = spec.conv_ratio * S_n

        # Base node value
        V_node = np.maximum(V_conv, V_cont)

        # Investor put (if active)
        if _put_is_active(spec, t_n1):
            V_node = np.maximum(V_node, spec.put_price)

        # Issuer call (if active)
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

        # Bond block
        face = st.number_input("Face value", value=100.0, step=1.0, format="%.2f", key="cv_face")
        coupon_rate = st.number_input("Annual coupon (%)", value=2.00, step=0.25, format="%.2f", key="cv_cpn") / 100.0

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

        # ---- Embedded options (always enabled) ----
        st.markdown("##### Embedded options")
        eo1, eo2 = st.columns(2)
        with eo1:
            call_price = st.number_input("Call price", value=105.0, step=0.5, format="%.2f", key="cv_call_px")
            call_start = st.number_input("Callable from (year)", value=3.0, step=0.5, format="%.1f", key="cv_call_from")
        with eo2:
            put_price = st.number_input("Put price", value=95.0, step=0.5, format="%.2f", key="cv_put_px")
            put_start = st.number_input("Puttable from (year)", value=2.0, step=0.5, format="%.1f", key="cv_put_from")

        steps = int(st.slider("Tree steps (binomial)", min_value=50, max_value=800, value=200, step=10, key="cv_steps"))
        st.caption("Numerical grid: more steps improve accuracy but increase computation time.")

    # -------------------
    # Top-Right: Method overview + Examples — Download
    # -------------------
    with colTR:
        st.subheader("Method overview")

        st.markdown(
            """
            Convertible bonds can be viewed as a straight bond plus an embedded conversion option, with an investor put and an issuer call.
            Because multiple continuous decisions can occur over time, pricing is path-dependent.
            """
        )

        st.markdown(
            """
            **Definitions**
            - *Issuer call*: the issuer may redeem the bond early at a preset call price.
            - *Investor put*: the holder may sell the bond back to the issuer at a preset put price.
            - *Conversion option*: the right for the bondholder to exchange each bond for a predetermined number of the issuer’s shares,
              according to the conversion ratio. The parity (the current equity value of one bond if converted) equals the current share price (S₀)
              multiplied by the conversion ratio.
            """
        )

        st.markdown(
            """
            **Approach (binomial)**
            1) Compute the straight bond leg by discounting coupons and principal at *(risk-free + credit spread)* → bond floor.  
            2) Price the conversion feature on a Cox–Ross–Rubinstein tree (risk-neutral, with dividend yield if any).  
            3) At each node, apply the optimal decision: continue, convert, exercise the put, or (if called) settle at the better of conversion vs call price. Backward induction yields today’s price.
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
        callable=True, call_price=call_price, call_start_year=call_start,
        putable=True, put_price=put_price, put_start_year=put_start,
        steps=steps
    )

    price = _binomial_convertible_pricer(spec)
    bond_floor = _straight_bond_price_floor(spec)
    conv_val_now, parity = _parity_and_conv_value(spec)
    delta, gamma = _greek_delta_gamma(spec)

    # -------------------
    # Bottom-Left: Chart — Convertible price vs current stock price S₀
    # -------------------


    # -------------------
    # Bottom-Left: Chart — Convertible price vs current stock price S₀
    # -------------------
    with colBL:
        st.subheader("Price vs Stock")

        # --- Build pricing curve over a range of S
        S_min = max(0.05, 0.25 * spec.s0)
        S_max = 2.25 * spec.s0
        n_pts = 61
        S_grid = np.linspace(S_min, S_max, n_pts)

        conv_prices = []
        for S in S_grid:
            sp = ConvSpec(**{**spec.__dict__, "s0": float(S)})
            conv_prices.append(_binomial_convertible_pricer(sp))

        df = pd.DataFrame({
            "Stock (S)": S_grid,
            "Convertible price": conv_prices,
        })
        df["Parity = S × ratio"] = df["Stock (S)"] * spec.conv_ratio
        df["Bond floor"] = bond_floor

        # --- Locate break-even where convertible price ≈ parity
        diff = df["Convertible price"] - df["Parity = S × ratio"]
        sign = np.sign(diff.values)
        cross_idx = np.where(np.diff(sign) != 0)[0]
        be_point = None
        if cross_idx.size > 0:
            i = cross_idx[0]
            # Linear interpolation for the crossing
            x0, x1 = df["Stock (S)"].iloc[i], df["Stock (S)"].iloc[i + 1]
            y0, y1 = diff.iloc[i], diff.iloc[i + 1]
            if (y1 - y0) != 0:
                s_be = float(x0 - y0 * (x1 - x0) / (y1 - y0))
                p_be = float(s_be * spec.conv_ratio)  # at break-even price ≈ parity
                be_point = pd.DataFrame({"Stock (S)": [s_be], "Price": [p_be]})

        # --- Altair chart
        df_long = df.melt(id_vars=["Stock (S)"], 
                          value_vars=["Convertible price", "Parity = S × ratio", "Bond floor"],
                          var_name="Series", value_name="Price")

        dash_map = {
            "Convertible price": [1, 0],     # solid
            "Parity = S × ratio": [6, 4],    # dashed
            "Bond floor": [2, 4],            # dotted
        }
        df_long["Dash"] = df_long["Series"].map(dash_map)

        line_main = alt.Chart(df_long).mark_line(strokeWidth=3).encode(
            x=alt.X("Stock (S):Q", title="Underlying stock price (S)"),
            y=alt.Y("Price:Q", title="Value (per bond)"),
            color=alt.Color("Series:N",
                            scale=alt.Scale(
                                domain=["Convertible price", "Parity = S × ratio", "Bond floor"],
                                range=["#2563eb", "#f4d03f", "#94a3b8"]
                            ),
                            legend=alt.Legend(title="Legend")),
            strokeDash="Dash:N",
            tooltip=[
                alt.Tooltip("Stock (S):Q", format=",.2f"),
                alt.Tooltip("Series:N"),
                alt.Tooltip("Price:Q", format=",.2f"),
            ]
        )

        # Vertical marker at current S₀
        vline = alt.Chart(pd.DataFrame({"Stock (S)": [spec.s0]})).mark_rule(
            stroke="#111827", strokeWidth=1.5, opacity=0.35
        ).encode(x="Stock (S):Q")

        vline_label = alt.Chart(pd.DataFrame({"Stock (S)": [spec.s0], "label": ["Current S₀"]})).mark_text(
            dy=-6, align="right", baseline="bottom", fontSize=11, color="#111827", opacity=0.7
        ).encode(
            x="Stock (S):Q",
            y=alt.value(0),
            text="label:N"
        )

        # Horizontal references: call & put prices
        ref_lines_df = pd.DataFrame({
            "label": ["Call price", "Put price"],
            "y": [spec.call_price, spec.put_price],
            "color": ["#ef4444", "#10b981"]
        })
        ref_rules = alt.Chart(ref_lines_df).mark_rule(strokeDash=[3, 4], strokeWidth=1.2).encode(
            y="y:Q",
            color=alt.Color("label:N", scale=alt.Scale(domain=["Call price", "Put price"],
                                                       range=["#ef4444", "#10b981"]),
                            legend=alt.Legend(title="Thresholds"))
        )
        ref_text = alt.Chart(ref_lines_df).mark_text(align="left", dx=6, dy=-4, fontSize=11).encode(
            y="y:Q",
            x=alt.value(6),
            text=alt.Text("label:N"),
            color=alt.Color("label:N",
                            scale=alt.Scale(domain=["Call price", "Put price"],
                                            range=["#ef4444", "#10b981"]))
        )

        # Break-even point (if any)
        be_layer = None
        if be_point is not None:
            be_dot = alt.Chart(be_point).mark_point(size=90, filled=True, color="#f59e0b").encode(
                x="Stock (S):Q", y="Price:Q"
            )
            be_label = alt.Chart(be_point.assign(label="Break-even (convert)").copy()).mark_text(
                dy=-12, fontSize=11, color="#92400e"
            ).encode(x="Stock (S):Q", y="Price:Q", text="label:N")
            be_layer = be_dot + be_label

        chart = alt.layer(line_main, vline, vline_label, ref_rules, ref_text, *( [be_layer] if be_layer is not None else [] )).properties(
            height=420
        ).interactive(bind_y=False)

        st.altair_chart(chart, use_container_width=True)

        # Short explainer under the chart
        st.markdown(
            """
            <div style="font-size:0.95rem; color:#4b5563; line-height:1.35;">
            <span style="color:#2563eb; font-weight:600;">Convertible price</span> (blue) rises with the stock and is floored by the
            straight bond value (grey). The <span style="color:#f4d03f; font-weight:600;">parity</span> (yellow, dashed) is the immediate equity
            value upon conversion (<code>S × ratio</code>). The orange dot marks the break-even where conversion becomes economically indifferent
            (price ≈ parity). Dashed green/red lines show investor <em>put</em> and issuer <em>call</em> cash settlement references.
            </div>
            """,
            unsafe_allow_html=True
        )


    # -------------------
    # Bottom-Right: Key metrics (rounded to 2 decimals)
    # -------------------
    with colBR:
        st.subheader("Key metrics")

        price_to_par = 100.0 * (price / spec.face)
        conv_premium = 100.0 * (price / parity - 1.0) if parity > 0 else np.nan

        def f2(x: float) -> str:
            try:
                return f"{x:.2f}"
            except Exception:
                return "—"

        left, right = st.columns(2)
        with left:
            st.metric("Convertible price", f2(price))
            st.metric("Bond floor (no conversion)", f2(bond_floor))
            st.metric("Parity = S × ratio", f2(parity))
        with right:
            st.metric("Delta (∂Price/∂S)", f2(delta))
            st.metric("Gamma (∂²Price/∂S²)", f2(gamma))
            st.metric("Price-to-Par (%)", f2(price_to_par))

        with st.expander("Learn More"):
            st.markdown(
                f"""
                <div class="subtle-grey" style="font-size:0.95rem;">
                  <div style="font-weight:600; margin-bottom:.15rem;">Convertible price</div>
                  Tree-based fair value reflecting coupons, conversion, put, and call.

                  <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Bond floor</div>
                  Present value of coupons and principal discounted at <code>r + credit spread</code> (frequency {spec.freq}).

                  <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Parity</div>
                  Immediate conversion value = <code>S₀ × ratio</code>.

                  <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Delta</div>
                  Delta measures how much the convertible’s price changes for a small change in the stock price
                  (first-order equity sensitivity), holding other inputs constant.

                  <div style="font-weight:600; margin-top:.6rem; margin-bottom:.15rem;">Gamma</div>
                  Gamma measures how fast Delta itself changes as the stock price moves (second-order equity convexity).
                  A higher Gamma means Delta adjusts more rapidly when the stock moves, reflecting stronger curvature of price with respect to S.
                </div>
                """,
                unsafe_allow_html=True
            )
