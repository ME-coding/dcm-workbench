from __future__ import annotations

from datetime import date
import io
import os
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st
import altair as alt

# =========================
# Helpers (schedules)
# =========================

def build_amortization(notional: float, rate_annual: float, years: float, freq: int, structure: str) -> pd.DataFrame:
    """Small schedule generator (same logic as in the Pricer)."""
    n = max(1, int(round(freq * years)))
    per = 1 / freq
    r_per = rate_annual / freq
    data = []
    outstanding = notional

    if structure == "bullet":
        coupon_amount = outstanding * r_per
        for i in range(1, n + 1):
            principal = notional if i == n else 0.0
            coupon = coupon_amount
            total = coupon + principal
            end_out = outstanding - principal
            data.append([i, i * per, outstanding, coupon, principal, total, end_out])
            outstanding = end_out

    elif structure == "equal_principal":
        principal_const = notional / n
        for i in range(1, n + 1):
            coupon = outstanding * r_per
            principal = principal_const
            total = coupon + principal
            end_out = outstanding - principal
            data.append([i, i * per, outstanding, coupon, principal, total, end_out])
            outstanding = end_out

    else:  # annuity
        if r_per == 0:
            payment = notional / n
        else:
            payment = notional * r_per / (1 - (1 + r_per) ** (-n))
        for i in range(1, n + 1):
            coupon = outstanding * r_per
            principal = payment - coupon
            total = payment
            end_out = outstanding - principal
            data.append([i, i * per, outstanding, coupon, principal, total, end_out])
            outstanding = end_out

    return pd.DataFrame(
        data,
        columns=[
            "Period",
            "Time (years)",
            "Outstanding (begin)",
            "Interest/Coupon",
            "Principal",
            "Total",
            "Outstanding (end)",
        ],
    )

# =========================
# TOOL 1 — Term Sheet Builder
# =========================
from pathlib import Path
import os
import io
from datetime import date
from typing import Dict, Any
import streamlit as st

# --- Optional deps (docxtpl) ---
try:
    from docxtpl import DocxTemplate
    DOCTPL_OK = True
except Exception:
    DOCTPL_OK = False

# --- Helper: robust template resolver ---
def _find_termsheet_template() -> str:
    """
    Resolve a portable path to 'Termsheet-Example.docx'.
    Order:
      1) st.secrets["TERMSHEET_TEMPLATE"]
      2) env var TEMPLATE_PATH
      3) <module_dir>/Library/Termsheet-Example.docx
      4) CWD/Library/Termsheet-Example.docx
      5) ascend a few levels to find a 'Library' folder containing the file
    """
    filename = "Termsheet-Example.docx"

    # 1) st.secrets
    try:
        if "TERMSHEET_TEMPLATE" in st.secrets:
            p = Path(st.secrets["TERMSHEET_TEMPLATE"]).expanduser().resolve()
            if p.is_file():
                return str(p)
    except Exception:
        pass

    # 2) env var
    env_p = os.getenv("TEMPLATE_PATH")
    if env_p:
        p = Path(env_p).expanduser().resolve()
        if p.is_file():
            return str(p)

    # 3) relative to this file
    here = Path(__file__).resolve()
    p3 = here.parent / "Library" / filename
    if p3.is_file():
        return str(p3)

    # 4) current working dir (when Streamlit sets CWD to repo root)
    p4 = Path.cwd() / "Library" / filename
    if p4.is_file():
        return str(p4)

    # 5) ascend a few levels to find a Library folder with the file
    cur = here.parent
    for _ in range(6):
        cand = cur / "Library" / filename
        if cand.is_file():
            return str(cand.resolve())
        if cur.parent == cur:
            break
        cur = cur.parent

    raise FileNotFoundError(
        f"Template not found: '{filename}'. "
        "Ensure it is committed at 'Library/Termsheet-Example.docx' (case-sensitive on Linux)."
    )

# --- Helper: render Word from template with context ---
def _render_termsheet_docx(template_path: str, context: Dict[str, Any]) -> bytes:
    if not DOCTPL_OK:
        raise RuntimeError("docxtpl not installed (pip install docxtpl)")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    doc = DocxTemplate(template_path)
    bio = io.BytesIO()
    doc.render(context)
    doc.save(bio)
    return bio.getvalue()

# --- Helper: compute yield-to-maturity (annualized) from clean price ---
def _ytm_from_price(clean_price_pct: float, coupon_rate_pct: float, years_to_mty: float, freq: int = 2, redemption: float = 100.0) -> float:
    """
    Approximate YTM solving price = PV(coupons + redemption).
    clean_price_pct: clean price as % of par (e.g., 99.50)
    coupon_rate_pct: annual coupon rate in % (e.g., 4.00)
    years_to_mty: years from settlement to maturity (float)
    freq: coupons per year (1, 2, 4)
    returns annualized YTM in %
    """
    if years_to_mty <= 0 or freq <= 0:
        return 0.0
    C = coupon_rate_pct / 100.0 * 100.0 / freq
    N = max(1, int(round(years_to_mty * freq)))
    P = clean_price_pct  # price per 100 par (clean)
    # Initial guess
    y = max(0.0001, (coupon_rate_pct / max(0.01, clean_price_pct)) + 0.01)
    r = y / freq
    for _ in range(50):
        pv = 0.0
        dp = 0.0
        for t in range(1, N + 1):
            disc = (1.0 + r) ** t
            pv += C / disc
            dp -= t * C / disc / (1.0 + r)
        pv += redemption / ((1.0 + r) ** N)
        dp -= N * redemption / ((1.0 + r) ** (N + 1))
        f = pv - P
        if abs(f) < 1e-8 or dp == 0:
            break
        r -= f / dp
        r = max(-0.9999, min(r, 1.0))
    y_annual = (1.0 + r) ** freq - 1.0
    return max(-99.0, min(y_annual * 100.0, 999.0))

def tool_termsheet():
    st.markdown("#### Term Sheet Builder")

    # === Inputs: uniquement ceux présents dans ton Word ===
    c1, c2, c3 = st.columns(3)
    with c1:
        issuer = st.text_input("Issuer", value="Sample Issuer SA", key="ts_issuer")
        guarantor = st.text_input("Guarantor", value="", key="ts_guarantor")
        currency = st.text_input("Currency", value="EUR", key="ts_currency")
        rating = st.text_input("Rating", value="", key="ts_rating")
        status = st.selectbox(
            "Status",
            ["Senior Unsecured", "Senior Preferred", "Senior Non-Preferred", "Subordinated", "Tier 2", "AT1"],
            index=0, key="ts_status"
        )
    with c2:
        governing_law = st.text_input("Governing law", value="English law", key="ts_law")
        issue_amount = st.number_input("Issue amount (notional)", min_value=100000.0, value=500_000_000.0,
                                       step=1_000_000.0, format="%.2f", key="ts_issue_amount")
        trade_date = st.date_input("Trade date", value=date.today(), key="ts_trade_date")
        issue_date = st.date_input("Issue date", value=date.today(), key="ts_issue_date")
        maturity_date = st.date_input("Maturity date",
                                      value=date.today().replace(year=date.today().year + 5),
                                      key="ts_maturity_date")
    with c3:
        # Coupon rate + frequency (utilisés pour le calcul de yield et pour {{coupon}})
        coupon_rate = st.number_input("Coupon rate (% p.a.)", min_value=0.0, value=4.00, step=0.10,
                                      format="%.4f", key="ts_coupon_rate")
        coupon_freq_label = st.selectbox("Coupon frequency", ["Annual", "Semi-annual", "Quarterly"],
                                         index=1, key="ts_coupon_freq")
        coupon_freq = {"Annual": 1, "Semi-annual": 2, "Quarterly": 4}[coupon_freq_label]

        use = st.text_area("Use of proceeds", value="General corporate purposes.", key="ts_use")
        clean_price = st.number_input("Clean price (% of par)", min_value=0.0, value=99.50, step=0.10,
                                      format="%.2f", key="ts_clean")
        accrued = st.number_input("Accrued (% of par)", min_value=0.0, value=0.50, step=0.10,
                                  format="%.2f", key="ts_accrued")
        issue_price = st.number_input("Issue price (% of par)", min_value=0.0, value=100.00, step=0.10,
                                      format="%.2f", key="ts_issue_px")
        joint_lead_managers = st.text_input(
            "Joint Lead Managers (semicolon- or comma-separated)",
            value="GS; BNP Paribas; J.P. Morgan",
            key="ts_jlm"
        )

    st.markdown("---")
    # === Calcul automatique de {{yield}} à partir du clean price, coupon rate & frequency ===
    years_to_mty = max(0.0, (maturity_date - issue_date).days / 365.0)
    computed_yield = _ytm_from_price(
        clean_price_pct=clean_price,
        coupon_rate_pct=coupon_rate,
        years_to_mty=years_to_mty,
        freq=coupon_freq,
        redemption=100.0
    )
    yield_display = f"{computed_yield:.2f}%"

    # Construire la chaîne {{coupon}} depuis rate + frequency
    coupon_str = f"{coupon_rate:.3f}% ({coupon_freq_label})"

    # === Contexte pour docxtpl (clefs EXACTEMENT celles de ton .docx) ===
    context: Dict[str, Any] = {
        "issuer": issuer.strip(),
        "guarantor": guarantor.strip(),
        "currency": currency.strip() or "EUR",
        "rating": rating.strip(),
        "joint_lead_managers": joint_lead_managers.strip(),
        "status": status,
        "governing_law": governing_law.strip(),
        "governing_law ": governing_law.strip(),  # sécurité si ta balise Word a un espace final
        "issue_amount": f"{currency.strip() or 'EUR'} {issue_amount:,.0f}".replace(",", " "),
        "trade_date": trade_date.strftime("%d %b %Y"),
        "maturity_date": maturity_date.strftime("%d %b %Y"),
        "coupon": coupon_str,                      # << injecté dans {{coupon}}
        "use_of_proceeds": use.strip(),
        "issue_date": issue_date.strftime("%d %b %Y"),
        "clean_price": f"{clean_price:.2f}%",
        "accrued": f"{accrued:.2f}%",
        "issue_price": f"{issue_price:.2f}%",
        "yield": yield_display,                    # calculé automatiquement
        "today": date.today().strftime("%d %b %Y"),
    }

    # === Génération Word ===
    try:
        template_path = _find_termsheet_template()
    except FileNotFoundError as e:
        st.error(str(e))
        return  # ou st.stop()

    if not DOCTPL_OK:
        st.error("`docxtpl` n'est pas installé. Installe-le avec: `pip install docxtpl`")

    if st.button("Generate Word Termsheet", disabled=not DOCTPL_OK):
        try:
            docx_bytes = _render_termsheet_docx(template_path, context)
            st.download_button(
                "Download Word Termsheet",
                data=docx_bytes,
                file_name=f"Termsheet_{issuer.replace(' ', '_')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="ts_docx_download",
            )
        except Exception as e:
            st.error(f"Word generation failed: {e}")

# =========================
# TOOL 2 — Amortization Builder (interactive)
# =========================
def tool_amortization():
    st.markdown("#### Amortization Builder")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        notional = st.number_input("Notional", min_value=1000.0, value=100_000.0,
                                   step=1000.0, format="%.2f", key="amort_notional")
    with c2:
        rate = st.number_input("Rate (annual, %)", min_value=0.0, value=4.00,
                               step=0.10, format="%.4f", key="amort_rate")
    with c3:
        years = st.number_input("Maturity (years)", min_value=0.25, value=5.0,
                                step=0.25, format="%.2f", key="amort_years")
    with c4:
        freq_label = st.selectbox("Frequency", sorted(["Annual", "Semi-annual", "Quarterly"]),
                                  index=1, key="amort_freq_label")
        f_map = {"Annual": 1, "Semi-annual": 2, "Quarterly": 4}
        f = f_map[freq_label]

    structure = st.selectbox("Structure", sorted(["bullet", "equal_principal", "annuity"]),
                             index=2, key="amort_structure")

    df = build_amortization(notional, rate / 100.0, years, f, structure)

    st.dataframe(df.style.format({
        "Outstanding (begin)": "{:,.2f}",
        "Interest/Coupon": "{:,.2f}",
        "Principal": "{:,.2f}",
        "Total": "{:,.2f}",
        "Outstanding (end)": "{:,.2f}",
    }), use_container_width=True)

    # ---- Interactive chart: Stacked bars + line (zoom/pan only)
    bars_df = df.melt(
        id_vars=["Period", "Time (years)", "Outstanding (end)"],
        value_vars=["Principal", "Interest/Coupon"],
        var_name="Component", value_name="Amount"
    )

    bars = alt.Chart(bars_df).mark_bar().encode(
        x=alt.X("Time (years):Q", title="Time (years)"),
        y=alt.Y("Amount:Q", stack="zero", title="Cashflow"),
        color=alt.Color(
            "Component:N",
            sort=["Interest/Coupon", "Principal"],
            legend=alt.Legend(title=None, orient="bottom")
        ),
        tooltip=[
            alt.Tooltip("Period:Q"),
            alt.Tooltip("Time (years):Q", format=",.2f"),
            alt.Tooltip("Component:N"),
            alt.Tooltip("Amount:Q", format=",.2f"),
        ],
    )

    line = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Time (years):Q"),
        y=alt.Y("Outstanding (end):Q", title="Outstanding"),
        tooltip=[
            alt.Tooltip("Period:Q"),
            alt.Tooltip("Time (years):Q", format=",.2f"),
            alt.Tooltip("Outstanding (end):Q", format=",.2f"),
        ],
    )

    chart = (
        alt.layer(bars, line)
          .resolve_scale(y="independent")
          .properties(
              height=360,
              title="Cashflows (stacked) & Outstanding profile"
          )
          .configure_legend(
              orient="bottom",
              direction="horizontal",
              columns=2      # optionnel, compacte la légende
          )
          .configure_title(
              anchor="start",  # ancre le titre au cadre gauche
              offset=12        # espace titre -> tracé
          )
          .configure(
              autosize="pad",
              padding={"left": 40, "right": 16, "top": 8, "bottom": 64}
          )
          .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "Download amortization (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="amortization.csv",
        mime="text/csv",
        key="amort_dl_csv",
    )

# =========================
# TOOL 2 — Amortization Builder (interactive)
# =========================
def tool_amortization():
    st.markdown("#### Amortization Builder")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        notional = st.number_input("Notional", min_value=1000.0, value=100_000.0,
                                   step=1000.0, format="%.2f", key="amort_notional")
    with c2:
        rate = st.number_input("Rate (annual, %)", min_value=0.0, value=4.00,
                               step=0.10, format="%.4f", key="amort_rate")
    with c3:
        years = st.number_input("Maturity (years)", min_value=0.25, value=5.0,
                                step=0.25, format="%.2f", key="amort_years")
    with c4:
        freq_label = st.selectbox("Frequency", sorted(["Annual", "Semi-annual", "Quarterly"]),
                                  index=1, key="amort_freq_label")
        f_map = {"Annual": 1, "Semi-annual": 2, "Quarterly": 4}
        f = f_map[freq_label]

    structure = st.selectbox("Structure", sorted(["bullet", "equal_principal", "annuity"]),
                             index=2, key="amort_structure")

    df = build_amortization(notional, rate / 100.0, years, f, structure)

    st.dataframe(df.style.format({
        "Outstanding (begin)": "{:,.2f}",
        "Interest/Coupon": "{:,.2f}",
        "Principal": "{:,.2f}",
        "Total": "{:,.2f}",
        "Outstanding (end)": "{:,.2f}",
    }), use_container_width=True)

    # ---- Data for stacked bars
    bars_df = df.melt(
        id_vars=["Period", "Time (years)", "Outstanding (end)"],
        value_vars=["Principal", "Interest/Coupon"],
        var_name="Component", value_name="Amount"
    )

    # ---- Title outside the chart (prevents clipping)
    st.markdown("##### Cashflows (stacked) & Outstanding profile")

    # ---- Bars (legend at bottom)
    bars = alt.Chart(bars_df).mark_bar().encode(
        x=alt.X("Time (years):Q", title="Time (years)"),
        y=alt.Y("Amount:Q", stack="zero", title="Cashflow"),
        color=alt.Color(
            "Component:N",
            sort=["Interest/Coupon", "Principal"],
            legend=alt.Legend(title=None, orient="bottom")
        ),
        tooltip=[
            alt.Tooltip("Period:Q"),
            alt.Tooltip("Time (years):Q", format=",.2f"),
            alt.Tooltip("Component:N"),
            alt.Tooltip("Amount:Q", format=",.2f"),
        ],
    )

    # ---- Line (right axis)
    line = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Time (years):Q"),
        y=alt.Y(
            "Outstanding (end):Q",
            title="Outstanding",
            axis=alt.Axis(orient="right", labelAlign="left", labelPadding=4, titlePadding=8)
        ),
        tooltip=[
            alt.Tooltip("Period:Q"),
            alt.Tooltip("Time (years):Q", format=",.2f"),
            alt.Tooltip("Outstanding (end):Q", format=",.2f"),
        ],
    )

    chart = (
        alt.layer(bars, line)
          .resolve_scale(y="independent")
          .properties(height=360)                # no Altair title inside
          .configure_legend(orient="bottom", direction="horizontal")
          .configure(autosize="pad",
                     padding={"left": 40, "right": 80, "top": 4, "bottom": 64})
          .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "Download amortization (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="amortization.csv",
        mime="text/csv",
        key="amort_dl_csv",
    )

# =========================
# Main render()
# =========================

def render():
    st.subheader("Structuring Desk — Tools")
    st.caption("Quick, self-contained utilities for everyday DCM tasks. No external data sources required.")

    tabs = st.tabs([
        "📑 Term Sheet Builder",
        "📊 Amortization Builder",
    ])

    with tabs[0]:
        tool_termsheet()
    with tabs[1]:
        tool_amortization()
