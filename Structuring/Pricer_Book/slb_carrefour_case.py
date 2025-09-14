# Structuring/Pricer_Book/slb_carrefour_case.py
# SLB — Carrefour (educational) | Component-style: expose render()

from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ----------------------------
# Helper data structures
# ----------------------------
@dataclass
class SLBTerms:
    notional: float
    coupon: float                 # e.g., 0.0375
    issue_date: date
    first_full_ipd: date
    maturity_date: date
    payment_month_day: Tuple[int, int]
    day_count: str = "Actual/Actual-ICMA"  # label only (simplified accrual)
    step_up_bps: float = 25.0
    step_up_years: List[int] = None
    denomination: float = 100_000.0

    def schedule_years(self) -> List[int]:
        start_year = self.first_full_ipd.year
        end_year = self.maturity_date.year
        return list(range(start_year, end_year + 1))

# ----------------------------
# Defaults (Carrefour June 2025)
# ----------------------------
_DEFAULT_TERMS = SLBTerms(
    notional=650_000_000.0,
    coupon=0.0375,
    issue_date=date(2025, 6, 24),
    first_full_ipd=date(2026, 5, 24),
    maturity_date=date(2033, 5, 24),
    payment_month_day=(5, 24),
    step_up_bps=25.0,
    step_up_years=[2031, 2032, 2033],
    denomination=100_000.0
)

# ----------------------------
# Internal helpers
# ----------------------------
def _build_cashflows(terms: SLBTerms, coupon_rate: float, step_up_bps: float,
                     step_up_years: List[int], apply_step_up: bool) -> pd.DataFrame:
    years = terms.schedule_years()
    rows = []
    for y in years:
        effective_coupon_rate = coupon_rate
        stepup_extra = 0.0
        if apply_step_up and (y in step_up_years):
            effective_coupon_rate = coupon_rate + step_up_bps / 10_000.0
            stepup_extra = (effective_coupon_rate - coupon_rate) * terms.notional
        coupon_cash = effective_coupon_rate * terms.notional
        principal = terms.notional if y == terms.maturity_date.year else 0.0
        rows.append({
            "year": y,
            "coupon_rate_%": effective_coupon_rate * 100.0,
            "coupon_cash": coupon_cash,
            "stepup_extra": stepup_extra,
            "principal": principal,
            "total_cf": coupon_cash + principal
        })
    return pd.DataFrame(rows)

def _price_from_cashflows(df: pd.DataFrame, flat_yield_pct: float) -> float:
    y = flat_yield_pct / 100.0
    price = 0.0
    # Pedagogical: discount each annual CF back to the first full IPD as t = 1..N
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        price += r["total_cf"] / ((1 + y) ** i)
    return price

# ----------------------------
# Public entrypoint
# ----------------------------
def render():
    # --- Header ---
    st.title("SLB — Carrefour (KPI Simulator & Pricing)")
    st.caption("Sandbox pédagogique : simulate les KPIs 2030 → observe step-ups (2031–2033), coupons, cash-flows et prix.")

    with st.expander("Résumé du cas (source publique)", expanded=True):
        st.markdown(
            """
**Aperçu.** Juin 2025, Carrefour émet un **SLB €650m**, **coupon 3.75%** annuel, maturité **24 mai 2033**.  
Les coupons **payés en 2031, 2032, 2033** montent de **+25 bps** si au 31 déc. 2030 **au moins un** des objectifs n’est atteint :  
(i) **–50%** d’émissions GES Scope 1&2 vs 2019 ; (ii) **150 fournisseurs** engagés dans une stratégie climat.  
*NB: conventions et discounting simplifiés pour fins pédagogiques.*
"""
        )

    st.markdown("---")
    st.header("Paramètres interactifs")

    colA, colB, colC = st.columns([1.2, 1.1, 1.1])

    # --- KPIs ---
    with colA:
        st.subheader("KPIs (au 31 déc. 2030)")
        ghg_now = st.slider(
            "Réduction GES Scope 1&2 vs 2019",
            min_value=0, max_value=100, value=45, step=1, format="%d%%",
            help="Cible: au moins 50%."
        )
        suppliers_now = st.slider(
            "Fournisseurs engagés dans une stratégie climat",
            min_value=0, max_value=300, value=120, step=5,
            help="Cible: au moins 150."
        )
        st.markdown(f"- Seuils cibles: **≥ 50%** GES ; **≥ 150** fournisseurs\n- Tes inputs: **{ghg_now}%** ; **{suppliers_now}**")

    # --- Terms ---
    with colB:
        st.subheader("Term sheet")
        notional = st.number_input("Notional (EUR)", min_value=1_000_000.0, value=_DEFAULT_TERMS.notional, step=1_000_000.0)
        coupon_pct = st.number_input("Coupon de base (%)", min_value=0.0, value=_DEFAULT_TERMS.coupon * 100, step=0.05, format="%.2f")
        step_up_bps = st.number_input("Step-up si objectifs manqués (bps)", min_value=0.0, value=_DEFAULT_TERMS.step_up_bps, step=5.0)
        step_up_years = st.multiselect(
            "Années de paiement où le step-up s'applique",
            options=_DEFAULT_TERMS.schedule_years(),
            default=_DEFAULT_TERMS.step_up_years
        )

    # --- Pricing ---
    with colC:
        st.subheader("Pricing (flat)")
        st.markdown('<div style="color:#6b7280;font-size:0.95rem;margin-top:-0.25rem;">Taux plat pour un prix propre sur dates coupon (vision pédagogique).</div>', unsafe_allow_html=True)
        y_base_pct = st.slider("Taux de discount (%)", 0.0, 10.0, 3.80, 0.05)
        show_pv_breakdown = st.checkbox("Afficher le tableau de cash-flows actualisés", value=False)

    # Trigger logic: step-up si ≥1 KPI manqué
    miss_ghg = ghg_now < 50
    miss_sup = suppliers_now < 150
    miss_any = miss_ghg or miss_sup

    # Terms instance (à partir des inputs)
    terms = SLBTerms(
        notional=notional,
        coupon=coupon_pct / 100.0,
        issue_date=_DEFAULT_TERMS.issue_date,
        first_full_ipd=_DEFAULT_TERMS.first_full_ipd,
        maturity_date=_DEFAULT_TERMS.maturity_date,
        payment_month_day=_DEFAULT_TERMS.payment_month_day,
        step_up_bps=step_up_bps,
        step_up_years=step_up_years,
        denomination=_DEFAULT_TERMS.denomination
    )

    # Cash-flows
    flows_base = _build_cashflows(terms, terms.coupon, terms.step_up_bps, terms.step_up_years, apply_step_up=False)
    flows_step = _build_cashflows(terms, terms.coupon, terms.step_up_bps, terms.step_up_years, apply_step_up=True) if miss_any else flows_base.copy()

    # Prices
    price_base = _price_from_cashflows(flows_base, y_base_pct)
    price_step = _price_from_cashflows(flows_step, y_base_pct)

    # Status pill
    pill_text = "Pas de step-up (objectifs atteints)" if not miss_any else "Step-up appliqué (≥1 objectif manqué)"
    pill_color = "#10b981" if not miss_any else "#ef4444"
    st.markdown(f'<span style="display:inline-block;padding:.2rem .6rem;border-radius:999px;background:#eefdf6;color:{pill_color};font-weight:600;font-size:.9rem;">{pill_text}</span>', unsafe_allow_html=True)

    st.markdown(
        f"""
<div style="background:#f9fafb;border:1px solid #e5e7eb;padding:.75rem .85rem;border-radius:.5rem;font-size:.95rem;">
<strong>Clean price (par rapport au notional total € {int(terms.notional):,}, discount {y_base_pct:.2f}%):</strong><br>
• Si objectifs atteints : € {price_base:,.0f}<br>
• Si objectifs manqués : € {price_step:,.0f}<br><br>
<strong>Écart de prix :</strong> € {price_step - price_base:,.0f}
</div>
        """,
        unsafe_allow_html=True
    )

    # --- Charts side-by-side ---
    left, right = st.columns(2)

    with left:
        st.subheader("Trajectoire de coupon")
        st.markdown('<div style="color:#6b7280;font-size:0.95rem;margin-top:-0.25rem;">Coupons annuels payés ; le step-up ne s’applique qu’aux derniers coupons si la cible 2030 est manquée.</div>', unsafe_allow_html=True)

        base_line = flows_base[["year", "coupon_rate_%"]].copy()
        base_line["scenario"] = "Base (objectifs atteints)"
        step_line = flows_step[["year", "coupon_rate_%"]].copy()
        step_line["scenario"] = "Selon tes inputs"
        chart_data = pd.concat([base_line, step_line], ignore_index=True)

        line = alt.Chart(chart_data).mark_line(point=True).encode(
            x=alt.X("year:O", title="Année de coupon"),
            y=alt.Y("coupon_rate_%:Q", title="Coupon (%)"),
            color=alt.Color("scenario:N", title="Scénario")
        ).properties(height=360).interactive()

        st.altair_chart(line, use_container_width=True)
        st.caption("Comparaison des coupons avec/sans step-up selon les KPIs 2030.")

    with right:
        st.subheader("Prix vs taux de discount")
        st.markdown('<div style="color:#6b7280;font-size:0.95rem;margin-top:-0.25rem;">Prix propre (dates de coupon) en fonction d’un taux plat, avec et sans step-up.</div>', unsafe_allow_html=True)

        y_grid = np.linspace(1.0, 7.0, 31)
        prices_base = [_price_from_cashflows(flows_base, y_) for y_ in y_grid]
        prices_step = [_price_from_cashflows(flows_step, y_) for y_ in y_grid]

        df_price = pd.DataFrame({
            "discount_rate_%": y_grid,
            "price_if_targets_met": prices_base,
            "price_if_targets_missed": prices_step
        }).melt(id_vars="discount_rate_%", var_name="scenario", value_name="price_eur")

        area = alt.Chart(df_price).mark_line().encode(
            x=alt.X("discount_rate_%:Q", title="Taux plat (%)"),
            y=alt.Y("price_eur:Q", title=f"Prix (EUR, notional € {int(terms.notional):,})"),
            color=alt.Color("scenario:N", title="Scénario")
        ).properties(height=360).interactive()

        st.altair_chart(area, use_container_width=True)
        st.caption("Le step-up augmente légèrement les derniers coupons → soutient le prix pour un même taux.")

    # --- PV table (optional) ---
    if show_pv_breakdown:
        st.markdown("### Tableau des cash-flows actualisés")
        y = y_base_pct / 100.0
        df_cf = flows_step.copy()
        df_cf["t"] = np.arange(1, len(df_cf) + 1)
        df_cf["df"] = 1.0 / ((1 + y) ** df_cf["t"])
        df_cf["pv_total_cf"] = df_cf["total_cf"] * df_cf["df"]
        df_cf.rename(columns={
            "year": "Année coupon",
            "coupon_rate_%": "Coupon (%)",
            "coupon_cash": "Coupon (€)",
            "stepup_extra": "Step-up (€)",
            "principal": "Principal (€)",
            "total_cf": "Total CF (€)",
            "df": "Facteur d'actualisation",
            "pv_total_cf": "VA du CF (€)"
        }, inplace=True)
        st.dataframe(df_cf, use_container_width=True)

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        """
**Lecture.** Dans un SLB, la **réalisation d’objectifs** (pas la liste de projets) pilote les **coupons futurs**.  
Si les cibles 2030 sont atteintes → coupons restent au taux de base. Sinon → **step-up** sur 2031–2033.  
Impact: plus de cash **tard dans la vie du bond**, donc un prix un peu plus élevé (à taux donné) et un meilleur alignement incitatif.
"""
    )

# (Optionnel) exécution directe locale
if __name__ == "__main__":
    render()
