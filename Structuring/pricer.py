# Structuring_Desk/pricer.py
# -----------------------------------------------------------------------------
# Structuring Desk — Pricer (router)
# -----------------------------------------------------------------------------

import streamlit as st

from Structuring.Pricer_Book.sukuk import render as sukuk_ui
from Structuring.Pricer_Book.option_bond import render as option_bond_ui
from Structuring.Pricer_Book.zero_coupon import render as zero_coupon_ui
from Structuring.Pricer_Book.fixed_floating_bond import render as fixed_floating_bond_ui
from Structuring.Pricer_Book.slb_carrefour_case import render as slb_carrefour_case_ui
from Structuring.Pricer_Book.convertible_bond import convertible_bond_ui

from .Pricer_Book.visuals import price_yield_chart

st.markdown(
    """
    <style>
    .inline-help { margin-top: 28px; }
    </style>
    """,
    unsafe_allow_html=True,
)

def render():
    st.subheader("Structuring Desk — Pricer")

    products = [
        "Fixed / Float / Fixed-to-Float Rate Bond",   # <— fusion des 3 choix
        "Bonds with Options (Callable / Puttable)",
        "Convertible Bond (LSMC)",
        "Sukuk (Fixed/Ijara-style)",
        "Sustainability-Linked Bond — Carrefour Case",
        "Zero-Coupon",
    ]

    product_defs = {
        "Fixed / Float / Fixed-to-Float Rate Bond":
            "Obligations à taux fixe, flottant (réf. SOFR/Euribor + spread) ou à bascule fixe→flottant.",
        "Bonds with Options (Callable / Puttable)":
            "Call émetteur / put investisseur à des dates/prix prédéfinis.",
        "Convertible Bond (LSMC)":
            "Dette + option actions ; pricer par régression LSMC.",
        "Sukuk (Fixed/Ijara-style)":
            "Conforme à la sharia ; modèle de cash-flows ‘profit rate’ pédagogique.",
        "Sustainability-Linked Bond — Carrefour Case":
            "Coupons modulés par KPIs (Scope1&2 ; fournisseurs).",
        "Zero-Coupon":
            "Titre à escompte ; pas de coupons périodiques.",
    }

    st.markdown("### Parameters")

    r1c1, r1c2 = st.columns([3, 0.6])
    with r1c1:
        product = st.selectbox("Select a product", products, index=0)
    with r1c2:
        st.markdown('<div class="inline-help">', unsafe_allow_html=True)
        if hasattr(st, "dialog"):
            @st.dialog("Product definitions")
            def _show_defs():
                st.markdown("\n\n".join([f"**{k}**: {v}" for k, v in product_defs.items()]))
            if st.button("💡", key="defs_btn"):
                _show_defs()
        else:
            with st.expander("💡 Product definitions"):
                st.markdown("\n\n".join([f"**{k}**: {v}" for k, v in product_defs.items()]))
        st.markdown('</div>', unsafe_allow_html=True)

    registry = {
        "Fixed / Float / Fixed-to-Float Rate Bond": fixed_floating_bond_ui,  # <— route unique
        "Bonds with Options (Callable / Puttable)": option_bond_ui,
        "Convertible Bond (LSMC)": convertible_bond_ui,
        "Sukuk (Fixed/Ijara-style)": sukuk_ui,
        "Sustainability-Linked Bond — Carrefour Case": slb_carrefour_case_ui,
        "Zero-Coupon": zero_coupon_ui,
    }

    ui_fn = registry.get(product)
    if ui_fn is None:
        st.error("UI not wired for this product yet. Please check the registry.")
        return

    ui_fn()

if __name__ == "__main__":
    render()
