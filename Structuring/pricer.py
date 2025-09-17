# Structuring_Desk/pricer.py
# -----------------------------------------------------------------------------
# Structuring Desk — Pricer (tabs router, v2 minimal)
# -----------------------------------------------------------------------------
import streamlit as st

from Structuring.Pricer_Book.vanilla_bond import render as vanilla_bond_ui
from Structuring.Pricer_Book.convertibles_options import render as conv_opt_ui
from Structuring.Pricer_Book.slb import render as slb_ui

def render():
    st.subheader("Structuring Desk — Pricer")

    # Petit texte d’intro — placeholder pour l’instant
    st.markdown(
        "In this Pricer, we’ll progressively add modules. "
        "Pick a tab below to explore the product family."
    )

    tabs = st.tabs([
        "🧾 Vanilla Bond",
        "🔀 Convertible & Options",
        "🌿 Sustainability-Linked Bond",
    ])

    with tabs[0]:
        vanilla_bond_ui()

    with tabs[1]:
        conv_opt_ui()

    with tabs[2]:
        slb_ui()

if __name__ == "__main__":
    render()
