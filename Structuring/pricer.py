import streamlit as st

from Structuring.Pricer_Book.vanilla_bond import render as vanilla_bond_ui
from Structuring.Pricer_Book.convertibles_options import render as conv_opt_ui
from Structuring.Pricer_Book.slb import render as slb_ui
from Structuring.Pricer_Book.zero_coupon import render as zero_coupon_ui

def render():
    st.subheader("Structuring Desk — Pricer")
    st.markdown(
        "In this Pricer, we’ll progressively add modules. "
        "Pick a tab below to explore the product family."
    )

    tabs = st.tabs([
        "🧾 Vanilla Bond",
        "0️⃣ Zero-Coupon",
        "🔀 Convertible & Options",
        "🌿 Sustainability-Linked Bond",
    ])

    with tabs[0]:
        vanilla_bond_ui()
    with tabs[1]:
        zero_coupon_ui()
    with tabs[2]:
        conv_opt_ui()
    with tabs[3]:
        slb_ui()