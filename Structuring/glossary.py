# Structuring/glossary.py
# -----------------------------------------------------------------------------
# DCM Workbench — Glossary & Learn More
# -----------------------------------------------------------------------------

from __future__ import annotations
import streamlit as st
import pandas as pd

def render():
    st.subheader("Glossary & Learn More")

    tabs = st.tabs(["📘 Glossary (pricing & convertibles)", "🔎 Learn More (exotic bonds)"])

    # =========================================================================
    # 📘 GLOSSARY  ———  >>> PARTIE MODIFIÉE SEULEMENT <<<
    # =========================================================================
    with tabs[0]:
        # Intro plus pédagogique (public: connaît la finance, moins le DCM)
        st.markdown(
            "This glossary provides a DCM dictionary of key vocabulary and formulas to better understand bond pricing methodologies."
        )

        # --- Données Glossaire (titres sans abréviations) --------------------
        entries = [
            # ===== Pricing — Core (en premier) =====
            {
                "Category": "Pricing — Core",
                "Term": "Present Value (Bond Price)",
                "FormulaLatex": r"P=\sum_{t=1}^{N}\frac{C}{\left(1+\frac{y}{m}\right)^{t}}+\frac{F}{\left(1+\frac{y}{m}\right)^{N}}",
                "Interpretation": (
                    "The bond price equals the discounted value of all coupon payments plus the redemption of face value; "
                    "a higher required yield lowers present value and a lower required yield raises it."
                )
            },
            {
                "Category": "Pricing — Core",
                "Term": "Coupon Rate",
                "FormulaLatex": r"\text{Coupon Rate}=\dfrac{\text{Annual Coupon}}{F}",
                "Interpretation": (
                    "The coupon rate is the fixed percentage of face value that determines the annual coupon amount; "
                    "periodic coupons equal the annual coupon divided by the number of payments per year."
                )
            },
            {
                "Category": "Pricing — Core",
                "Term": "Current Yield",
                "FormulaLatex": r"\text{Current Yield}=\dfrac{\text{Annual Coupon}}{P}",
                "Interpretation": (
                    "Income return at today’s market price; it ignores time value and any capital gain or loss at maturity."
                )
            },
            {
                "Category": "Pricing — Core",
                "Term": "Yield to Maturity",
                "FormulaLatex": r"P=\sum_{t=1}^{N}\frac{C}{(1+\text{Yield to Maturity})^{t}}+\frac{F}{(1+\text{Yield to Maturity})^{N}}",
                "Interpretation": (
                    "The internal rate of return of the bond if held to maturity and coupons are reinvested at the same rate; "
                    "it is the discount rate that exactly reproduces the market price."
                )
            },
            {
                "Category": "Pricing — Core",
                "Term": "Accrued Interest and Clean versus Dirty Price",
                "FormulaLatex": r"\text{Dirty Price}=\text{Clean Price}+\text{Accrued Interest},\quad \text{Accrued Interest}= \text{Coupon}\times \dfrac{\text{Days since last}}{\text{Days in period}}",
                "Interpretation": (
                    "The quoted (clean) price excludes accrued interest since the last coupon date, while the cash (dirty) price includes it; "
                    "the fraction of the period is set by the relevant day-count convention."
                )
            },
            {
                "Category": "Pricing — Core",
                "Term": "Day-Count Convention (overview)",
                "FormulaLatex": r"\text{Accrued Interest}\propto \dfrac{\text{Actual or 30 days since last}}{\text{Actual or 360 days in period}}",
                "Interpretation": (
                    "Markets use conventions such as **Actual/Actual**, **30/360**, or **Actual/360** to measure time between coupon dates; "
                    "the choice affects accrued interest and quoted yields."
                )
            },

            # ===== Convertibles (fusion des sous-catégories) =====
            {
                "Category": "Convertibles",
                "Term": "Conversion Ratio",
                "FormulaLatex": r"\text{Conversion Ratio}=\dfrac{\text{Face Value}}{\text{Conversion Price}}",
                "Interpretation": "Number of shares received when one bond is converted at the stated conversion price."
            },
            {
                "Category": "Convertibles",
                "Term": "Conversion Value (also called Parity)",
                "FormulaLatex": r"\text{Conversion Value}=\text{Share Price}\times \text{Conversion Ratio}",
                "Interpretation": "Equity value embedded in the convertible if it were converted at the current share price."
            },
            {
                "Category": "Convertibles",
                "Term": "Conversion Premium",
                "FormulaLatex": r"\text{Conversion Premium}=\dfrac{\text{Convertible Price}-\text{Conversion Value}}{\text{Conversion Value}}",
                "Interpretation": "Percentage markup investors pay above immediate equity value in exchange for downside protection."
            },
            {
                "Category": "Convertibles",
                "Term": "Break-Even Time",
                "FormulaLatex": r"\text{Break-Even (years)}=\dfrac{\text{Conversion Premium}}{\text{Yield on Convertible}-\text{Dividend Yield on Shares}}",
                "Interpretation": "Years of income advantage needed for the convertible to recoup its premium relative to the underlying shares."
            },
            {
                "Category": "Convertibles",
                "Term": "Investment Value and Floor",
                "FormulaLatex": r"\text{Investment Value}=\sum_{t=1}^{N}\dfrac{CF_t}{(1+r_{\text{straight}})^{t}},\quad \text{Floor}=\max\{\text{Conversion Value},\ \text{Investment Value}\}",
                "Interpretation": "Minimum economic value of a convertible: the greater of its equity value if converted and its straight-bond value."
            },
            {
                "Category": "Convertibles",
                "Term": "Adjusted Duration of a Convertible Bond",
                "FormulaLatex": r"D_{\text{adjusted}}\approx D_{\text{convertible}}\!\left(1-\dfrac{\text{Equity Component}}{\text{Total Value}}\right)",
                "Interpretation": "Equity participation reduces interest-rate sensitivity; a larger equity component leads to lower duration."
            },

            # ===== Risk — Interest Rate Sensitivity =====
            {
                "Category": "Risk — Interest Rate Sensitivity",
                "Term": "Macaulay Duration",
                "FormulaLatex": r"D=\dfrac{\sum_{t=1}^{N}\dfrac{t\cdot CF_t}{(1+i)^{t}}}{\sum_{t=1}^{N}\dfrac{CF_t}{(1+i)^{t}}}",
                "Interpretation": "Cash-flow-weighted average time to receive payments; useful baseline for interest-rate risk."
            },
            {
                "Category": "Risk — Interest Rate Sensitivity",
                "Term": "Modified Duration",
                "FormulaLatex": r"D_{\text{modified}}=\dfrac{D}{1+i}",
                "Interpretation": "First-order price sensitivity to yield changes: approximate percentage price change ≈ −(modified duration) × yield change."
            },
            {
                "Category": "Risk — Interest Rate Sensitivity",
                "Term": "Convexity (outline)",
                "FormulaLatex": r"\dfrac{\Delta P}{P}\approx -D_{\text{modified}}\Delta y+\tfrac{1}{2}\text{Convexity}\cdot(\Delta y)^2",
                "Interpretation": "Second-order effect that refines duration by capturing curve-shape benefits when yields move."
            },
        ]

        df = pd.DataFrame(entries)

        # --- Contrôles (filtre + mode tableau) --------------------------------
        col1, col2 = st.columns([1,1])
        with col1:
            q = st.text_input("Filter by term or category", "")
        with col2:
            table_mode = st.toggle("Show compact table view", value=False)

        if q:
            mask = df["Term"].str.contains(q, case=False, na=False) | df["Category"].str.contains(q, case=False, na=False)
            df = df[mask]

        # Ordre des catégories (Pricing — Core en premier)
        cat_order = ["Pricing — Core", "Convertibles", "Risk — Interest Rate Sensitivity"]
        df["__cat_rank"] = df["Category"].apply(lambda c: cat_order.index(c) if c in cat_order else 999)

        # --- Rendu -------------------------------------------------------------
        if table_mode:
            out = df.sort_values(["__cat_rank", "Category", "Term"])[["Category", "Term", "FormulaLatex", "Interpretation"]]
            out = out.rename(columns={"FormulaLatex": "Formula"})
            st.dataframe(out, use_container_width=True, hide_index=True)
        else:
            for cat, block in (
                df.sort_values(["__cat_rank", "Term"])
                  .groupby("Category", sort=False)
            ):
                with st.expander(f"🔹 {cat}", expanded=(cat == "Pricing — Core")):
                    for _, row in block.sort_values("Term").iterrows():
                        st.markdown(f"**{row['Term']}**")
                        st.latex(row["FormulaLatex"])
                        st.markdown(f"*{row['Interpretation']}*")
                        st.markdown("---")  # dashed separator between terms

            # --------- Légende des symboles / variables utilisées --------------
            st.markdown("#### Notation and inputs used in formulas")
            st.markdown(
                """
- **P**: Present value or market price of the bond at settlement.  
- **F**: Face value (also called par value or principal) repaid at maturity.  
- **C**: Coupon payment per period (annual coupon divided by number of payments per year).  
- **y**: Nominal annual required yield (quoted yield); **m**: number of coupon payments per year; the per-period rate is *y / m*.  
- **N**: Total number of coupon periods remaining until maturity; **t**: period index (1, 2, …, N).  
- **CFₜ**: Cash flow paid at period *t* (coupon or coupon plus redemption at maturity).  
- **i**: Effective discount rate per period used in duration and convexity formulas.  
- **Days since last / Days in period**: Fraction of the coupon period used to compute accrued interest (per the day-count convention).  
- **Share Price**: Current price of the underlying share for a convertible bond.  
- **Conversion Price**: Price per share at which the bond converts into equity; set at issuance (subject to anti-dilution provisions).  
- **Face Value (for convertibles)**: Principal amount used to compute the conversion ratio.  
- **Yield on Convertible**: Yield to maturity calculated on the convertible bond’s market price.  
- **Dividend Yield on Shares**: Expected cash dividends divided by the current share price.  
- **Straight-Bond Discount Rate** (*r*<sub>straight</sub>): Yield appropriate for valuing the bond component of a convertible (credit- and term-matched).  
                """
            )

    # =========================================================================
    # 🔎 LEARN MORE  ———  (inchangé)
    # =========================================================================
    with tabs[1]:
        st.markdown(
            "There are many bond types designed to meet different issuer and investor needs. Here is a **non-exhaustive** list to spark curiosity and further exploration!"
        )

        sections = [
            ("Based on Issuer",
             [
                 ("Bulldog Bond", "A sterling-denominated bond issued in the U.K. by a non-British entity."),
                 ("Dim Sum Bond", "A renminbi-denominated bond issued outside mainland China, typically in Hong Kong."),
                 ("Formosa Bond", "A Taiwan-issued bond, typically denominated in foreign currencies like USD, sold to local investors."),
                 ("Kangaroo Bond", "An Australian dollar-denominated bond issued in Australia by a non-Australian entity."),
                 ("Maple Bond", "A Canadian dollar-denominated bond issued in Canada by a non-Canadian entity."),
                 ("Masala Bond", "A rupee-denominated bond issued outside India by Indian entities to attract foreign investors."),
                 ("Panda Bond", "A renminbi-denominated bond issued in mainland China by foreign issuers."),
                 ("Samurai Bond", "A yen-denominated bond issued in Japan by a foreign entity, targeting Japanese investors."),
                 ("Sukuk (Islamic Bond)", "A Sharia-compliant instrument where investors receive returns linked to asset performance rather than interest."),
                 ("Yankee Bond", "A U.S. dollar-denominated bond issued in the U.S. by a non-U.S. entity."),
             ]),
            ("Based on Structure",
             [
                 ("GDP-Linked Bond", "A sovereign instrument where coupon or principal is tied to the issuer country’s GDP growth."),
                 ("Perpetual Bond (Perp)", "A bond with no maturity date that pays coupons indefinitely."),
                 ("Step-Up Bond", "A bond whose coupon increases at predetermined intervals or upon trigger events."),
             ]),
            ("Environment-Linked Bonds",
             [
                 ("Blue Bond", "A sustainability-focused bond financing marine and ocean-related projects."),
                 ("Carbon Credit Bond", "Debt tied to revenues from carbon credit markets, financing green transition projects."),
                 ("Catastrophe Bond (Cat Bond)", "A high-yield bond transferring natural disaster risk from insurers to capital markets."),
                 ("Cocoa Bond", "A commodity-linked bond where payments depend on cocoa price levels."),
                 ("Debt-for-Nature Swap Bond", "A bond structure where debt is exchanged for commitments to fund environmental conservation."),
                 ("Outcome Bond (Social Impact Bond)", "A performance-linked instrument where returns depend on achieving pre-agreed social outcomes."),
             ]),
            ("Next-Gen",
             [
                 ("Tokenized Bond", "A digital bond issued and traded on blockchain platforms for greater transparency and settlement efficiency."),
                 ("Volcano Bond", "A bitcoin-backed sovereign bond issued by El Salvador to fund geothermal energy from volcanoes."),
             ]),
        ]

        for title, items in sections:
            st.markdown(f"### {title}")
            for name, desc in sorted(items, key=lambda x: x[0].lower()):
                st.markdown(f"- **{name}**: {desc}")

if __name__ == "__main__":
    st.set_page_config(page_title="Glossary & Learn More", page_icon="📘", layout="wide")
    render()
