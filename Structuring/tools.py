from __future__ import annotations

from datetime import date
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

def tool_termsheet():
    st.markdown("#### Term Sheet Builder")

    c1, c2, c3 = st.columns(3)
    with c1:
        issuer = st.text_input("Issuer", value="Sample Issuer SA", key="ts_issuer")
        guarantor = st.text_input("Guarantor (if any)", value="", key="ts_guarantor")
        currency = st.text_input("Currency", value="EUR", key="ts_currency")
        format_ = st.selectbox("Format", sorted(["Reg S", "144A/Reg S", "Domestic"]), index=1, key="ts_format")
        status = st.selectbox(
            "Status",
            sorted(["Senior Unsecured", "Senior Preferred", "Senior Non-Preferred", "Subordinated", "Tier 2", "AT1"]),
            key="ts_status",
        )
    with c2:
        notional = st.number_input("Notional", min_value=100000.0, value=500_000_000.0,
                                   step=1_000_000.0, format="%.2f", key="ts_notional")
        issue_px = st.number_input("Issue price (% of par)", min_value=0.0, value=99.50,
                                   step=0.10, format="%.2f", key="ts_issue_px")
        redemption_px = st.number_input("Redemption price (% of par)", min_value=0.0, value=100.00,
                                        step=0.10, format="%.2f", key="ts_redemption_px")
        coupon_type = st.selectbox("Coupon type", sorted(["Fixed", "Floating (FRN)", "Zero-Coupon"]), key="ts_coupon_type")
        coupon = st.text_input("Coupon / Ref + spread", value="4.000% annual (Act/Act, semi-annual)", key="ts_coupon")
    with c3:
        maturity = st.date_input(
            "Maturity date",
            value=date.today().replace(year=date.today().year + 5),
            key="ts_maturity",
        )
        denom = st.text_input("Denomination", value="€100,000 + €100,000", key="ts_denom")
        listing = st.text_input("Listing (if any)", value="LuxSE", key="ts_listing")
        law = st.text_input("Governing law", value="English law", key="ts_law")
        use = st.text_area("Use of proceeds", value="General corporate purposes.", key="ts_use")
        sustainability = st.checkbox("Sustainability-linked features", value=False, key="ts_sust")

    covenants = st.text_area(
        "Covenants / Optionality (summary)",
        value="Change of Control put @ 101%. Make-whole call prior to Maturity. Standard negative pledge.",
        key="ts_cov",
    )

    # Build the markdown
    lines = []
    lines.append(f"# Indicative Terms & Conditions")
    lines.append("")
    lines.append(f"**Issuer:** {issuer}")
    if guarantor.strip():
        lines.append(f"**Guarantor:** {guarantor}")
    lines.append(f"**Currency / Notional:** {currency} {notional:,.0f}")
    lines.append(f"**Format:** {format_}")
    lines.append(f"**Status:** {status}")
    lines.append(f"**Maturity:** {maturity.strftime('%d %b %Y')}")
    lines.append(f"**Issue / Redemption Price:** {issue_px:.2f}% / {redemption_px:.2f}% of par")
    lines.append(f"**Coupon:** {coupon_type} — {coupon}")
    lines.append(f"**Denomination:** {denom}")
    lines.append(f"**Listing:** {listing}")
    lines.append(f"**Governing Law:** {law}")
    if sustainability:
        lines.append("**Sustainability:** Applicable (see KPI framework and step up mechanics).")
    lines.append(f"**Use of Proceeds:** {use}")
    lines.append(f"**Covenants / Optionality:** {covenants}")
    lines.append("")
    lines.append("_This term sheet is for discussion purposes only and does not constitute an offer or solicitation._")

    # Editable area + copy-friendly code block (keeps Streamlit's native 'Copy' button)
    st.markdown("---")
    ts_text = st.text_area("Generated Term Sheet (editable)", value="\n".join(lines), height=350, key="ts_textarea")
    st.code(ts_text, language="markdown")


# =========================
# TOOL — Fees & Net Proceeds (simplified, with explanation + clearer chart)
# =========================

def _waterfall_data(gross: float, fees_total: float, other_costs: float) -> pd.DataFrame:
    """Prepare a true waterfall dataset with Start/End for each step."""
    steps = [
        {"Step": "Gross proceeds", "Type": "Base", "Amount": gross},
        {"Step": "Fees (total)", "Type": "Decrease", "Amount": -fees_total},
        {"Step": "Other costs", "Type": "Decrease", "Amount": -other_costs},
    ]
    df = pd.DataFrame(steps)
    df["Cumulative"] = df["Amount"].cumsum()
    df["Start"] = df["Cumulative"] - df["Amount"]
    df["End"] = df["Cumulative"]

    # Terminal "Net" bar
    net = gross - fees_total - other_costs
    df_net = pd.DataFrame([{
        "Step": "Net proceeds", "Type": "Result",
        "Amount": net, "Cumulative": net, "Start": 0.0, "End": net
    }])
    return pd.concat([df, df_net], ignore_index=True)


def tool_fees():
    st.markdown("#### Fees & Net Proceeds (simplified)")

    # Short explanation (why this tool & how to read the chart)
    st.caption(
        "This tool estimates **net proceeds** after underwriting fees and fixed costs. "
        "The **waterfall** shows how Gross proceeds step down to Net: each red bar is a deduction; "
        "the green bar is the final Net amount."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        notional = st.number_input("Notional", min_value=100000.0, value=500_000_000.0,
                                   step=1_000_000.0, format="%.2f", key="fees_notional")
    with c2:
        issue_px = st.number_input("Issue price (% of par)", min_value=0.0, value=99.50,
                                   step=0.10, format="%.2f", key="fees_issue_px")
    with c3:
        total_bps = st.number_input("Total fees (bps)", min_value=0.0, value=30.0,
                                    step=0.5, format="%.1f", key="fees_total_bps")
    with c4:
        other_costs = st.number_input("Other costs (fixed)", min_value=0.0, value=250_000.0,
                                      step=50_000.0, format="%.2f", key="fees_other_costs")

    gross = notional * issue_px / 100.0
    fees_amount = notional * (total_bps / 10000.0)
    net = gross - fees_amount - other_costs

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Gross proceeds", f"{gross:,.2f}")
    with k2:
        st.metric("Total fees", f"{fees_amount:,.2f}", delta=f"{total_bps:.1f} bps")
    with k3:
        st.metric("Net proceeds", f"{net:,.2f}")

    # Waterfall chart (cleaner & annotated)
    wf_df = _waterfall_data(gross, fees_amount, other_costs)

    color_scale = alt.Scale(domain=["Base", "Decrease", "Result"],
                            range=["#6B7280", "#EF4444", "#10B981"])

    bars = alt.Chart(wf_df).mark_bar().encode(
        x=alt.X("Step:N", sort=None, title=""),
        y=alt.Y("Start:Q", title="Amount"),
        y2="End:Q",
        color=alt.Color("Type:N", scale=color_scale, legend=alt.Legend(title="")),
        tooltip=[
            alt.Tooltip("Step:N"),
            alt.Tooltip("Amount:Q", format=",.2f"),
            alt.Tooltip("End:Q", title="Post-step", format=",.2f"),
        ],
    )

    labels = alt.Chart(wf_df).mark_text(dy=-6).encode(
        x=alt.X("Step:N", sort=None),
        y=alt.Y("End:Q"),
        text=alt.Text("End:Q", format=",.0f"),
        color=alt.value("#111827"),
    )

    st.altair_chart((bars + labels).properties(height=340, title="Gross → Net Proceeds: Waterfall"), use_container_width=True)

    # CSV export (kept for quick sharing / audit)
    out_df = pd.DataFrame({
        "Component": ["Gross proceeds", "Total fees", "Other costs", "Net proceeds"],
        "Amount": [gross, fees_amount, other_costs, net],
    })
    st.download_button("Download fees breakdown (CSV)", data=out_df.to_csv(index=False).encode("utf-8"),
                       file_name="fees_breakdown.csv", mime="text/csv", key="fees_dl_csv")


# =========================
# TOOL 3 — Amortization Builder (improved chart)
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

    # ---- Improved chart: Stacked bars (Principal + Interest) + line (Outstanding end)
    bars_df = df.melt(id_vars=["Period", "Time (years)", "Outstanding (end)"],
                      value_vars=["Principal", "Interest/Coupon"],
                      var_name="Component", value_name="Amount")

    bar = alt.Chart(bars_df).mark_bar().encode(
        x=alt.X("Time (years):Q", title="Time (years)"),
        y=alt.Y("Amount:Q", stack="zero", title="Cashflow"),
        color=alt.Color("Component:N", sort=["Interest/Coupon", "Principal"]),
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

    chart = alt.layer(bar, line).resolve_scale(y="independent").properties(
        height=340,
        title="Cashflows (stacked) & Outstanding profile"
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
        "Term Sheet Builder",
        "Fees & Net Proceeds",
        "Amortization Builder",
    ])

    with tabs[0]:
        tool_termsheet()
    with tabs[1]:
        tool_fees()
    with tabs[2]:
        tool_amortization()
