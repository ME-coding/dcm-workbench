from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import date

# ---------------------
# Helpers & demo data
# ---------------------

def _load_csv(file, expected_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(file)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"CSV is missing columns: {missing}. Found: {list(df.columns)}")
    return df

# 10 secteurs cibles (cohérents entre spreads et prices)
SECTOR_POOL = [
    "Banks", "Insurance", "Utilities", "Energy", "Industrials",
    "Telecoms", "Consumer Discretionary", "Consumer Staples", "Healthcare", "Technology"
]

def _demo_ibxx_spreads() -> pd.DataFrame:
    dates = pd.date_range(end=date.today(), periods=80, freq="B")
    base = {
        "Banks": 160, "Insurance": 145, "Utilities": 120, "Energy": 155, "Industrials": 130,
        "Telecoms": 125, "Consumer Discretionary": 150, "Consumer Staples": 110,
        "Healthcare": 105, "Technology": 115
    }
    rows = []
    for i, d in enumerate(dates):
        macro = 10.0*np.sin(i/22.0)
        for s in SECTOR_POOL:
            sector_drift = {"Energy": 0.06, "Banks": 0.04, "Technology": -0.03}.get(s, 0.00) * (i - len(dates)/2)
            noise = np.random.normal(0, 3.0)
            val = max(35, base[s] + macro + sector_drift + noise)
            rows.append([d.date(), s, val])
    return pd.DataFrame(rows, columns=["Date", "Sector", "Spread (bps)"])

def _demo_sector_prices(spread_df: pd.DataFrame) -> pd.DataFrame:
    if spread_df.empty:
        return pd.DataFrame(columns=["Date", "Sector", "Price"])
    df = spread_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Sector", "Date"])
    base_price = {
        "Banks": 100.0, "Insurance": 101.5, "Utilities": 103.0, "Energy": 99.5, "Industrials": 101.0,
        "Telecoms": 100.5, "Consumer Discretionary": 98.5, "Consumer Staples": 103.5,
        "Healthcare": 104.0, "Technology": 105.0
    }
    beta = {
        "Banks": -0.030, "Insurance": -0.028, "Utilities": -0.022, "Energy": -0.035, "Industrials": -0.026,
        "Telecoms": -0.022, "Consumer Discretionary": -0.032, "Consumer Staples": -0.018,
        "Healthcare": -0.016, "Technology": -0.028
    }
    out = []
    for s, g in df.groupby("Sector"):
        b = base_price.get(s, 100.0)
        bta = beta.get(s, -0.025)
        prices, eps_prev = [], 0.0
        s0 = float(g["Spread (bps)"].iloc[0])
        for _, r in g.iterrows():
            eps = 0.6*eps_prev + np.random.normal(0, 0.25)
            px = b + bta*(float(r["Spread (bps)"]) - s0) + eps
            prices.append(px)
            eps_prev = eps
        tmp = g.copy()
        tmp["Price"] = prices
        out.append(tmp[["Date", "Sector", "Price"]])
    return pd.concat(out, ignore_index=True)

# ---------------------
# Main render
# ---------------------

def render():
    st.subheader("Structuring Desk — Data Visualisation")
    st.caption("All charts use user-provided (or demo) data. No external data sources.")

    tabs = st.tabs([
        "iBoxx Spreads",
        "Sector Price Tracker",
    ])

    # -------------------------
    # Tab 1 — iBoxx Spreads
    # -------------------------
    with tabs[0]:
        st.markdown("#### iBoxx Sector Spreads")
        st.caption("Credit risk proxy per sector (bp vs gov/OIS). Lower = tighter; higher = wider.")

        left, right = st.columns([2, 1])
        with left:
            ib_file = st.file_uploader("Upload iBoxx spreads CSV (`Date, Sector, Spread (bps)`)", type=["csv"], key="ib_csv")
            if ib_file:
                ib = _load_csv(ib_file, ["Date", "Sector", "Spread (bps)"])
            else:
                if "ib_df" not in st.session_state:
                    st.session_state.ib_df = _demo_ibxx_spreads()
                ib = st.data_editor(st.session_state.ib_df, num_rows="dynamic", use_container_width=True, key="ib_editor")
                st.session_state.ib_df = ib.copy()
        with right:
            if st.button("Load demo spreads"):
                st.session_state.ib_df = _demo_ibxx_spreads()
                st.success("Demo iBoxx spreads loaded.")

            st.markdown("**Sectors to display**")
            colA, colB = st.columns(2)
            chosen_flags = {}
            for i, s in enumerate(SECTOR_POOL):
                default_on = (i < 4)
                target_col = colA if i < 5 else colB
                with target_col:
                    chosen_flags[s] = st.checkbox(s, value=default_on, key=f"ib_ck_{s}")
            chosen_spreads = [s for s, v in chosen_flags.items() if v]

            view_spreads = st.radio("View", ["Indexed (100 = first)", "Absolute (bps)"], index=1, key="ib_view")

        if not ib.empty:
            ib["Date"] = pd.to_datetime(ib["Date"])
            available = sorted(ib["Sector"].unique())
            chosen_final = [s for s in chosen_spreads if s in available] or available[:4]
            ib = ib[ib["Sector"].isin(chosen_final)].copy().sort_values(["Sector", "Date"])

            # MA10 + index 100
            ib["MA10"] = ib.groupby("Sector")["Spread (bps)"].transform(lambda s: s.rolling(10, min_periods=1).mean())
            ib["Index100"] = ib.groupby("Sector")["Spread (bps)"].transform(lambda s: 100*s/s.iloc[0])
            ib["Index100_MA10"] = ib.groupby("Sector")["MA10"].transform(lambda s: 100*s/s.iloc[0])

            # Déterminer focus / référence automatiquement avec les 2 premiers cochés
            focus_sector = chosen_final[0] if len(chosen_final) >= 1 else None
            ref_sector   = chosen_final[1] if len(chosen_final) >= 2 else None

            c1, c2 = st.columns(2)
            with c1:
                if view_spreads.startswith("Indexed"):
                    y_field, y_ma = "Index100", "Index100_MA10"
                    y_title = "Index (base 100)"
                else:
                    y_field, y_ma = "Spread (bps)", "MA10"
                    y_title = "Spread (bps)"

                line = alt.Chart(ib).mark_line().encode(
                    x="Date:T",
                    y=alt.Y(f"{y_field}:Q", title=y_title, scale=alt.Scale(zero=False)),
                    color="Sector:N",
                    tooltip=["Date","Sector", alt.Tooltip(f"{y_field}:Q", format=",.2f")],
                )
                ma = alt.Chart(ib).mark_line(strokeDash=[4,3], opacity=0.9).encode(
                    x="Date:T",
                    y=alt.Y(f"{y_ma}:Q", title="", scale=alt.Scale(zero=False)),
                    color="Sector:N",
                    tooltip=["Date","Sector", alt.Tooltip(f"{y_ma}:Q", format=",.2f")],
                )
                chart_multi = (line + ma).properties(height=300, title="Spreads by sector (with 10D MA)")
                st.altair_chart(chart_multi, use_container_width=True)

            with c2:
                chart_box = alt.Chart(ib).mark_boxplot(extent="min-max").encode(
                    x=alt.X("Sector:N"),
                    y=alt.Y("Spread (bps):Q", scale=alt.Scale(zero=False)),
                    color="Sector:N",
                ).properties(height=300, title="Distribution per sector")
                st.altair_chart(chart_box, use_container_width=True)

            # ---- INTERPRETATIONS (empilées)
            st.caption("**Spreads by sector (with 10D MA) — interpretation** — Each line tracks a sector’s credit spread. "
                       "Downward moves = tightening (risk improves); upward = widening. "
                       "The dotted 10-day MA smooths noise. Indexed view normalizes starting levels for cross-sector comparison.")
            st.caption("**Distribution per sector — interpretation** — Box = median & IQR; whiskers/outliers = tails. "
                       "Tight boxes suggest stable risk; wide boxes/outliers indicate higher volatility.")

            # ---- Différentiel focus vs ref (si au moins 2 secteurs cochés)
            if focus_sector and ref_sector:
                st.markdown("##### Sector vs Reference (spread differential)")
                try:
                    a = ib[ib["Sector"] == focus_sector][["Date","Spread (bps)"]].rename(columns={"Spread (bps)":"S1"})
                    b = ib[ib["Sector"] == ref_sector][["Date","Spread (bps)"]].rename(columns={"Spread (bps)":"S2"})
                    diff = pd.merge(a, b, on="Date", how="inner")
                    diff["Diff (bps)"] = diff["S1"] - diff["S2"]
                    chart_diff = alt.Chart(diff).mark_line().encode(
                        x="Date:T",
                        y=alt.Y("Diff (bps):Q", scale=alt.Scale(zero=False)),
                        tooltip=["Date", alt.Tooltip("Diff (bps):Q", format=",.1f")],
                    ).properties(height=240, title=f"{focus_sector} – {ref_sector} (bps)")
                    st.altair_chart(chart_diff, use_container_width=True)
                    st.caption("**Spread differential — interpretation** — Positive = focus wider than reference; "
                               "negative = tighter. Useful for relative-value monitoring. "
                               "Tip: change focus/ref by cochant/décochant pour modifier les deux premiers secteurs sélectionnés.")
                except Exception:
                    st.info("Need overlapping dates for the two selected sectors to display the differential.")
            else:
                st.info("Select at least two sectors to display the focus–reference differential.")

    # -------------------------
    # Tab 2 — Sector Price Tracker
    # -------------------------
    with tabs[1]:
        st.markdown("#### Sector Price Tracker")
        st.caption("Synthetic sector ‘price’ proxy built from spreads: lower spreads → higher prices (demo).")

        left, right = st.columns([2, 1])
        with left:
            pr_file = st.file_uploader("Upload sector prices CSV (`Date, Sector, Price`)", type=["csv"], key="pr_csv")
            if pr_file:
                pr = _load_csv(pr_file, ["Date", "Sector", "Price"])
            else:
                if "ib_df" not in st.session_state:
                    st.session_state.ib_df = _demo_ibxx_spreads()
                demo = _demo_sector_prices(st.session_state.ib_df)
                if "pr_df" not in st.session_state:
                    st.session_state.pr_df = demo
                pr = st.data_editor(st.session_state.pr_df, num_rows="dynamic", use_container_width=True, key="pr_editor")
                st.session_state.pr_df = pr.copy()
        with right:
            if st.button("Load demo prices"):
                if "ib_df" not in st.session_state:
                    st.session_state.ib_df = _demo_ibxx_spreads()
                st.session_state.pr_df = _demo_sector_prices(st.session_state.ib_df)
                st.success("Demo prices loaded.")

            st.markdown("**Sectors to display**")
            colA, colB = st.columns(2)
            chosen_flags = {}
            for i, s in enumerate(SECTOR_POOL):
                default_on = (i < 3)
                target_col = colA if i < 5 else colB
                with target_col:
                    chosen_flags[s] = st.checkbox(s, value=default_on, key=f"px_ck_{s}")
            chosen = [s for s, v in chosen_flags.items() if v]

            view_mode = st.radio("View", ["Indexed (100 = first)", "Absolute price"], index=0)

        if not pr.empty:
            pr["Date"] = pd.to_datetime(pr["Date"])
            available = sorted(pr["Sector"].unique())
            chosen_final = [s for s in chosen if s in available] or available[:3]

            pr_sorted = pr.sort_values(["Sector", "Date"]).copy()
            pr_sorted["MA10"] = pr_sorted.groupby("Sector")["Price"].transform(lambda s: s.rolling(10, min_periods=1).mean())
            pr_sorted["Index100"] = pr_sorted.groupby("Sector")["Price"].transform(lambda s: 100*s/s.iloc[0])
            pr_sorted["Index100_MA10"] = pr_sorted.groupby("Sector")["MA10"].transform(lambda s: 100*s/s.iloc[0])

            show = pr_sorted[pr_sorted["Sector"].isin(chosen_final)]

            c1, c2 = st.columns(2)
            with c1:
                if view_mode.startswith("Indexed"):
                    y_field, y_ma = "Index100", "Index100_MA10"
                    y_title = "Index (base 100)"
                else:
                    y_field, y_ma = "Price", "MA10"
                    y_title = "Price"

                line = alt.Chart(show).mark_line().encode(
                    x="Date:T",
                    y=alt.Y(f"{y_field}:Q", title=y_title, scale=alt.Scale(zero=False)),
                    color="Sector:N",
                    tooltip=["Date","Sector", alt.Tooltip(f"{y_field}:Q", format=",.2f")],
                )
                ma = alt.Chart(show).mark_line(strokeDash=[4,3], opacity=0.9).encode(
                    x="Date:T",
                    y=alt.Y(f"{y_ma}:Q", title="", scale=alt.Scale(zero=False)),
                    color="Sector:N",
                    tooltip=["Date","Sector", alt.Tooltip(f"{y_ma}:Q", format=",.2f")],
                )
                chart_price = (line + ma).properties(height=320, title=f"Sector {y_title.lower()} (with 10D moving average)")
                st.altair_chart(chart_price, use_container_width=True)
                st.caption("**Sector price — interpretation** — Prices rise when spreads tighten (risk improves) and fall when spreads widen. "
                           "The 10-day MA highlights the trend; the Index-100 view normalizes starting levels to compare sectors.")

            with c2:
                if "ib_df" in st.session_state and not st.session_state.ib_df.empty:
                    ib = st.session_state.ib_df.copy()
                    ib["Date"] = pd.to_datetime(ib["Date"])
                    merged = pd.merge(show, ib, on=["Date","Sector"], how="inner")
                    merged = merged[merged["Sector"].isin(chosen_final)]
                    if not merged.empty:
                        base_scatter = alt.Chart(merged).encode(
                            x=alt.X("Spread (bps):Q", title="Spread (bps)", scale=alt.Scale(zero=False)),
                            y=alt.Y("Price:Q", title="Price", scale=alt.Scale(zero=False)),
                            color="Sector:N",
                            tooltip=["Date","Sector",
                                     alt.Tooltip("Spread (bps):Q", format=",.0f"),
                                     alt.Tooltip("Price:Q", format=",.2f")],
                        )
                        sc = base_scatter.mark_circle(size=55, opacity=0.7)
                        tl = base_scatter.transform_regression("Spread (bps)", "Price", groupby=["Sector"]).mark_line(opacity=0.9)
                        st.altair_chart((sc + tl).properties(height=320, title="Price vs Spread (by sector, demo relationship)"),
                                        use_container_width=True)
                        st.caption("**Price vs Spread — interpretation** — Expect a negative slope: higher spreads → lower price. "
                                   "The fitted line per sector gives a quick beta (sensitivity).")
                    else:
                        st.info("No overlapping dates between price and spread data.")
                else:
                    st.info("Load spreads in the previous tab to enable the Price vs Spread scatter.")
