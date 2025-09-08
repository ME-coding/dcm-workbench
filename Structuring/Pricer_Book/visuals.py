# Page1/Pricer_Book/visuals.py
from __future__ import annotations
import pandas as pd
import numpy as np
import altair as alt

def price_yield_chart(y_grid_pct: np.ndarray, prices_per_100: np.ndarray, title: str = "Price–Yield Curve"):
    df = pd.DataFrame({"Yield (%)": y_grid_pct, "Clean Price (per 100)": prices_per_100})
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("Yield (%):Q", title="Yield (%)"),
            y=alt.Y("Clean Price (per 100):Q", title="Clean Price (per 100)"),
            tooltip=["Yield (%)", "Clean Price (per 100)"],
        )
        .properties(title=title, height=260)
    )

def cashflow_breakdown_chart(schedule: pd.DataFrame, title: str = "Cash Flow Breakdown"):
    cf_df = schedule[["Time (years)", "Coupon/Profit", "Principal"]].melt(
        id_vars=["Time (years)"], var_name="Component", value_name="Amount"
    )
    return (
        alt.Chart(cf_df)
        .mark_bar()
        .encode(
            x=alt.X("Time (years):Q", title="Time (years)"),
            y=alt.Y("Amount:Q", title="Cash Flow"),
            color="Component:N",
            tooltip=["Time (years)", "Component", "Amount"],
        )
        .properties(title=title, height=260)
    )

def amortization_chart(schedule: pd.DataFrame, title: str = "Amortization Profile (Outstanding)"):
    df = schedule[["Time (years)", "Outstanding (end)"]].copy()
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("Time (years):Q", title="Time (years)"),
            y=alt.Y("Outstanding (end):Q", title="Outstanding (end)"),
            tooltip=["Time (years)", "Outstanding (end)"],
        )
        .properties(title=title, height=260)
    )

def rate_path_chart(times_years: np.ndarray, per_period_rates_pct: np.ndarray, title: str = "Reference / Profit Rate Path"):
    df = pd.DataFrame({"Time (years)": times_years, "Rate (%)": per_period_rates_pct})
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Time (years):Q", title="Time (years)"),
            y=alt.Y("Rate (%):Q", title="Rate (%)"),
            tooltip=["Time (years)", "Rate (%)"],
        )
        .properties(title=title, height=260)
    )

__all__ = [
    "price_yield_chart",
    "cashflow_breakdown_chart",
    "amortization_chart",
    "rate_path_chart",
]

print("visuals loaded, dir():", dir())
