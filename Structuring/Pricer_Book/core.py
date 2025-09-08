from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np
import pandas as pd

# =========================
# Core utilities (extraits de pricer.py, noms conservés)
# =========================

def build_schedule_fixed(
    notional: float,
    coupon_rate: float,
    freq: int,
    years: float,
    structure: str = "bullet",
) -> pd.DataFrame:
    n = max(1, int(round(freq * years)))
    per = 1 / freq
    r_per = coupon_rate / freq
    data, outstanding = [], notional

    if structure == "bullet":
        coupon_amount = notional * r_per
        for i in range(1, n + 1):
            principal_payment = notional if i == n else 0.0
            coupon_payment = coupon_amount
            total = coupon_payment + principal_payment
            outstanding_end = outstanding - principal_payment
            data.append([i, i * per, outstanding, coupon_payment, principal_payment, total, outstanding_end])
            outstanding = outstanding_end

    elif structure == "equal_principal":
        principal_payment_const = notional / n
        for i in range(1, n + 1):
            interest_payment = outstanding * r_per
            principal_payment = principal_payment_const
            total = interest_payment + principal_payment
            outstanding_end = outstanding - principal_payment
            data.append([i, i * per, outstanding, interest_payment, principal_payment, total, outstanding_end])
            outstanding = outstanding_end

    elif structure == "annuity":
        payment = notional / n if r_per == 0 else notional * (r_per) / (1 - (1 + r_per) ** (-n))
        for i in range(1, n + 1):
            interest_payment = outstanding * r_per
            principal_payment = payment - interest_payment
            total = payment
            outstanding_end = outstanding - principal_payment
            data.append([i, i * per, outstanding, interest_payment, principal_payment, total, outstanding_end])
            outstanding = outstanding_end

    df = pd.DataFrame(
        data,
        columns=["Period","Time (years)","Outstanding (begin)","Coupon/Profit","Principal","Total CF","Outstanding (end)"],
    )
    return df

def build_schedule_variable(
    notional: float,
    rates_annual: List[float],
    freq: int,
    structure: str = "bullet",
) -> pd.DataFrame:
    n = max(1, len(rates_annual))
    per = 1 / freq
    data, outstanding = [], notional

    if structure == "bullet":
        for i in range(1, n + 1):
            r_per = rates_annual[i - 1] / freq
            coupon_payment = outstanding * r_per
            principal_payment = notional if i == n else 0.0
            total = coupon_payment + principal_payment
            outstanding_end = outstanding - principal_payment
            data.append([i, i * per, outstanding, coupon_payment, principal_payment, total, outstanding_end])
            outstanding = outstanding_end

    elif structure == "equal_principal":
        principal_payment_const = notional / n
        for i in range(1, n + 1):
            r_per = rates_annual[i - 1] / freq
            interest_payment = outstanding * r_per
            principal_payment = principal_payment_const
            total = interest_payment + principal_payment
            outstanding_end = outstanding - principal_payment
            data.append([i, i * per, outstanding, interest_payment, principal_payment, total, outstanding_end])
            outstanding = outstanding_end

    df = pd.DataFrame(
        data,
        columns=["Period","Time (years)","Outstanding (begin)","Coupon/Profit","Principal","Total CF","Outstanding (end)"],
    )
    return df

def present_value(cashflows: pd.Series, ytm_annual: float, freq: int) -> float:
    y = max(1e-12, ytm_annual / freq)
    disc = np.array([(1 + y) ** i for i in range(1, len(cashflows) + 1)])
    return float(np.sum(cashflows.values / disc))

def price_from_yield(
    schedule: pd.DataFrame,
    ytm_annual: float,
    freq: int,
    notional: float,
    accrued_frac: float = 0.0
) -> Tuple[float, float, float]:
    pv = present_value(schedule["Total CF"], ytm_annual, freq)
    scale = 100.0 / notional
    clean_price = pv * scale
    try:
        # ✅ correction: pas de parenthèse en trop après ] et ternary complet
        denom = schedule.loc[0, "Outstanding (begin)"]
        r_per_inferred = (
            (schedule.loc[0, "Coupon/Profit"] / denom) if denom > 0 else 0.0
        )
    except Exception:
        r_per_inferred = 0.0

    current_outstanding = schedule.loc[0, "Outstanding (begin)"] if len(schedule) > 0 else notional
    accrued = (current_outstanding * r_per_inferred * accrued_frac) * scale
    return float(clean_price), float(accrued), float(clean_price + accrued)

def yield_from_price(schedule: pd.DataFrame, target_clean_price_per_100: float, freq: int, notional: float, tol: float = 1e-8, max_iter: int = 200) -> float:
    target_pv = target_clean_price_per_100 * notional / 100.0
    low, high = -0.99, 1.5
    mid = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        pv = present_value(schedule["Total CF"], mid, freq)
        if abs(pv - target_pv) < tol:
            return mid
        if pv > target_pv:
            low = mid
        else:
            high = mid
    return mid

def macaulay_duration_convexity(schedule: pd.DataFrame, ytm_annual: float, freq: int):
    y = max(1e-12, ytm_annual / freq)
    cfs = schedule["Total CF"].values
    disc = np.array([(1 + y) ** i for i in range(1, len(cfs) + 1)])
    pv_cf = cfs / disc
    price = float(np.sum(pv_cf))
    t = np.arange(1, len(cfs) + 1)
    mac_years = (np.sum(t * pv_cf) / price) / freq if price > 0 else 0.0
    mod_duration = mac_years / (1 + y)
    conv_years2 = (np.sum(cfs * t * (t + 1) / ((1 + y) ** (t + 2))) / price) / (freq ** 2) if price > 0 else 0.0
    return price, mac_years, mod_duration, conv_years2

# ---------- Call/Put helper ----------
def truncate_with_redemption(schedule_base: pd.DataFrame, t_years: float, freq: int, redemption_cash: float) -> pd.DataFrame:
    per = 1 / freq
    k = min(len(schedule_base), int(math.floor(t_years / per)))
    if k < 1:
        return pd.DataFrame(columns=schedule_base.columns)
    df = schedule_base.copy().iloc[:k].reset_index(drop=True)
    df["Principal"] = 0.0
    df.loc[k - 1, "Principal"] = redemption_cash
    df["Total CF"] = df["Coupon/Profit"] + df["Principal"]
    return df

# --- API explicit ---
__all__ = [
    "build_schedule_fixed",
    "build_schedule_variable",
    "present_value",
    "price_from_yield",
    "yield_from_price",
    "macaulay_duration_convexity",
    "truncate_with_redemption",
]
