from __future__ import annotations
from .core import (
    build_schedule_fixed,
    build_schedule_variable,
    present_value,
    price_from_yield,
    yield_from_price,
    macaulay_duration_convexity,
    truncate_with_redemption,
)

# Pricer_Book/convertible_bond.py

from dataclasses import dataclass
import numpy as np
from numpy.polynomial import laguerre as lag
import matplotlib.pyplot as plt
import streamlit as st
import pathlib

from .visuals import (
    price_yield_chart,
    cashflow_breakdown_chart,
    amortization_chart,
    rate_path_chart,
)
import math

# =============================
# Parameters & Simulation
# =============================

@dataclass
class ConvertibleBondParams:
    T: float = 2.0            # maturité (années)
    steps: int = 2*252        # pas de temps
    S0: float = 100.0         # spot
    r: float = 0.05           # taux sans risque
    sigma: float = 0.40       # volatilité
    q: float = 0.10           # dividend yield
    face: float = 100.0       # nominal
    N: float = 1.0            # ratio de conversion
    laguerre_deg: int = 6     # degré max série de Laguerre
    paths: int = 10_000       # nb de trajectoires
    seed: int | None = 42     # reproductibilité

    @property
    def conversion_price(self) -> float:
        return self.face / self.N

def simulate_paths(p: ConvertibleBondParams) -> np.ndarray:
    if p.seed is not None:
        np.random.seed(p.seed)
    dt = p.T / p.steps
    S = np.full((p.paths, p.steps), p.S0, dtype=float)
    z = np.random.normal(0.0, 1.0, size=(p.paths, p.steps-1))
    drift = (p.r - p.q) * dt
    vol   = p.sigma * np.sqrt(dt)
    for t in range(1, p.steps):
        S[:, t] = S[:, t-1] + drift * S[:, t-1] + vol * S[:, t-1] * z[:, t-1]
    return S

def _cashflow_maturity(N: float, S_T: np.ndarray, face: float) -> np.ndarray:
    return np.maximum(N * S_T, face)

# =============================
# Pricing LSMC
# =============================

def price_lsmc(p: ConvertibleBondParams, return_all=False):
    S = simulate_paths(p)
    time = np.linspace(0.0, p.T, p.steps)
    CB = _cashflow_maturity(p.N, S[:, -1], p.face)
    P = p.conversion_price

    for t in range(p.steps - 1, 0, -1):
        dt = time[t] - time[t-1]
        df = np.exp(-p.r * dt)

        St = S[:, t]
        itm = St > P
        x = St[itm]
        y = df * CB[itm]
        if x.size >= 2:
            deg = min(p.laguerre_deg, max(1, x.size - 1))
            coefs = lag.lagfit(x, y, deg=deg)
            cont = lag.lagval(St, coefs)
        else:
            cont = np.full_like(St, df * np.mean(CB))

        immediate = p.N * St
        exercise = itm & (immediate > cont)

        CB = df * CB
        CB[exercise] = immediate[exercise]

    price = float(np.mean(CB))
    std_err = float(np.std(CB, ddof=1) / np.sqrt(p.paths))
    if return_all:
        return price, std_err, {"CB_paths_value": CB, "S": S, "time": time, "conv_price": P}
    return price, std_err

# =============================
# Graphs
# =============================

def plot_underlying_paths(S: np.ndarray, time: np.ndarray):
    plt.figure(figsize=(12,7))
    for i in range(min(50, S.shape[0])):
        plt.plot(time, S[i], lw=1)
    plt.xlabel("Time (Years)"); plt.ylabel("Price S(t)")
    plt.title("Underlying Paths (GBM, sample)")
    plt.grid(True); plt.tight_layout()
    return plt.gcf()

def plot_itm_colouring(S: np.ndarray, time: np.ndarray, conv_price: float, many: bool = False):
    plt.figure(figsize=(12,7))
    if many:
        for path in S[:min(50, S.shape[0])]:
            colors = ['teal' if s > conv_price else 'orangered' for s in path]
            plt.plot(time, path, color="gray", lw=1)
            plt.scatter(time, path, c=colors, s=15, alpha=0.5)
        title = "Multiple Paths | Teal=ITM, Orange=OTM"
    else:
        path = S[0]
        colors = ['teal' if s > conv_price else 'orangered' for s in path]
        plt.plot(time, path, color="gray", lw=1.5)
        plt.scatter(time, path, c=colors, s=25, alpha=0.8)
        title = "One Path | Teal=ITM, Orange=OTM"
    plt.xlabel("Time (Years)"); plt.ylabel("Price S(t)")
    plt.title(title); plt.grid(True); plt.tight_layout()
    return plt.gcf()

def plot_laguerre_fit(S: np.ndarray, time: np.ndarray, p: ConvertibleBondParams):
    Stn   = S[:, -1]
    Stnm1 = S[:, -2]
    dt = time[-1] - time[-2]
    df = np.exp(-p.r * dt)
    disc_cf = df * _cashflow_maturity(p.N, Stn, p.face)
    deg = min(p.laguerre_deg, max(1, S.shape[0] - 1))
    coefs = lag.lagfit(Stnm1, disc_cf, deg=deg)
    xs = np.linspace(Stnm1.min(), Stnm1.max(), 200)
    fitted = lag.lagval(xs, coefs)
    plt.figure(figsize=(12,7))
    plt.scatter(Stnm1, disc_cf, label="Discounted Cashflows")
    plt.plot(xs, fitted, label=f"Laguerre Fit (deg={deg})")
    plt.title("Discounted Cashflows vs Underlying Price")
    plt.xlabel("S(t-1)"); plt.ylabel("Discounted CFs")
    plt.legend(); plt.grid(True); plt.tight_layout()
    return plt.gcf()

# =============================
# UI in Streamlit
# =============================

def render():
    st.subheader("Convertible Bond (LSMC Pricer)")
    col1, col2 = st.columns(2)
    with col1:
        p = ConvertibleBondParams(
            T=st.number_input("Maturity T (years)", 0.25, 30.0, 2.0, 0.25),
            steps=st.number_input("Time steps", 25, 5000, 504, 1),
            S0=st.number_input("Spot S₀", 0.01, 10000.0, 100.0, 1.0),
            r=st.number_input("Risk-free rate r", -0.05, 0.25, 0.05, 0.005),
            sigma=st.number_input("Volatility σ", 0.01, 3.0, 0.40, 0.01),
            q=st.number_input("Dividend yield q", 0.0, 1.0, 0.10, 0.005),
            face=st.number_input("Face value", 0.01, 1e7, 100.0, 1.0),
            N=st.number_input("Conversion ratio N", 0.0001, 1e6, 1.0, 0.1),
            laguerre_deg=st.slider("Laguerre degree", 1, 12, 6),
            paths=st.number_input("Monte Carlo paths", 100, 200_000, 10_000, 100),
            seed=st.number_input("Seed (None = random)", value=42)
        )
        st.caption(f"Conversion price P = Face/N = **{p.conversion_price:.2f}**")

    with col2:
        if st.button("Price"):
            price, se = price_lsmc(p)
            st.metric("Convertible Bond Price", f"${price:,.2f}", help=f"MC std. error ≈ {se:.4f}")

    if st.checkbox("Show Graphs", value=True):
        p_vis = ConvertibleBondParams(**{**p.__dict__, "paths": min(200, p.paths), "steps": min(200, p.steps)})
        S = simulate_paths(p_vis)
        time = np.linspace(0.0, p_vis.T, p_vis.steps)
        st.pyplot(plot_underlying_paths(S, time))
        st.pyplot(plot_itm_colouring(S, time, p_vis.conversion_price, many=False))
        st.pyplot(plot_itm_colouring(S, time, p_vis.conversion_price, many=True))
        st.pyplot(plot_laguerre_fit(S, time, p_vis))

    # Theory & buttons
    st.markdown("### Theory & Resources")
    st.write("A **Convertible Bond** is a fixed-income instrument that gives the holder the right "
             "to convert the bond into a predetermined number of shares of the issuing company. "
             "It combines features of debt (fixed coupon, redemption) and equity (conversion option).")

    # --- Boutons verticaux ---
    st.link_button("Learn More", "https://en.wikipedia.org/wiki/Convertible_bond")

    example_path = pathlib.Path("Library") / "Convertible Bond Example - Hoffmann-Green OCEANE (2024, by Portzamparc BNPP).pdf"
    if example_path.exists():
        st.download_button(
            "Download: Convertible Bond Example - Hoffmann-Green OCEANE (2024, by Portzamparc BNPP) (PDF)",
            data=example_path.read_bytes(),
            file_name=example_path.name,
            mime="application/pdf",
        )
    else:
        st.info("Place the Hoffmann-Green OCEANE (2024) example PDF in `Library/` to enable this button.")

    if st.button("How do we calculate it?"):
        st.markdown("""
        **Pricing Methodology (LSMC with Laguerre Regression):**
        1. Simulate many stock price paths with a Geometric Brownian Motion.
        2. At maturity, payoff = max(N×S, Face).
        3. Work backwards: at each date, compare immediate conversion (N×S) to continuation value.
        4. Continuation value estimated by regression of discounted future cashflows on Laguerre polynomials of stock price.
        5. If immediate > continuation → convert; else hold.
        6. Price = average of all simulated discounted payoffs.
        """)

convertible_bond_ui = render
