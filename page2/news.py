# page2/news.py
from __future__ import annotations

import io
import os
import re
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------- Optional libs ----------
try:
    import feedparser  # pip install feedparser
except Exception:
    feedparser = None

try:
    import yfinance as yf  # pip install yfinance
except Exception:
    yf = None

try:
    import requests  # pip install requests
except Exception:
    requests = None


# =======================================================
# Helpers & cache
# =======================================================

@st.cache_data(ttl=900)
def fetch_rss(urls: List[str], limit_per_feed: int = 10) -> pd.DataFrame:
    rows = []
    if not feedparser or not isinstance(urls, list) or len(urls) == 0:
        return pd.DataFrame(columns=["source", "title", "link", "published"])
    for u in urls:
        if not isinstance(u, str) or not u.strip():
            continue
        try:
            feed = feedparser.parse(u)
            src_title = None
            try:
                src_title = getattr(feed, "feed", None)
                if src_title and hasattr(src_title, "title"):
                    src_title = src_title.title
                elif isinstance(src_title, dict):
                    src_title = src_title.get("title")
            except Exception:
                src_title = None
            src = (src_title or u.split("//")[-1]).strip()

            entries = getattr(feed, "entries", [])[: int(limit_per_feed)]
            for e in entries:
                title = str(getattr(e, "title", "")).strip()
                link = str(getattr(e, "link", "")).strip()
                if hasattr(e, "published_parsed") and e.published_parsed:
                    ts = datetime(*e.published_parsed[:6])
                elif hasattr(e, "updated_parsed") and e.updated_parsed:
                    ts = datetime(*e.updated_parsed[:6])
                else:
                    ts = datetime.utcnow()
                rows.append({"source": src, "title": title, "link": link, "published": ts})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
        df = df.sort_values("published", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=900)
def fetch_prices(symbols: Dict[str, str], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if yf is None or not isinstance(symbols, dict) or len(symbols) == 0:
        return demo_prices(symbols)
    try:
        for name, ticker in symbols.items():
            try:
                hist = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
                if hist is None or getattr(hist, "empty", True):
                    raise RuntimeError("empty")
                hist = hist.dropna()
                out[name] = hist
            except Exception:
                out[name] = demo_series(name)
        return out
    except Exception:
        return demo_prices(symbols)


def demo_series(name: str) -> pd.DataFrame:
    idx = pd.date_range(end=datetime.utcnow().date(), periods=252, freq="B")
    base = 3.0 if any(tag in name for tag in ["US", "DE", "FR", "UK", "IT", "ES"]) else 1.10
    vals = base + 0.3 * np.sin(np.linspace(0, 8, len(idx))) + np.random.normal(0, 0.05, len(idx))
    df = pd.DataFrame({"Close": vals}, index=idx)
    return df


def demo_prices(symbols: Dict[str, str] | None) -> Dict[str, pd.DataFrame]:
    symbols = symbols or {}
    return {k: demo_series(k) for k in symbols.keys()}


def pct_change(latest: float, prev: float) -> float:
    if prev == 0:
        return 0.0
    return (latest - prev) / prev * 100.0


# =======================================================
# Policy rates (Fed / ECB / BoE)
# =======================================================

@st.cache_data(ttl=3600)
def fetch_policy_rates() -> pd.DataFrame:
    rows: List[Dict[str, str]] = []

    def fred_last(series_id: str):
        if requests is None:
            return None, None
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            csv = requests.get(url, timeout=10)
            csv.raise_for_status()
            df = pd.read_csv(io.StringIO(csv.text))
            vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            last_valid = vals.last_valid_index()
            if last_valid is None:
                return None, None
            last_val = float(vals.iloc[last_valid])
            date = pd.to_datetime(df.iloc[last_valid, 0], errors="coerce", utc=True)
            return last_val, pd.to_datetime(date)
        except Exception:
            return None, None

    up, up_date = fred_last("DFEDTARU")
    lo, lo_date = fred_last("DFEDTARL")
    asof = up_date or lo_date
    if up is not None and lo is not None:
        rows.append({
            "Authority": "Federal Reserve (FOMC)",
            "Measure": "Fed Funds Target Range",
            "Latest": f"{lo:.2f}–{up:.2f}%",
            "AsOf": asof.tz_convert("Europe/Paris").strftime("%Y-%m-%d") if isinstance(asof, pd.Timestamp) else "",
        })

    mro, mro_date = fred_last("ECBMRRFR")
    dfr, dfr_date = fred_last("ECBDFR")
    if mro is not None:
        rows.append({
            "Authority": "European Central Bank",
            "Measure": "Main Refinancing Rate (MRO)",
            "Latest": f"{mro:.2f}%",
            "AsOf": mro_date.tz_convert("Europe/Paris").strftime("%Y-%m-%d") if isinstance(mro_date, pd.Timestamp) else "",
        })
    if dfr is not None:
        rows.append({
            "Authority": "European Central Bank",
            "Measure": "Deposit Facility Rate (DFR)",
            "Latest": f"{dfr:.2f}%",
            "AsOf": dfr_date.tz_convert("Europe/Paris").strftime("%Y-%m-%d") if isinstance(dfr_date, pd.Timestamp) else "",
        })

    if requests is not None:
        try:
            resp = requests.get("https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp", timeout=10)
            txt = resp.text
            m = re.search(r"Current\s+official\s+Bank\s+Rate.*?([0-9]+(?:\.[0-9]+)?)\s*%", txt, re.IGNORECASE | re.DOTALL)
            if m:
                boe_rate = float(m.group(1))
                rows.append({
                    "Authority": "Bank of England",
                    "Measure": "Official Bank Rate",
                    "Latest": f"{boe_rate:.2f}%",
                    "AsOf": datetime.utcnow().astimezone().strftime("%Y-%m-%d"),
                })
        except Exception:
            pass

    return pd.DataFrame(rows, columns=["Authority", "Measure", "Latest", "AsOf"])


# =======================================================
# Mistral summary (HTTP, no SDK)
# =======================================================

def _mistral_key() -> str:
    k = st.secrets.get("MISTRAL_API_KEY", "") if hasattr(st, "secrets") else ""
    return k or os.environ.get("MISTRAL_API_KEY", "") or ""

def summarize_with_mistral(text: str, max_bullets: int = 8, temperature: float = 0.2) -> Optional[str]:
    """
    Improved: forces a compact, sectioned Markdown summary suitable for DCM users.
    Output sections (omit empty ones): Macroeconomics / Markets / Geopolitics & Policy / Primary & Corporate.
    """
    api_key = _mistral_key()
    if not api_key or requests is None or not text.strip():
        return None
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    prompt = (
        "You are a financial news editor for a Global Markets journal. "
        "From the headlines below, produce a VERY concise Markdown summary with clear sections. "
        f"Use at most {max_bullets} bullets TOTAL across all sections. No intro or outro.\n\n"
        "Sections (omit if no signal):\n"
        "1) *Macroeconomics* – growth/inflation data, policy guidance, fiscal updates, energy supply/demand shocks.\n"
        "2) *Markets* – Rates/Credit/Equities/FX/Commodities: key moves, drivers, and liquidity/volatility notes.\n"
        "3) *Geopolitics & Policy* – elections, conflicts, sanctions, trade/industrial policy with market impact.\n"
        "4) *Primary & Corporate* – DCM/ECM/M&A, supply outlook (IG/HY/SSA), notable deals, funding costs.\n\n"
        "Rules:\n"
        "- Deduplicate similar items; prefer cross-market takeaways over raw headlines.\n"
        "- Each bullet ≤ 30 words; add a short driver or implication when possible.\n"
        "- Make clear sentences, not abbreviations.\n"
        "- Output MUST be valid Markdown with the four section headers above (omit empty ones)."
        "\n\nHeadlines:\n" + text
    )

    messages = [
        {"role": "system", "content": "Write crisp, neutral market summaries for professional DCM users. Output Markdown only."},
        {"role": "user", "content": prompt},
    ]
    payload = {"model": "mistral-small-latest", "messages": messages, "temperature": temperature}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


# =======================================================
# UI helpers (links & tables)
# =======================================================

def render_links_note():
    st.markdown("#### DCM quick links")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Global Markets**")
        st.markdown(
            "- [Bloomberg — Markets](https://www.bloomberg.com/markets)\n"
            "- [Financial Times — Markets](https://www.ft.com/markets)\n"
            "- [Reuters — Markets](https://www.reuters.com/markets/)\n"
            "- [WSJ — Markets & Finance](https://www.wsj.com/finance)\n"
            "- [Les Echos — Finance & Marchés](https://www.lesechos.fr/finance-marches)\n"
            "- [MarketWatch — Bonds](https://www.marketwatch.com/markets/bonds)\n"
        )
        st.markdown("**DCM / Primary Market**")
        st.markdown(
            "- [IFR — International Financing Review](https://www.ifre.com/)\n"
            "- [GlobalCapital](https://www.globalcapital.com/)\n"
        )

    with col2:
        st.markdown("**Central Banks**")
        st.markdown(
            "- [ECB — Press](https://www.ecb.europa.eu/press)\n"
            "- [Federal Reserve — Press](https://www.federalreserve.gov/newsevents.htm)\n"
            "- [Bank of England — News](https://www.bankofengland.co.uk/news)\n"
            "- [BIS — Press & Publications](https://www.bis.org/press/)\n"
        )
        st.markdown("**Regulators & Policy**")
        st.markdown(
            "- [ESMA — Press & News](https://www.esma.europa.eu/press-news)\n"
            "- [EBA — News](https://www.eba.europa.eu/news-press/news)\n"
            "- [OECD — Newsroom](https://www.oecd.org/newsroom/)\n"
            "- [IOSCO](https://www.iosco.org/)\n"
        )

    with col3:
        st.markdown("**Sovereign & SSA**")
        st.markdown(
            "- [EU — NGEU / EU Bonds](https://europa.eu/next-generation-eu/)\n"
            "- [France Trésor (AFT)](https://www.aft.gouv.fr/)\n"
            "- [Germany — Finanzagentur](https://www.deutsche-finanzagentur.de/)\n"
            "- [UK Debt Management Office](https://www.dmo.gov.uk/)\n"
        )
        st.markdown("**Sustainable Finance**")
        st.markdown(
            "- [ICMA — Sustainable Finance](https://www.icmagroup.org/sustainable-finance/)\n"
            "- [Climate Bonds Initiative](https://www.climatebonds.net/)\n"
        )

    st.caption("Some sources require subscriptions. Links open in your browser.")

def _render_clickable_table(df_view: pd.DataFrame):
    if df_view.empty:
        st.info("No items to display.")
        return

    # Inject CSS une seule fois
    if not st.session_state.get("_compact_table_css", False):
        st.markdown(
            """
            <style>
              table.compact-table{
                width:100%;
                table-layout:fixed;
                border-collapse:collapse;
                font-size:0.90rem;
                line-height:1.25;
              }
              table.compact-table th, table.compact-table td{
                padding:6px 8px;
                vertical-align:top;
              }
              table.compact-table td:nth-child(1),
              table.compact-table th:nth-child(1){ width:110px; }
              table.compact-table td:nth-child(2),
              table.compact-table th:nth-child(2){ width:160px; }
              table.compact-table td:nth-child(4),
              table.compact-table th:nth-child(4){ width:70px; text-align:center; }
              table.compact-table td:nth-child(3){
                overflow:hidden;
                text-overflow:ellipsis;
                display:-webkit-box;
                -webkit-line-clamp:2;
                -webkit-box-orient:vertical;
                white-space:normal;
                word-break:break-word;
              }
              table.compact-table a{ text-decoration:none; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["_compact_table_css"] = True

    fmt = {"Link": lambda u: f'<a href="{u}" target="_blank">Open</a>' if isinstance(u, str) and u else ""}
    html = df_view.to_html(escape=False, index=False, formatters=fmt, classes="compact-table")
    st.markdown(html, unsafe_allow_html=True)

def _sources_editor(default: Dict[str, str], key: str) -> Dict[str, str]:
    st.caption("Edit sources (one per line: `Name | URL`)")
    default_text = "\n".join([f"{k} | {v}" for k, v in default.items()])
    txt = st.text_area("Sources", value=default_text, key=key, height=140, label_visibility="collapsed")
    out: Dict[str, str] = {}
    for line in txt.splitlines():
        if "|" in line:
            name, url = line.split("|", 1)
            name, url = name.strip(), url.strip()
            if name and url:
                out[name] = url
    return out or default

# =======================================================
# Sections
# =======================================================

def render_news_aggregator():
    st.markdown("#### News aggregator (RSS)")
    st.caption("Pulls headlines from public RSS sources. Filter by keyword and optionally summarize with AI.")

    # ---------- EXPANDED DEFAULT SOURCES (macro/markets/geopolitics) ----------
    sources = {
        # Markets / Business
        "Reuters Business": "http://feeds.reuters.com/reuters/businessNews",
        "Reuters Markets": "http://feeds.reuters.com/reuters/marketsNews",
        "Reuters Commodities": "http://feeds.reuters.com/reuters/commoditiesNews",
        "Reuters Energy": "http://feeds.reuters.com/reuters/energyNews",
        "Reuters Forex": "http://feeds.reuters.com/reuters/forexNews",
        "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "MarketWatch Top": "https://www.marketwatch.com/marketwatch/rss/topstories",

        # World / Geopolitics
        "Reuters World": "http://feeds.reuters.com/Reuters/worldNews",
        "WSJ World": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
        "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",

        # Official / Policy
        "Fed — All Press": "https://www.federalreserve.gov/feeds/press_all.xml",
        "ECB — Press & Speeches": "https://www.ecb.europa.eu/rss/press.html",
        "BIS — Press": "https://www.bis.org/press_rss.xml",
        "OECD — Newsroom": "https://www.oecd.org/newsroom/rss.xml",
    }

    c1, c2 = st.columns([2, 1])
    with c1:
        kw = st.text_input(
            "Keyword filter (e.g., 'ECB', 'syndicate', 'IG issuance', 'oil', 'FX', 'inflation', 'sanctions')",
            ""
        )
    with c2:
        limit = st.number_input("Items per feed", min_value=5, max_value=50, value=10, step=5)

    with st.expander("Show / edit RSS sources", expanded=False):
        sources = _sources_editor(sources, key="rss_sources_editor")

    urls = list(sources.values())
    df = fetch_rss(urls, limit_per_feed=int(limit)) if urls else pd.DataFrame(columns=["source", "title", "link", "published"])

    if not df.empty:
        # 1) Dédup inter-flux
        df["link"] = df["link"].astype(str)
        df["title_norm"] = df["title"].astype(str).str.strip().str.lower()
        df = df.drop_duplicates(subset=["link"]).drop_duplicates(subset=["title_norm"]).copy()

        # 2) Filtre mot-clé
        if kw.strip():
            mask = df["title"].str.contains(kw, case=False, na=False) | df["source"].str.contains(kw, case=False, na=False)
            df = df[mask].copy()

        # 3) Cap GLOBAL à `limit`
        df = df.sort_values("published", ascending=False).head(int(limit))

        # 4) Rendu
        df["Time"] = df["published"].dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d %H:%M")
        df_view = df[["Time", "source", "title", "link"]].rename(columns={"source": "Source", "title": "Title", "link": "Link"})
        _render_clickable_table(df_view)

        # 5) Résumé IA (sur le sous-ensemble capé) — affiché en NOIR
        if _mistral_key():
            sample = "\n".join([f"- {t}" for t in df["title"].head(10).astype(str).tolist()])
            if st.button("Summarize top 10 (AI)"):
                with st.spinner("Summarizing…"):
                    summary = summarize_with_mistral(sample)
                if summary:
                    st.markdown("**AI market summary**")
                    st.markdown(f'<div style="color:black">{summary}</div>', unsafe_allow_html=True)
                else:
                    st.info("Could not summarize right now.")

    else:
        st.info("No headlines fetched yet. Wait until the end of the day!")

def _to_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=["Date", "Close"])
    out = df.copy()
    if "Close" not in out.columns:
        if "Adj Close" in out.columns:
            out = out.rename(columns={"Adj Close": "Close"})
        else:
            return pd.DataFrame(columns=["Date", "Close"])
    out["Date"] = out.index
    out = out.reset_index(drop=True)
    return out[["Date", "Close"]]


def _sparkline(name: str, plot_df: pd.DataFrame) -> alt.Chart:
    if plot_df.empty:
        return alt.Chart(pd.DataFrame({"Date": [], "Close": []})).mark_line()
    s = pd.to_numeric(plot_df["Close"], errors="coerce").dropna()
    y_min, y_max = float(s.min()), float(s.max())
    span = max(y_max - y_min, 1e-6)
    pad = max(span * 0.12, 0.02)
    domain = [y_min - pad, y_max + pad]

    step = None
    if span <= 1.0:
        step = 0.10
    elif span <= 2.5:
        step = 0.20
    elif span <= 5.0:
        step = 0.50

    axis = alt.Axis(title=name, format=",.3f")
    if step:
        start = np.floor(domain[0] / step) * step
        end = np.ceil(domain[1] / step) * step
        ticks = list(np.arange(start, end + 1e-9, step))
        axis = alt.Axis(title=name, format=",.3f", values=ticks)

    return (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(title=None)),
            y=alt.Y("Close:Q", scale=alt.Scale(zero=False, domain=domain, clamp=False), axis=axis),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Close:Q", format=",.3f")],
        )
        .properties(height=160)
    )


def _latest_and_prev(df: pd.DataFrame) -> tuple[float, float]:
    if df is None or getattr(df, "empty", True) or "Close" not in df.columns:
        return (np.nan, np.nan)
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
    return last, prev


# =========================
# NEW: Macroeconomics Dashboard (fusion)
# =========================
def render_macroeconomics_dashboard():
    st.markdown("#### Macroeconomics Dashboard")
    st.caption("Central banks & macro feeds — Official feeds and latest policy rates for Fed / ECB / BoE. Edit/add your own if needed.")

    # --- Policy rates table only (keep) ---
    rates_df = fetch_policy_rates()
    if not rates_df.empty:
        st.dataframe(rates_df, use_container_width=True, hide_index=True)
    else:
        st.info("Could not fetch policy rates automatically.")

    st.markdown("##### Latest levels (10Y)")
    # Keep only these 4 series
    symbols = {
        "DE 10Y Bund (%)": "DE10Y.BOND",
        "FR 10Y OAT (%)":  "FR10Y.BOND",
        "UK 10Y Gilt (%)": "GB10Y.BOND",
        "US 10Y UST (%)":  "^TNX",
    }
    data = fetch_prices(symbols, period="1y", interval="1d")

    cols = st.columns(4)
    ordered_names = list(symbols.keys())
    for i, name in enumerate(ordered_names):
        df = data.get(name, pd.DataFrame())
        if getattr(df, "empty", True):
            continue
        close, prev = _latest_and_prev(df)
        if name.startswith("US "):
            close /= 10.0
            prev /= 10.0
        delta = pct_change(close, prev)
        with cols[i]:
            st.metric(name, f"{close:,.3f}", f"{delta:+.2f}% d/d")

    # --- One interactive multi-series chart (zoom/pan) ---
    plot_rows = []
    for nm, df in data.items():
        if getattr(df, "empty", True):
            continue
        tmp = _to_plot_df(df)
        if nm.startswith("US "):
            tmp = tmp.copy()
            tmp["Close"] = tmp["Close"] / 10.0
        tmp["Series"] = nm
        plot_rows.append(tmp)

    if plot_rows:
        merged = pd.concat(plot_rows, ignore_index=True)
        base = alt.Chart(merged).mark_line().encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Close:Q", title="Yield (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Series:N"), alt.Tooltip("Close:Q", format=",.3f")]
        )
        st.altair_chart(base.interactive().properties(height=260), use_container_width=True)
    else:
        st.info("No data to plot yet.")


# (Legacy sections kept unchanged for the rest of the app, but no longer used in tabs)
def render_rates_dashboard():
    st.markdown("#### Rates & markets dashboard")
    st.caption("Focus Europe. Yahoo Finance when available; otherwise demo series. Axes auto-rescaled for better readability.")

    euro_symbols = {
        "DE 10Y Bund (%)": "DE10Y.BOND",
        "FR 10Y OAT (%)":  "FR10Y.BOND",
        "IT 10Y BTP (%)":  "IT10Y.BOND",
        "ES 10Y Bonos (%)":"ES10Y.BOND",
        "UK 10Y Gilt (%)": "GB10Y.BOND",
    }
    fx_symbols = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X"}
    show_us = st.toggle("Include U.S. Treasuries section", value=False)
    us_symbols = {
        "US 5Y UST (%)":  "^FVX",
        "US 10Y UST (%)": "^TNX",
        "US 30Y UST (%)": "^TYX",
    } if show_us else {}

    symbols = {**euro_symbols, **fx_symbols, **us_symbols}
    data = fetch_prices(symbols, period="1y", interval="1d")

    st.markdown("##### Latest levels")
    cols = st.columns(3)
    ordered_names = list(euro_symbols.keys()) + list(fx_symbols.keys()) + list(us_symbols.keys())
    for i, name in enumerate(ordered_names):
        df = data.get(name, pd.DataFrame())
        if getattr(df, "empty", True):
            continue
        close, prev = _latest_and_prev(df)
        if name.startswith("US "):
            close /= 10.0
            prev /= 10.0
        delta = pct_change(close, prev)
        with cols[i % 3]:
            st.metric(name, f"{close:,.3f}", f"{delta:+.2f}% d/d")

    st.markdown("##### 1Y sparklines (rescaled axes)")

    def show_row(names):
        r = st.columns(3)
        for i, nm in enumerate(names):
            df = data.get(nm, pd.DataFrame())
            if getattr(df, "empty", True):
                continue
            plot_df = _to_plot_df(df)
            if nm.startswith("US "):
                plot_df = plot_df.copy()
                plot_df["Close"] = plot_df["Close"] / 10.0
            ch = _sparkline(nm, plot_df)
            r[i % 3].altair_chart(ch, use_container_width=True)

    show_row(list(euro_symbols.keys()))
    show_row(list(fx_symbols.keys()))
    if show_us and len(us_symbols) > 0:
        st.markdown("###### U.S. Treasuries (optional)")
        show_row(list(us_symbols.keys()))


def render_central_banks():
    st.markdown("#### Central banks & macro feeds")
    st.caption("Official feeds and latest policy rates for Fed / ECB / BoE. Edit/add your own if needed.")

    rates_df = fetch_policy_rates()
    if not rates_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(rates_df, use_container_width=True, hide_index=True)
        with c2:
            # Removed success callout per request
            pass
    else:
        st.info("Could not fetch policy rates automatically.")

    # Removed RSS table of central banks news per request

def render_deal_watch():
    from datetime import datetime  # utilisé pour le nom du fichier

    st.markdown("#### Deal Tracker")
    st.caption("Auto-pulls bond issuance press releases via RSS (editable), filters by keywords, and lets you export.")

    deal_sources = {
        "GlobeNewswire — Prospectus/Announcement": "https://www.globenewswire.com/RssFeed/subjectcode/63-Prospectus%202fAnnouncement%20Of%20Prospectus/feedTitle/GlobeNewswire%20-%20Prospectus%2C%20Announcement%20Of%20Prospectus",
        "GlobeNewswire — Press Releases": "https://www.globenewswire.com/RssFeed/subjectcode/72-Press%20Releases/feedTitle/GlobeNewswire%20-%20Press%20Releases",
        "PR Newswire — All": "https://www.prnewswire.com/rss/news-releases-list.rss",
    }
    with st.expander("Edit deal feeds", expanded=False):
        deal_sources = _sources_editor(deal_sources, key="deal_sources_editor")

    # --- Hidden keywords (not displayed) ---
    default_kw = r"bond|notes|covered|green bond|subordinated|hybrid|AT1|Tier 2|convertible|syndicated|midswap|spread|coupon|maturity|issuance"
    kw = default_kw  # pas d'UI — on garde les filtres cachés

    limit = st.number_input("Items per feed", min_value=5, max_value=50, value=15, step=5)

    urls = list(deal_sources.values())
    df = fetch_rss(urls, limit_per_feed=int(limit)) if urls else pd.DataFrame(columns=["source", "title", "link", "published"])
    if not df.empty:
        mask = df["title"].str.contains(kw, case=False, na=False, regex=True)
        df = df[mask].copy()
        df["Time"] = df["published"].dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d %H:%M")
        view = df[["Time", "source", "title", "link"]].rename(columns={"source": "Source", "title": "Title", "link": "Link"})
        _render_clickable_table(view)
        st.download_button(
            "Download deals list (CSV)",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name=f"deal_tracker_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
        )
    else:
        st.info("No deal headlines fetched yet. Edit/add sources if needed.")

# =======================================================
# Main render() with hard guards (prevents safe_tab warnings)
# =======================================================

def render():
    try:
        st.subheader("Intelligence Desk — Latest News and Insights")
        st.caption("Curated links, aggregated headlines, market rates, policy rates, and primary market updates. Internet optional; demo mode kicks in if data is unavailable.")

        # Quick links
        try:
            render_links_note()
        except Exception as e:
            st.error(f"Links section error: {e}")

        st.markdown("---")

        # Updated tabs
        tabs = st.tabs(["News Aggregator", "Macroeconomics Dashboard", "Deal Tracker"])

        with tabs[0]:
            try:
                render_news_aggregator()
            except Exception as e:
                st.error(f"News Aggregator error: {e}")

        with tabs[1]:
            try:
                render_macroeconomics_dashboard()
            except Exception as e:
                st.error(f"Macroeconomics Dashboard error: {e}")

        with tabs[2]:
            try:
                render_deal_watch()
            except Exception as e:
                st.error(f"Deal Tracker error: {e}")

    except Exception as e:
        st.error(f"News page load error: {e}")
