# page2/news.py
from __future__ import annotations

# =======================================================
# Standard libs
# =======================================================
import io
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# =======================================================
# Third-party libs
# =======================================================
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------- Optional libs (loaded best-effort) ----------
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
# Helpers & Cache
# =======================================================

@st.cache_data(ttl=900)
def fetch_rss(urls: List[str], limit_per_feed: int = 10) -> pd.DataFrame:
    """
    Fetch multiple RSS feeds (if feedparser is available) and return
    a unified dataframe sorted by published date desc.
    """
    rows = []
    if not feedparser or not isinstance(urls, list) or len(urls) == 0:
        return pd.DataFrame(columns=["source", "title", "link", "published"])

    for u in urls:
        if not isinstance(u, str) or not u.strip():
            continue
        try:
            feed = feedparser.parse(u)
            # Extract source title if present
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

            # Entries
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
    """
    Fetch historical prices with yfinance; fallback to demo series if unavailable.
    """
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
    """
    Generate a sinusoidal + noise demo timeseries for 'Close' prices.
    """
    idx = pd.date_range(end=datetime.utcnow().date(), periods=252, freq="B")
    base = 3.0 if any(tag in name for tag in ["US", "DE", "FR", "UK", "IT", "ES"]) else 1.10
    vals = base + 0.3 * np.sin(np.linspace(0, 8, len(idx))) + np.random.normal(0, 0.05, len(idx))
    df = pd.DataFrame({"Close": vals}, index=idx)
    return df


def demo_prices(symbols: Dict[str, str] | None) -> Dict[str, pd.DataFrame]:
    """
    Return demo series for each requested symbol name.
    """
    symbols = symbols or {}
    return {k: demo_series(k) for k in symbols.keys()}


def pct_change(latest: float, prev: float) -> float:
    """
    Percentage change helper.
    """
    if prev == 0:
        return 0.0
    return (latest - prev) / prev * 100.0


# =======================================================
# Policy rates (Fed / ECB / BoE)
# =======================================================

@st.cache_data(ttl=3600)
def fetch_policy_rates() -> pd.DataFrame:
    """
    Fetch latest policy rates (Fed range, ECB MRO/DFR) via FRED CSV endpoints.
    Also attempts BoE Bank Rate by scraping the BoE page.
    Displays a styled table in Streamlit and returns a DataFrame of rows.
    """
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

    # -------- Fed (target range upper/lower) --------
    rows = []
    up, up_date = fred_last("DFEDTARU")
    lo, lo_date = fred_last("DFEDTARL")
    asof = up_date or lo_date
    if up is not None and lo is not None:
        rows.append({
            "Authority": "Federal Reserve (FOMC)",
            "Measure": "Fed Funds Target Range",
            "Latest": f"{lo:.2f}–{up:.2f}%",
            "As of": asof.tz_convert("Europe/Paris").strftime("%Y-%m-%d") if isinstance(asof, pd.Timestamp) else "",
        })

    # -------- ECB (MRO / DFR) --------
    mro, mro_date = fred_last("ECBMRRFR")
    dfr, dfr_date = fred_last("ECBDFR")
    if mro is not None:
        rows.append({
            "Authority": "European Central Bank",
            "Measure": "Main Refinancing Rate (MRO)",
            "Latest": f"{mro:.2f}%",
            "As of": mro_date.tz_convert("Europe/Paris").strftime("%Y-%m-%d") if isinstance(mro_date, pd.Timestamp) else "",
        })
    if dfr is not None:
        rows.append({
            "Authority": "European Central Bank",
            "Measure": "Deposit Facility Rate (DFR)",
            "Latest": f"{dfr:.2f}%",
            "As of": dfr_date.tz_convert("Europe/Paris").strftime("%Y-%m-%d") if isinstance(dfr_date, pd.Timestamp) else "",
        })

    # -------- Streamlit render (bold 'Latest') --------
    df = pd.DataFrame(rows)

    def highlight_latest(_):
        return "font-weight: bold"

    styled_df = df.style.applymap(highlight_latest, subset=["Latest"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # -------- BoE (scrape current Bank Rate) --------
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
                    "As of": datetime.utcnow().astimezone().strftime("%Y-%m-%d"),
                })
        except Exception:
            pass

    # Note: return columns use 'AsOf' key as originally implemented
    return pd.DataFrame(rows, columns=["Authority", "Measure", "Latest", "AsOf"])


# =======================================================
# Mistral summary (HTTP, no SDK)
# =======================================================

def _mistral_key() -> str:
    k = st.secrets.get("MISTRAL_API_KEY", "") if hasattr(st, "secrets") else ""
    return k or os.environ.get("MISTRAL_API_KEY", "") or ""


def summarize_with_mistral(text: str, max_bullets: int = 6, temperature: float = 0.2) -> Optional[str]:
    """
    Summarize headlines into concise DCM-focused bullets using Mistral chat endpoint.
    """
    api_key = _mistral_key()
    if not api_key or requests is None or not text.strip():
        return None
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = (
        f"Summarize the following DCM-relevant headlines into up to {max_bullets} bullet points. "
        f"Focus on rates, credit markets, supply (deals), and central banks. "
        "Return concise bullets in English.\n\n" + text
    )
    messages = [
        {"role": "system", "content": "You write concise market summaries for Debt Capital Markets."},
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
    """
    Quick curated links to markets / central banks / SSA / sustainable finance.
    """
    st.markdown("#### Links to financial news")
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
        st.markdown("**Sovereign news (SSA)**")
        st.markdown(
            "- [EU — NGEU / EU Bonds](https://european-union.europa.eu/index_fr)\n"
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
    """
    Render a compact HTML table with clickable 'Open' links.
    """
    if df_view.empty:
        st.info("No items to display.")
        return

    # Inject CSS once per session
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
    """
    Textarea editor: one source per line in the form 'Name | URL'.
    """
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
# Section: News Aggregator (RSS)
# =======================================================

def render_news_aggregator():
    """
    RSS-based news aggregator with keyword filter and optional AI summary.
    """
    st.markdown("#### News aggregator (RSS)")
    st.caption("Pulls headlines from public RSS sources. Filter by keyword and optionally summarize with AI.")

    # ----- Default sources (editable) -----
    sources = {
        "Reuters Business": "http://feeds.reuters.com/reuters/businessNews",
        "Reuters Markets": "http://feeds.reuters.com/reuters/marketsNews",
        "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "MarketWatch Top": "https://www.marketwatch.com/marketwatch/rss/topstories",
        "Fed — All Press": "https://www.federalreserve.gov/feeds/press_all.xml",
        "ECB — Press & Speeches": "https://www.ecb.europa.eu/rss/press.html",
    }

    # Controls
    c1, c2 = st.columns([2, 1])
    with c1:
        kw = st.text_input("Keyword filter (e.g., 'ECB', 'syndicate', 'IG issuance', 'swaps')", "")
    with c2:
        limit = st.number_input("Items per feed", min_value=5, max_value=50, value=10, step=5)

    # Editable list
    with st.expander("Show / edit RSS sources", expanded=False):
        sources = _sources_editor(sources, key="rss_sources_editor")

    # Fetch
    urls = list(sources.values())
    df = fetch_rss(urls, limit_per_feed=int(limit)) if urls else pd.DataFrame(columns=["source", "title", "link", "published"])

    if not df.empty:
        # 1) Cross-feed dedup
        df["link"] = df["link"].astype(str)
        df["title_norm"] = df["title"].astype(str).str.strip().str.lower()
        df = df.drop_duplicates(subset=["link"]).drop_duplicates(subset=["title_norm"]).copy()

        # 2) Keyword filter
        if kw.strip():
            mask = df["title"].str.contains(kw, case=False, na=False) | df["source"].str.contains(kw, case=False, na=False)
            df = df[mask].copy()

        # 3) Global cap to `limit`
        df = df.sort_values("published", ascending=False).head(int(limit))

        # 4) Render
        df["Time"] = df["published"].dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d %H:%M")
        df_view = df[["Time", "source", "title", "link"]].rename(columns={"source": "Source", "title": "Title", "link": "Link"})
        _render_clickable_table(df_view)

        # 5) AI summary (Mistral) — top 10 titles
        if _mistral_key():
            sample = "\n".join([f"- {t}" for t in df["title"].head(10).astype(str).tolist()])
            if st.button("Summarize top 10 (AI)"):
                with st.spinner("Summarizing…"):
                    summary = summarize_with_mistral(sample)
                if summary:
                    st.markdown("**AI market summary**")
                    st.markdown(summary)
                else:
                    st.info("Could not summarize right now.")
    else:
        st.info("No headlines fetched yet. Check your internet connection or adjust sources.")


# =======================================================
# Plot helpers (sparklines)
# =======================================================

def _sparkline(name: str, plot_df: pd.DataFrame) -> alt.Chart:
    """
    Build a compact Altair line chart with padded y-domain and optional tick step.
    """
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
    """
    Return last and previous close values for delta calculation.
    """
    if df is None or getattr(df, "empty", True) or "Close" not in df.columns:
        return (np.nan, np.nan)
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
    return last, prev


# =======================================================
# Macroeconomics Dashboard (Fusion)
# =======================================================

def render_macroeconomics_dashboard():
    """
    Dashboard with:
    - Policy rates table (Fed/ECB/BoE)
    - Latest 10Y levels (US/DE/FR/UK) from YF (TNX scaled) with FRED fallback
    - Monthly OECD/MEI spreads (US−FR, FR−DE) for last 10 years (CSV files)
    """
    global pd
    st.markdown("#### Macroeconomics Dashboard")
    st.caption("Central banks & macro feeds — Official feeds and latest policy rates for Fed / ECB / BoE. Edit/add your own if needed.")

    # --- Policy rates table: fetch & display inside fetch_policy_rates ---
    _ = fetch_policy_rates()

    # ---------- Latest levels (10Y) ----------
    st.markdown("##### Latest levels (10Y)")

    SYMBOLS_YF = {
        "German 10Y Bund Yield (%)":   "DE10Y.BOND",
        "French 10Y OAT Yield (%)":    "FR10Y.BOND",
        "UK 10Y Gilt Yield (%)":       "GB10Y.BOND",
        "US 10Y Treasury Yield (%)":   "^TNX",   # TNX = tenths of a percent
    }

    FRED_SERIES = {
        "US 10Y Treasury Yield (%)":   "DGS10",
        "German 10Y Bund Yield (%)":   "IRLTLT01DEM156N",
        "French 10Y OAT Yield (%)":    "IRLTLT01FRM156N",
        "UK 10Y Gilt Yield (%)":       "IRLTLT01GBM156N",
    }

    def _clean_yf_df(df: pd.DataFrame) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                return None
            s = df[num_cols[0]]
        out = pd.DataFrame({"Date": df.index, "Close": pd.to_numeric(s, errors="coerce")})
        out = out.dropna().reset_index(drop=True)
        return out if not out.empty else None

    def fetch_yf_one(ticker: str, period="1y", interval="1d") -> pd.DataFrame | None:
        if yf is None:
            return None
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
            return _clean_yf_df(df)
        except Exception:
            return None

    def fetch_fred_series(series_id: str) -> pd.DataFrame | None:
        if requests is None:
            return None
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            raw = pd.read_csv(io.StringIO(r.text))
            date_col, val_col = raw.columns[0], raw.columns[1]
            df = pd.DataFrame({
                "Date": pd.to_datetime(raw[date_col], errors="coerce"),
                "Close": pd.to_numeric(raw[val_col], errors="coerce"),
            }).dropna().sort_values("Date").reset_index(drop=True)
            return df if not df.empty else None
        except Exception:
            return None

    # Real data fetch (Yahoo -> FRED fallback)
    real_data: dict[str, pd.DataFrame] = {}
    for name, ticker in SYMBOLS_YF.items():
        df = fetch_yf_one(ticker, period="1y", interval="1d")
        if df is not None and not df.empty:
            if name == "US 10Y Treasury Yield (%)":
                df = df.copy()
                df["Close"] = df["Close"] / 10.0  # TNX tenths → %
            real_data[name] = df
        else:
            sid = FRED_SERIES.get(name)
            df_fred = fetch_fred_series(sid) if sid else None
            if df_fred is not None and not df_fred.empty:
                real_data[name] = df_fred

    # KPI render with date (mm/dd/yyyy)
    cols = st.columns(4)
    for i, name in enumerate(SYMBOLS_YF.keys()):
        df = real_data.get(name)
        with cols[i]:
            if df is None or df.empty:
                st.metric(label=name, value="n/a", delta="n/a")
                st.caption("Data as of —")
                continue

            close, prev = _latest_and_prev(df)

            # Daily vs monthly tag (gap heuristic)
            delta_days = (df["Date"].iloc[-1] - df["Date"].iloc[-2]).days if len(df) >= 2 else 1
            delta_tag = "m/m" if delta_days >= 15 else "d/d"

            st.metric(label=name, value=f"{close:,.3f}", delta=f"{pct_change(close, prev):+.2f}% {delta_tag}")

            asof = pd.to_datetime(df["Date"].iloc[-1])
            st.caption(f"Data as of {asof.strftime('%m/%d/%Y')}")

    # ===================================================
    # Spreads chart (US−FR & FR−DE) — monthly, last 10y
    # ===================================================

    # ---- Data directory resolution ----
    def _find_dir_up(start: Path, target_dirname: str) -> Path | None:
        """Climb up from 'start' to find a directory named 'target_dirname'."""
        cur = start
        for _ in range(6):
            candidate = cur / target_dirname
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()
            if cur.parent == cur:
                break
            cur = cur.parent
        return None

    def _get_data_dir() -> Path:
        # a) st.secrets["DATA_DIR"] if present
        try:
            if "DATA_DIR" in st.secrets:
                p = Path(st.secrets["DATA_DIR"]).expanduser().resolve()
                if p.exists():
                    return p
        except Exception:
            pass

        # b) environment variable
        env_p = os.getenv("DATA_DIR")
        if env_p:
            p = Path(env_p).expanduser().resolve()
            if p.exists():
                return p

        # c) search for 'xlx' upward from this file
        here = Path(__file__).resolve()
        found = _find_dir_up(here.parent, "xlx")
        if found:
            return found

        # d) fallback: CWD/xlx
        return (Path.cwd() / "xlx").resolve()

    DATA_DIR = str(_get_data_dir())

    def _pick_file(patterns: list[str]) -> Optional[str]:
        try:
            files = os.listdir(DATA_DIR)
        except Exception:
            return None
        for p in patterns:
            regex = re.compile(p, re.IGNORECASE)
            candidates = [f for f in files if regex.search(f)]
            if candidates:
                candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(DATA_DIR, fn)), reverse=True)
                return os.path.join(DATA_DIR, candidates[0])
        return None

    def _load_monthly_series(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        # normalize columns
        df.columns = [c.strip().lower() for c in df.columns]

        # date column
        date_col = None
        for c in ["date", "observation_date", "month"]:
            if c in df.columns:
                date_col = c
                break
        if not date_col:
            date_col = df.columns[0]

        # value column
        val_col = None
        for c in ["value", "yield", "yld", "rate", "close", "price"]:
            if c in df.columns:
                val_col = c
                break
        if not val_col:
            num_cols = [c for c in df.columns if c != date_col]
            for c in num_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                raise ValueError(f"No numeric value column in {os.path.basename(path)}")
            val_col = num_cols[0]

        out = df[[date_col, val_col]].copy()
        out.columns = ["Date", "Value"]
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
        out = out.dropna(subset=["Date", "Value"]).sort_values("Date")

        # resample to monthly if needed (last point of month)
        monthly_points = out["Date"].dt.to_period("M").nunique()
        if monthly_points < len(out) * 0.9:
            out = out.set_index("Date").resample("M").last().reset_index()

        # end-of-month alignment
        out["Date"] = out["Date"].dt.to_period("M").dt.to_timestamp("M")
        out = out.drop_duplicates(subset=["Date"], keep="last")
        return out

    # Expected files (US, FR, DE)
    us_path = _pick_file([r"^IRLTLT01USM156N.*\.csv$", r"^US-?10Y.*\.csv$", r".*US.*10.*\.csv$"])
    fr_path = _pick_file([r"^IRLTLT01FRM156N.*\.csv$", r"^FR-?10Y.*\.csv$", r".*(FR|FRA).*10.*\.csv$"])
    de_path = _pick_file([r"^IRLTLT01DEM156N.*\.csv$", r"^DE-?10Y.*\.csv$", r".*(DE|GER|DEU).*10.*\.csv$"])

    if not us_path or not fr_path or not de_path:
        missing = []
        if not us_path: missing.append("US (IRLTLT01USM156N*.csv / US-10Y*.csv)")
        if not fr_path: missing.append("France (IRLTLT01FRM156N*.csv / FR-10Y*.csv)")
        if not de_path: missing.append("Germany (IRLTLT01DEM156N*.csv / DE-10Y*.csv)")
        st.info(f"Spread chart: CSV introuvables dans `{DATA_DIR}`. Manquants: {', '.join(missing)}")
        return

    try:
        us = _load_monthly_series(us_path).rename(columns={"Value": "US_10Y"})
        fr = _load_monthly_series(fr_path).rename(columns={"Value": "FR_10Y"})
        de = _load_monthly_series(de_path).rename(columns={"Value": "DE_10Y"})

        df = us.merge(fr, on="Date", how="inner").merge(de, on="Date", how="inner").sort_values("Date")

        # Spreads (renamed as requested)
        df["spread_pp_FR_vs_US"] = (df["US_10Y"] - df["FR_10Y"]).round(3)  # US − FR
        df["spread_pp_FR_vs_DE"] = (df["FR_10Y"] - df["DE_10Y"]).round(3)  # FR − DE (OAT − Bund)

        # Last 10 years
        today = pd.Timestamp.today().normalize()
        cutoff = (today - pd.DateOffset(years=10)).to_period("M").to_timestamp("M")
        df10 = df[df["Date"] >= cutoff].copy()

        # ----- Altair chart (layered with zero line) -----
        series_labels = {
            "spread_pp_FR_vs_US": "Spread (US 10Y − FR 10Y)",
            "spread_pp_FR_vs_DE": "Spread (FR 10Y − DE 10Y)",
        }
        plot_long = pd.melt(
            df10,
            id_vars=["Date"],
            value_vars=["spread_pp_FR_vs_US", "spread_pp_FR_vs_DE"],
            var_name="Series",
            value_name="Spread_pp"
        )
        plot_long["Series"] = plot_long["Series"].map(series_labels)

        st.markdown("##### US−FR & FR−DE 10Y Long-Term Government Bond Yield Spreads (OECD/MEI, Monthly) — Last 10 Years")

        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[6, 4]).encode(y="y:Q")

        chart = (
            alt.layer(
                alt.Chart(plot_long)
                   .mark_line()
                   .encode(
                       x=alt.X(
                           "Date:T",
                           title="Date (month end)",
                           axis=alt.Axis(
                               format="%Y",
                               tickCount="year",
                               labelAngle=0,
                               ticks=True,
                               domain=True,
                           ),
                       ),
                       y=alt.Y("Spread_pp:Q", title="Spread (percentage points)", scale=alt.Scale(zero=False)),
                       color=alt.Color("Series:N", legend=alt.Legend(title=None, orient="bottom")),
                       tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Series:N"), alt.Tooltip("Spread_pp:Q", title="Spread", format=",.3f")],
                   ),
                zero_line
            )
            .properties(height=360)
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "Spreads: US−FR = US 10Y − France 10Y ; FR−DE = France 10Y − Germany 10Y. "
            "Source: OECD/MEI via FRED — US: IRLTLT01USM156N, FR: IRLTLT01FRM156N, DE: IRLTLT01DEM156N. "
            "Voir FRED (US 10Y): https://fred.stlouisfed.org/series/IRLTLT01USM156N#:~:text=Observations"
        )

        with st.expander("Latest observations (QA)"):
            qa_cols = ["Date", "US_10Y", "FR_10Y", "DE_10Y", "spread_pp_FR_vs_US", "spread_pp_FR_vs_DE"]
            st.dataframe(df10[qa_cols].tail(24).set_index("Date"), use_container_width=True)

    except Exception as e:
        st.error(f"US–FR / FR–DE 10Y spreads error: {e}")


# =======================================================
# Central Banks (compact section)
# =======================================================

def render_central_banks():
    """
    Compact central banks section (policy rates only; RSS removed per request).
    """
    st.markdown("#### Central banks & macro feeds")
    st.caption("Official feeds and latest policy rates for Fed / ECB / BoE. Edit/add your own if needed.")

    rates_df = fetch_policy_rates()
    if not rates_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(rates_df, use_container_width=True, hide_index=True)
        with c2:
            # Reserved for future widgets (e.g., notes, switches)
            pass
    else:
        st.info("Could not fetch policy rates automatically.")


# =======================================================
# Deal Watch (DCM-hardened)
# =======================================================

def render_deal_watch():
    """
    Tracker that pulls deal headlines via RSS, filters for DCM-style actions/instruments,
    attempts issuer extraction, and detects banks from title and article body.
    """
    from html import unescape as _unescape

    st.markdown("#### Deal Tracker")
    st.caption(
        "Auto-pulls bond issuance press releases via RSS (editable), filters by keywords, and shows latest issuers. "
        "Only headlines that combine a DCM instrument AND a primary action verb are kept. "
        "Non-DCM 'platform/product launches' are excluded."
    )

    # --- Default deal sources (editable) ---
    deal_sources = {
        "GlobeNewswire — Prospectus/Announcement": "https://www.globenewswire.com/RssFeed/subjectcode/63-Prospectus%202fAnnouncement%20Of%20Prospectus/feedTitle/GlobeNewswire%20-%20Prospectus%2C%20Announcement%20Of%20Prospectus",
        "GlobeNewswire — Press Releases": "https://www.globenewswire.com/RssFeed/subjectcode/72-Press%20Releases/feedTitle/GlobeNewswire%20-%20Press%20Releases",
        "PR Newswire — All": "https://www.prnewswire.com/rss/news-releases-list.rss",
        "PR Newswire — Financial Services": "https://www.prnewswire.com/rss/financial-services/all-financial-services-news.rss",
        "Euronext — News": "https://live.euronext.com/en/rss_feed/news",
        "London Stock Exchange — RNS (All)": "https://www.londonstockexchange.com/exchange/feeds/rss/news.xml",
    }

    # ======= DCM filter: require INSTRUMENT & ACTION; exclude negatives =======
    _INSTRUMENT = r"(?:green\s+bond|sustainability[-\s]?linked|sustainability|social|transition|sukuk|bond|notes?|debenture|schuldschein|covered|mortgage[-\s]?backed|abs|mbs|emtn|eurobond|private\s+placement|ppn|at1|additional\s+tier\s*1|tier\s*2|subordinated|senior(?!\s+living)|convertible|exchangeable)"
    _ACTION = r"(?:announce[sd]?|price[sd]?|launch(?:ed|es|ing)?|issue[sd]?|offer[sd]?|tap(?:ped|s)?|reopen(?:ed|s|ing)?|upsiz(?:e|ed|es|ing)|book(?:build|built|building)|mandate[sd]?|appoints?\s+bookrunners?)"
    _SIZE = r"(?:\$\s?\d+|\d+\s?(?:bn|billion|m|million|mm|€|eur|usd|gbp|chf|cad|aud)|\b\d{2,4}\s?(?:year|yr|ans)\b)"  # optional info
    _NEGATIVE = r"(?:platform|software|app|webinar|token|crypto|nft|ai\s+model|partnership|conference|award|etf|equity|stock|ipo|dividend|earnings|results|buyback|share\s+repurchase|product\s+launch|clinical\s+trial|vaccine|gaming|e-?commerce)"

    DCM_PATTERN = re.compile(rf"(?is)^(?=.*\b{_INSTRUMENT}\b)(?=.*\b{_ACTION}\b)(?!.*\b{_NEGATIVE}\b).*")

    def _is_dcm_headline(title: str) -> bool:
        if not isinstance(title, str) or not title.strip():
            return False
        t = title.strip()
        if not DCM_PATTERN.search(t):
            return False
        return True

    # Fetch
    limit = st.number_input("Items per feed", min_value=5, max_value=50, value=15, step=5)
    urls = list(deal_sources.values())
    df = fetch_rss(urls, limit_per_feed=int(limit)) if urls else pd.DataFrame(columns=["source", "title", "link", "published"])

    # ---------- Bank synonyms ----------
    BANK_SYNONYMS = {
        "J.P. Morgan": ["J.P. Morgan", "JP Morgan", "JPMorgan"],
        "Morgan Stanley": ["Morgan Stanley"],
        "Goldman Sachs & Co. LLC": ["Goldman Sachs & Co. LLC"],
        "Goldman Sachs": ["Goldman Sachs"],
        "BofA Securities": ["BofA Securities", "BofA", "Bank of America"],
        "Citigroup": ["Citigroup", "Citi"],
        "Barclays": ["Barclays", "Barclays Bank PLC"],
        "BNP Paribas": ["BNP Paribas"],
        "Société Générale": ["Société Générale", "Societe Generale"],
        "Crédit Agricole CIB": ["Crédit Agricole CIB", "Credit Agricole CIB"],
        "Crédit Agricole": ["Crédit Agricole", "Credit Agricole"],
        "Natixis": ["Natixis"],
        "HSBC": ["HSBC"],
        "Deutsche Bank": ["Deutsche Bank"],
        "UBS": ["UBS"],
        "Wells Fargo": ["Wells Fargo"],
        "Jefferies": ["Jefferies"],
        "RBC Capital Markets": ["RBC Capital Markets", "RBC"],
        "Scotiabank": ["Scotiabank"],
        "TD Securities": ["TD Securities"],
        "Mizuho": ["Mizuho"],
        "SMBC Nikko": ["SMBC Nikko", "SMBC"],
        "Nomura": ["Nomura"],
        "ING": ["ING", "ING Bank"],
        "Intesa Sanpaolo": ["Intesa Sanpaolo"],
        "UniCredit": ["UniCredit"],
        "Santander": ["Santander", "Banco Santander"],
        "BBVA": ["BBVA"],
        # Additions (Great Elm example)
        "Lucid Capital Markets": ["Lucid Capital Markets", "Lucid Capital Markets LLC", "Lucid Capital Markets, LLC"],
        "Piper Sandler": ["Piper Sandler", "Piper Sandler & Co.", "Piper Sandler & Co", "Piper Sandler and Co."],
        "Clear Street": ["Clear Street", "Clear Street LLC", "Clear Street, LLC"],
        "InspereX": ["InspereX", "InspereX LLC", "Insperex", "Insperex LLC"],
        "Janney Montgomery Scott": ["Janney Montgomery Scott", "Janney Montgomery Scott LLC", "Janney"],
    }

    ALT_TO_CANON = {}
    for canon, alts in BANK_SYNONYMS.items():
        for a in alts:
            ALT_TO_CANON[a.lower()] = canon

    def _alts_to_regex_parts(alts: List[str]) -> List[str]:
        parts = []
        for a in alts:
            esc = re.escape(a)
            esc = esc.replace(r"\ ", r"\s+")
            parts.append(esc)
        return parts

    BANK_PATTERN = re.compile(
        r"(?<![A-Za-z])(?:%s)(?![A-Za-z])" % "|".join(_alts_to_regex_parts([k for sub in BANK_SYNONYMS.values() for k in sub])),
        re.IGNORECASE,
    )

    def _extract_banks_from_text(text: str) -> list[str]:
        hits = []
        for m in BANK_PATTERN.finditer(text or ""):
            canon = ALT_TO_CANON.get(m.group(0).lower())
            if canon:
                hits.append(canon)
        return list(dict.fromkeys(hits))  # unique & order-preserving

    # Cache for fetched article HTML→plaintext
    if "deal_article_cache" not in st.session_state:
        st.session_state.deal_article_cache = {}

    def _fetch_article_plaintext(url: str) -> str:
        cache = st.session_state.deal_article_cache
        if url in cache:
            return cache[url]
        if 'requests' in globals() and requests is not None and isinstance(url, str) and url.startswith("http"):
            try:
                hdr = {"User-Agent": "Mozilla/5.0 (DealTracker/1.1)"}
                r = requests.get(url, timeout=8, headers=hdr)
                r.raise_for_status()
                html = r.text
                html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
                html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
                text = _unescape(re.sub(r"(?s)<[^>]+>", " ", html))
                text = re.sub(r"\s+", " ", text).strip()
            except Exception:
                text = ""
        else:
            text = ""
        cache[url] = text
        return text

    _VERBS = r"(?:announces?|prices?|launches?|issues?|offers?|files|to\s+issue|to\s+offer|mandates?|appoints?)"
    ISSUER_VERB_RE = re.compile(rf"^(.*?)(?:\s+){_VERBS}\b", re.IGNORECASE)

    def _extract_issuer(title: str) -> str | None:
        if not _is_dcm_headline(title):
            return None
        t = re.sub(r"\s+", " ", title or "").strip()
        t = re.sub(r"\((?:NASDAQ|NYSE|EPA|LSE|TSX|SIX|FWB|XETRA|HKEX|ASX|BME)[^)]*\)", "", t, flags=re.IGNORECASE)
        t = t.split(" | ")[0]
        m = ISSUER_VERB_RE.search(t)
        cand = m.group(1).strip(" -—:;,.\u2013\u2014") if m else None
        if not cand:
            cand = t.split(":")[0].split("—")[0].split("-")[0].strip()
        if cand and 1 <= len(cand.split()) <= 8 and not cand.lower().startswith(("press release", "company", "the ")):
            return cand
        return None

    # ---------- Main flow ----------
    if not df.empty:
        mask = df["title"].apply(_is_dcm_headline)
        df = df[mask].copy()

        if df.empty:
            st.info("No DCM-style headlines detected (try adjusting sources or broaden the filter).")
            return

        df["Time"] = df["published"].dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d %H:%M")
        df["Issuer"] = df["title"].apply(_extract_issuer)

        # Banks from title
        df["Banks_title"] = df["title"].apply(_extract_banks_from_text)

        # Banks from page (limited fetch count)
        banks_from_page = []
        max_fetch = 12
        fetched = 0
        for _, row in df.sort_values("published", ascending=False).iterrows():
            if fetched >= max_fetch:
                banks_from_page.append([])
                continue
            text = _fetch_article_plaintext(row.get("link", ""))
            banks_from_page.append(_extract_banks_from_text(text))
            fetched += 1

        df_sorted = df.sort_values("published", ascending=False).copy()
        df_sorted["Banks_page"] = banks_from_page
        df = df_sorted.sort_index()

        # Merge banks (title + page)
        def _merge_banks(row) -> list[str]:
            seen = {}
            for b in (row.get("Banks_title", []) or []) + (row.get("Banks_page", []) or []):
                if b:
                    seen[b] = True
            return sorted(seen.keys(), key=lambda x: x.lower())

        df["Banks"] = df.apply(_merge_banks, axis=1)

        # Clickable table
        view = df[["Time", "source", "title", "link"]].rename(columns={"source": "Source", "title": "Title", "link": "Link"})
        _render_clickable_table(view)

        # Latest issuers summary
        issuers_lines = []
        for _, row in df.sort_values("published", ascending=False).iterrows():
            issuer = row.get("Issuer")
            banks = row.get("Banks") or []
            if issuer:
                if banks:
                    issuers_lines.append(f"{issuer} ({', '.join(banks[:-1]) + ' and ' + banks[-1] if len(banks) > 1 else banks[0]})")
                else:
                    issuers_lines.append(issuer)
        issuers_lines = list(dict.fromkeys(issuers_lines))  # unique

        if issuers_lines:
            st.markdown("**Latest issuers detected**")
            st.markdown("- " + "\n- ".join(issuers_lines[:25]))
        else:
            st.info("No clear issuer names detected in headlines that passed the DCM filter.")
    else:
        st.info("No deal headlines fetched yet. Edit/add sources if needed.")


# =======================================================
# Main render() — Tabbed layout with guards
# =======================================================

def render():
    """
    Page entrypoint. Renders:
      - Links section
      - Tabs: News Aggregator / Macroeconomics Dashboard / Deal Tracker
    """
    try:
        st.subheader("Intelligence Desk — Latest News and Insights")
        st.caption("Curated links, aggregated headlines, market rates, policy rates, and primary market updates. Internet optional; demo mode kicks in if data is unavailable.")

        # Quick links
        try:
            render_links_note()
        except Exception as e:
            st.error(f"Links section error: {e}")

        st.markdown("---")

        # Tabs
        tabs = st.tabs(["📰 News Aggregator", "🌐 Macroeconomics Dashboard", "📌 Deal Tracker"])

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
