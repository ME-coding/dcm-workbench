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

    # Ton code qui remplit rows
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

    # Transformation en DataFrame
    df = pd.DataFrame(rows)

    # Application du style : mettre en gras uniquement la colonne "Latest"
    def highlight_latest(val):
        return "font-weight: bold"

    styled_df = df.style.applymap(highlight_latest, subset=["Latest"])

    # Affichage dans Streamlit (index masqué)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

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

    return pd.DataFrame(rows, columns=["Authority", "Measure", "Latest", "AsOf"])


# =======================================================
# Mistral summary (HTTP, no SDK)
# =======================================================

def _mistral_key() -> str:
    k = st.secrets.get("MISTRAL_API_KEY", "") if hasattr(st, "secrets") else ""
    return k or os.environ.get("MISTRAL_API_KEY", "") or ""


def summarize_with_mistral(text: str, max_bullets: int = 6, temperature: float = 0.2) -> Optional[str]:
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

    # === Version préférée (seconde) ===
    sources = {
        "Reuters Business": "http://feeds.reuters.com/reuters/businessNews",
        "Reuters Markets": "http://feeds.reuters.com/reuters/marketsNews",
        "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "MarketWatch Top": "https://www.marketwatch.com/marketwatch/rss/topstories",
        "Fed — All Press": "https://www.federalreserve.gov/feeds/press_all.xml",
        "ECB — Press & Speeches": "https://www.ecb.europa.eu/rss/press.html",
    }

    c1, c2 = st.columns([2, 1])
    with c1:
        kw = st.text_input("Keyword filter (e.g., 'ECB', 'syndicate', 'IG issuance', 'swaps')", "")
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

        # 5) Résumé IA (version préférée, sans wrapper couleur)
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

    # --- Policy rates table: fetch & display inside fetch_policy_rates (no duplicate table) ---
    _ = fetch_policy_rates()

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

    # === REPLACED / UPDATED: Spreads chart (US−FR & FR−DE), mensuel, 10 dernières années ===
    DATA_DIR = r"C:\Users\Maxime\Dev\DCM_Workbench\xlx"

    def _pick_file(patterns: list[str]) -> Optional[str]:
        try:
            files = os.listdir(DATA_DIR)
        except Exception:
            return None
        for p in patterns:
            regex = re.compile(p, re.IGNORECASE)
            candidates = [f for f in files if regex.search(f)]
            if candidates:
                # prend le plus récent (mtime)
                candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(DATA_DIR, fn)), reverse=True)
                return os.path.join(DATA_DIR, candidates[0])
        return None

    def _load_monthly_series(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        # normalise colonnes
        df.columns = [c.strip().lower() for c in df.columns]
        # colonne date
        date_col = None
        for c in ["date", "observation_date", "month"]:
            if c in df.columns:
                date_col = c
                break
        if not date_col:
            # fallback: première colonne
            date_col = df.columns[0]
        # colonne valeur
        val_col = None
        for c in ["value", "yield", "yld", "rate", "close", "price"]:
            if c in df.columns:
                val_col = c
                break
        if not val_col:
            # fallback: première colonne numérique != date
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

        # ré-échantillonne en mensuel si nécessaire (dernier point du mois)
        monthly_points = out["Date"].dt.to_period("M").nunique()
        if monthly_points < len(out) * 0.9:
            out = (
                out.set_index("Date")
                   .resample("M")
                   .last()
                   .reset_index()
            )

        # date = fin de mois pour alignement propre
        out["Date"] = out["Date"].dt.to_period("M").dt.to_timestamp("M")
        out = out.drop_duplicates(subset=["Date"], keep="last")
        return out

    # Fichiers attendus (US, FR, DE)
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

        # Spreads (renommage demandé)
        df["spread_pp_FR_vs_US"] = (df["US_10Y"] - df["FR_10Y"]).round(3)  # US − FR
        df["spread_pp_FR_vs_DE"] = (df["FR_10Y"] - df["DE_10Y"]).round(3)  # FR − DE (OAT − Bund)

        # Filtre 10 dernières années
        today = pd.Timestamp.today().normalize()
        cutoff = (today - pd.DateOffset(years=10)).to_period("M").to_timestamp("M")
        df10 = df[df["Date"] >= cutoff].copy()

        # ----- Graphique spreads (Altair) -----
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

        # Titre via Streamlit (évite qu’il soit masqué par les métriques au-dessus)
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
                               format="%Y",          # affichage des années
                               tickCount="year",     # un tick par année
                               labelAngle=0,         # labels horizontaux
                               ticks=True,
                               domain=True
                           ),
                       ),
                       y=alt.Y(
                           "Spread_pp:Q",
                           title="Spread (percentage points)",
                           scale=alt.Scale(zero=False)
                       ),
                       color=alt.Color("Series:N", legend=alt.Legend(title=None)),
                       tooltip=[
                           alt.Tooltip("Date:T"),
                           alt.Tooltip("Series:N"),
                           alt.Tooltip("Spread_pp:Q", title="Spread", format=",.3f"),
                       ],
                   ),
                zero_line
            )
            .properties(height=260)   # <-- plus de title Altair ici
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)

        # Source en caption + lien FRED (demandé)
        st.caption(
            "Spreads: US−FR = US 10Y − France 10Y ; FR−DE = France 10Y − Germany 10Y. "
            "Source: OECD/MEI via FRED — US: IRLTLT01USM156N, FR: IRLTLT01FRM156N, DE: IRLTLT01DEM156N. "
            "Voir FRED (US 10Y): https://fred.stlouisfed.org/series/IRLTLT01USM156N#:~:text=Observations"
        )
        # # FRED source link (as requested):
        # https://fred.stlouisfed.org/series/IRLTLT01USM156N#:~:text=Observations

        # Tableau QA enrichi
        with st.expander("Latest observations (QA)"):
            qa_cols = ["Date", "US_10Y", "FR_10Y", "DE_10Y", "spread_pp_FR_vs_US", "spread_pp_FR_vs_DE"]
            st.dataframe(df10[qa_cols].tail(24).set_index("Date"), use_container_width=True)

    except Exception as e:
        st.error(f"US–FR / FR–DE 10Y spreads error: {e}")

    # === END UPDATED ===


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

# =======================================================
# Deal Watch
# =======================================================

def render_deal_watch():
    from datetime import datetime  # utilisé pour l'affichage horaire local
    from html import unescape as _unescape

    st.markdown("#### Deal Tracker")
    st.caption("Auto-pulls bond issuance press releases via RSS (editable), filters by keywords, and shows latest issuers.")

    # --- 6 deal sources par défaut (éditables) ---
    deal_sources = {
        "GlobeNewswire — Prospectus/Announcement": "https://www.globenewswire.com/RssFeed/subjectcode/63-Prospectus%202fAnnouncement%20Of%20Prospectus/feedTitle/GlobeNewswire%20-%20Prospectus%2C%20Announcement%20Of%20Prospectus",
        "GlobeNewswire — Press Releases": "https://www.globenewswire.com/RssFeed/subjectcode/72-Press%20Releases/feedTitle/GlobeNewswire%20-%20Press%20Releases",
        "PR Newswire — All": "https://www.prnewswire.com/rss/news-releases-list.rss",
        "PR Newswire — Financial Services": "https://www.prnewswire.com/rss/financial-services/all-financial-services-news.rss",
        "Euronext — News": "https://live.euronext.com/en/rss_feed/news",
        "London Stock Exchange — RNS (All)": "https://www.londonstockexchange.com/exchange/feeds/rss/news.xml",
    }
    # --- Filtres (cachés) ---
    default_kw = r"bond|notes|covered|green bond|sustainability|subordinated|hybrid|AT1|Tier 2|senior|convertible|syndicated|benchmark|tap|midswap|spread|coupon|maturity|issuance|new issue|priced|pricing|launch"
    kw = default_kw  # pas d'UI

    limit = st.number_input("Items per feed", min_value=5, max_value=50, value=15, step=5)

    urls = list(deal_sources.values())
    df = fetch_rss(urls, limit_per_feed=int(limit)) if urls else pd.DataFrame(columns=["source", "title", "link", "published"])

    # ---------- Banques : dictionnaire de synonymes -> libellé canonique ----------
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
    }
    # mapping "synonyme -> canonique"
    ALT_TO_CANON = {}
    for canon, alts in BANK_SYNONYMS.items():
        for a in alts:
            ALT_TO_CANON[a.lower()] = canon

    # Regex robuste: mots entiers + espaces variables, évite "ing" dans "pricing"
    def _alts_to_regex_parts(alts: List[str]) -> List[str]:
        parts = []
        for a in alts:
            esc = re.escape(a)
            esc = esc.replace(r"\ ", r"\s+")  # tolère espaces multiples
            parts.append(esc)
        return parts
    BANK_PATTERN = re.compile(
        r"(?<![A-Za-z])(?:%s)(?![A-Za-z])" % "|".join(
            _alts_to_regex_parts([k for sub in BANK_SYNONYMS.values() for k in sub])
        ),
        re.IGNORECASE,
    )

    def _extract_banks_from_text(text: str) -> list[str]:
        hits = []
        for m in BANK_PATTERN.finditer(text):
            canon = ALT_TO_CANON.get(m.group(0).lower())
            if canon:
                hits.append(canon)
        # unique en conservant l'ordre d'apparition
        return list(dict.fromkeys(hits))

    # Récupération du HTML de l'article et conversion en texte brut
    if "deal_article_cache" not in st.session_state:
        st.session_state.deal_article_cache = {}
    def _fetch_article_plaintext(url: str) -> str:
        # cache simple
        cache = st.session_state.deal_article_cache
        if url in cache:
            return cache[url]
        if 'requests' in globals() and requests is not None:
            try:
                hdr = {"User-Agent": "Mozilla/5.0 (DealTracker/1.0)"}
                r = requests.get(url, timeout=8, headers=hdr)
                r.raise_for_status()
                html = r.text
                # retire scripts/styles puis tags
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

    def _extract_issuer(title: str) -> str | None:
        t = re.sub(r"\s+", " ", title).strip()
        # supprime les tickers entre parenthèses
        t = re.sub(r"\((?:NASDAQ|NYSE|EPA|LSE|TSX|SIX|FWB|XETRA|HKEX|ASX|BME)[^)]*\)", "", t, flags=re.IGNORECASE)
        m = re.search(r"^(.*?)(?:\s+)(announces|prices?|launches|issues?|offers?|files|to issue|to offer)\b", t, flags=re.IGNORECASE)
        cand = m.group(1).strip(" -—:;,.") if m else None
        if not cand:
            cand = t.split(":")[0].split("—")[0].split("-")[0].strip()
        if cand and len(cand.split()) <= 8 and not cand.lower().startswith(("press release", "company", "the ")):
            return cand
        return None

    def _join_and(names: list[str]) -> str:
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + " and " + names[-1]

    if not df.empty:
        # Filtre mots-clés
        mask = df["title"].str.contains(kw, case=False, na=False, regex=True)
        df = df[mask].copy()

        df["Time"] = df["published"].dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d %H:%M")
        df["Issuer"] = df["title"].apply(_extract_issuer)

        # 1) banques depuis le titre (souvent vide)
        df["Banks_title"] = df["title"].apply(_extract_banks_from_text)

        # 2) banques depuis l'article (limite de requêtes pour rester léger)
        banks_from_page = []
        max_fetch = 12  # on limite le nombre de pages à récupérer
        fetched = 0
        for _, row in df.sort_values("published", ascending=False).iterrows():
            if fetched >= max_fetch:
                banks_from_page.append([])
                continue
            text = _fetch_article_plaintext(row.get("link", ""))
            banks_from_page.append(_extract_banks_from_text(text))
            fetched += 1
        # réaligne sur l'index courant
        df_sorted = df.sort_values("published", ascending=False).copy()
        df_sorted["Banks_page"] = banks_from_page
        df = df_sorted.sort_index()

        # Fusionne banques titre + page
        def _merge_banks(row) -> list[str]:
            seen = {}
            for b in (row.get("Banks_title", []) or []) + (row.get("Banks_page", []) or []):
                seen[b] = True
            # tri alpha pour stabilité
            return sorted(seen.keys(), key=lambda x: x.lower())

        df["Banks"] = df.apply(_merge_banks, axis=1)

        # Vue table cliquable (conserve l'existant)
        view = df[["Time", "source", "title", "link"]].rename(columns={"source": "Source", "title": "Title", "link": "Link"})
        _render_clickable_table(view)

        # ---------- Nouveau : panneau des derniers émetteurs + banques détectées ----------
        issuers_lines = []
        for _, row in df.sort_values("published", ascending=False).iterrows():
            issuer = row.get("Issuer")
            banks = row.get("Banks") or []
            if issuer:
                if banks:
                    issuers_lines.append(f"{issuer} ({_join_and(banks)})")
                else:
                    issuers_lines.append(issuer)
        issuers_lines = list(dict.fromkeys(issuers_lines))  # unique, ordre conservé

        if issuers_lines:
            st.markdown("**Latest issuers detected**")
            st.markdown("- " + "\n- ".join(issuers_lines[:25]))
        else:
            st.info("No clear issuer names detected in headlines (try editing sources or keywords).")

        # (toujours pas de bouton CSV)
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
