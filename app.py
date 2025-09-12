import streamlit as st
import importlib
from pathlib import Path
import base64  # <-- pour la bannière en background

# ---------- Page config ----------
st.set_page_config(
    page_title="Debt Capital Markets Workbench",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Paths (images) ----------
logo_dir = Path(__file__).parent / "Images"
escp_path = logo_dir / "image_escp.jpg"
sorbonne_path = logo_dir / "image_sorbonne.jpg"
hero_path = logo_dir / "background.jpg"
hero_struct_path = logo_dir / "header-structuring.jpg"
hero_intel_path = logo_dir / "header-intelligence.jpg"

# ---------- CSS ----------
st.markdown(
    """
    <style>
    :root{
      --brand:#0b63c5;
      --brand2:#1c2e6e;
      --surface:rgba(255,255,255,.04);
      --border:rgba(255,255,255,.18); /* plus marqué */
      --text-muted:rgba(255,255,255,.72);
    }

    .main .block-container{max-width:1200px;}

    .hero{
      width:100%;
      height:220px;
      border-radius:18px;
      margin: .5rem 0 1.25rem 0;
      background-size:cover;
      background-position:center;
      position:relative;
      overflow:hidden;
      border:2px solid var(--border);
      box-shadow:0 10px 24px rgba(0,0,0,.18);
    }
    .hero::after{
      content:"";
      position:absolute; inset:0;
      background:linear-gradient(90deg, rgba(12,18,28,.55), rgba(11,99,197,.18));
    }

    .subtitle{ font-size:1.15rem; font-weight:500; margin-top:0.25rem; margin-bottom:1rem; color:gray; }
    .justify{ text-align:justify; text-justify:inter-word; }
    .overview-text{ font-size:1.05rem; line-height:1.6; }

    .card h4{ margin:0 0 .5rem 0; }
    .muted{ color:var(--text-muted); font-size:.92rem; }

    .section-head{ font-weight:700; font-size:1.25rem; margin-bottom:.5rem; }

    .pillbox div[data-testid="stButton"] > button[kind="secondary"]{
      width:100%;
      text-align:left;
      border-radius:999px !important;
      border:1.5px solid var(--border) !important;
      background:rgba(11,99,197,.08) !important;
      color:#cfe4ff !important;
      box-shadow:none !important;
      padding:10px 16px !important;
      margin:.25rem 0;
      transition:transform .08s ease, background .12s ease, border-color .12s ease;
      font-weight:600;
    }
    .pillbox div[data-testid="stButton"] > button[kind="secondary"]::before{
      content:"›";
      display:inline-block;
      margin-right:6px;
      font-weight:700;
      color:#9ec6ff;
    }
    .pillbox div[data-testid="stButton"] > button[kind="secondary"]:hover{
      transform:translateY(-2px);
      text-decoration:none !Important;
      background:rgba(11,99,197,.14) !important;
      border-color:rgba(11,99,197,.40) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.title("Navigation")

if "nav_target" not in st.session_state:
    st.session_state["nav_target"] = "Home Page"

_options = ["Home Page", "Structuring Desk", "Intelligence Desk"]
_default_idx = _options.index(st.session_state["nav_target"])
_radio_key = f"nav_radio_{st.session_state['nav_target'].replace(' ', '_')}"
section = st.sidebar.radio("Go to:", _options, index=_default_idx, key=_radio_key, label_visibility="collapsed")

if section != st.session_state["nav_target"]:
    st.session_state["nav_target"] = section

st.sidebar.markdown("---")
st.sidebar.markdown("by [Maxime Eneau](https://www.linkedin.com/in/maxime-eneau/)", help="Author")

c1, c2 = st.sidebar.columns(2)
with c1:
    if escp_path.exists():
        st.image(str(escp_path), use_container_width=True)
with c2:
    if sorbonne_path.exists():
        st.image(str(sorbonne_path), use_container_width=True)

# ---------- Safe import helper ----------
def safe_tab(module_path: str, label: str):
    try:
        mod = importlib.import_module(module_path)
        getattr(mod, "render", lambda: st.info(f"{label} is empty.")).__call__()
    except Exception as e:
        st.warning(f"{label} – import error `{module_path}`: {e}")

# ---------- Helpers: hero banners ----------
def _render_hero():
    if not hero_path.exists():
        return
    b = hero_path.read_bytes()
    src = "data:image/jpg;base64," + base64.b64encode(b).decode("utf-8")
    st.markdown(f'<div class="hero" style="background-image:url({src});"></div>', unsafe_allow_html=True)

def _render_hero_from(path: Path):
    if not path.exists():
        return
    b = path.read_bytes()
    src = "data:image/jpg;base64," + base64.b64encode(b).decode("utf-8")
    st.markdown(f'<div class="hero" style="background-image:url({src});"></div>', unsafe_allow_html=True)

# ---------- Helper: programmatic navigation ----------
def _goto(target_section: str, target_tab: str | None = None):
    st.session_state["nav_target"] = target_section
    if target_tab:
        st.session_state["pending_tab"] = target_tab
    st.rerun()

def _maybe_select_pending_tab():
    target = st.session_state.pop("pending_tab", None)
    if not target:
        return
    st.markdown(
        f"""
        <script>
        const targetLabel = {target!r};
        let tries = 0;
        const clickTab = () => {{
            tries += 1;
            const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
            for (const t of tabs) {{
                if ((t.innerText || t.textContent).trim() === targetLabel) {{
                    t.click();
                    return;
                }}
            }}
            if (tries < 20) setTimeout(clickTab, 100);
        }};
        setTimeout(clickTab, 50);
        </script>
        """,
        unsafe_allow_html=True,
    )

# ---------- Routing ----------
if section == "Home Page":
    _render_hero()

    st.title("Debt Capital Markets Workbench")
    st.markdown(
        '<div class="subtitle">Your all-in-one platform for Debt Capital Markets analytics & execution · Master’s Thesis – Paris I Panthéon-Sorbonne</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Overview")
    st.markdown(
        """
        <p class="justify overview-text">
        The Debt Capital Markets Workbench is a modular Streamlit application that consolidates core DCM workflows into a single, reproducible interface.
        It supports rapid pricing scenarios, primary & secondary market analytics, execution utilities, and learning integrations. The roadmap includes market
        data integrations (rates, credit curves, ESG frameworks), a copilot agent for repetitive tasks, and audit-ready exports to streamline transaction
        preparation, governance materials, and performance tracking.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Two desk CONTAINERS côte à côte (même hauteur)
    col_left, col_right = st.columns(2, vertical_alignment="stretch")

    with col_left:
        with st.container(border=True):
            st.markdown('<div class="section-head">Structuring Desk</div>', unsafe_allow_html=True)
            st.markdown('<div class="muted">Pricing, analytics and utilities for primary DCM workflows.</div>', unsafe_allow_html=True)
            st.markdown('<div class="pillbox">', unsafe_allow_html=True)
            if st.button("Pricer", type="secondary", key="home_sd_pricer"):
                _goto("Structuring Desk", "Pricer")
            st.caption("Multi-product pricing tools and analytics")
            if st.button("Data Visualisation", type="secondary", key="home_sd_dataviz"):
                _goto("Structuring Desk", "Data Visualisation")
            st.caption("Market curves, spreads, and sector visualisations")
            if st.button("Tools", type="secondary", key="home_sd_tools"):
                _goto("Structuring Desk", "Tools")
            st.caption("Utilities for term sheets, fees, amortization")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="section-head">Intelligence Desk</div>', unsafe_allow_html=True)
            st.markdown('<div class="muted">AI agent and curated news & insights.</div>', unsafe_allow_html=True)
            st.markdown('<div class="pillbox">', unsafe_allow_html=True)
            if st.button("AI Agent", type="secondary", key="home_id_agent"):
                _goto("Intelligence Desk", "AI Agent")
            st.caption("Chatbot with memory and RAG on uploaded files")
            if st.button("Latest News and Insights", type="secondary", key="home_id_news"):
                _goto("Intelligence Desk", "Latest News and Insights")
            st.caption("Aggregated news, rates dashboards, deal watch")
            st.markdown('</div>', unsafe_allow_html=True)

elif section == "Structuring Desk":
    _render_hero_from(hero_struct_path)

    st.title("Structuring Desk")
    tab_pricer, tab_dataviz, tab_tools = st.tabs(["Pricer", "Data Visualisation", "Tools"])
    with tab_pricer:
        safe_tab("Structuring.pricer", "Structuring Desk — Pricer")
    with tab_dataviz:
        safe_tab("Structuring.dataviz", "Structuring Desk — Data Visualisation")
    with tab_tools:
        safe_tab("Structuring.tools", "Structuring Desk — Tools")
    _maybe_select_pending_tab()

elif section == "Intelligence Desk":
    _render_hero_from(hero_intel_path)

    st.title("Intelligence Desk")
    tab_agent, tab_news = st.tabs(["AI Agent", "Latest News and Insights"])
    with tab_agent:
        safe_tab("page2.agent", "Intelligence Desk — AI Agent")
    with tab_news:
        safe_tab("page2.news", "Intelligence Desk — Latest News and Insights")
    _maybe_select_pending_tab()
