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
hero_path = logo_dir / "background.jpg"                 # header Home
hero_struct_path = logo_dir / "header-structuring.jpg"  # header Structuring Desk
hero_intel_path = logo_dir / "header-intelligence.jpg"  # header Intelligence Desk

# ---------- CSS ----------
st.markdown(
    """
    <style>
    :root{
      --brand:#0b63c5;         /* bleu Sorbonne */
      --brand2:#1c2e6e;        /* bleu ESCP */
      --surface:rgba(255,255,255,.04);
      --border:rgba(255,255,255,.08);
      --text-muted:rgba(255,255,255,.72);
    }

    /* max width for readability */
    .main .block-container{max-width:1200px;}

    /* Hero banner */
    .hero{
      width:100%;
      height:220px;
      border-radius:18px;
      margin: .5rem 0 1.25rem 0;
      background-size:cover;
      background-position:center;
      position:relative;
      overflow:hidden;
      border:1px solid var(--border);
      box-shadow:0 10px 24px rgba(0,0,0,.18);
    }
    .hero::after{
      content:"";
      position:absolute; inset:0;
      background:linear-gradient(90deg, rgba(12,18,28,.55), rgba(11,99,197,.18));
    }

    .subtitle{ font-size:1.15rem; font-weight:500; margin-top:0.25rem; margin-bottom:1rem; }
    .justify{ text-align:justify; text-justify:inter-word; }

    /* Cards */
    .card{
      background:var(--surface);
      border:1px solid var(--border);
      border-radius:16px;
      padding:24px;
      box-shadow:0 10px 24px rgba(0,0,0,.12);
      backdrop-filter: blur(2px);
      margin-bottom:1rem;
    }
    .card h4{ margin:0 0 .5rem 0; }
    .muted{ color:var(--text-muted); font-size:.92rem; }

    /* Section headers inside columns */
    .section-head{ font-weight:700; font-size:1.05rem; margin-bottom:.25rem; }

    /* Pills: style buttons only when wrapped in .pillbox */
    .pillbox div[data-testid="stButton"] > button[kind="secondary"]{
      width:100%;
      text-align:left;
      border-radius:999px !important;
      border:1px solid var(--border) !important;
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

    /* Legacy linklike scope kept in case used elsewhere */
    .linklike div[data-testid="stButton"] > button[kind="secondary"]{
        width:100%; text-align:left; background:transparent!important; color:#0b63c5!important;
        border:none!important; box-shadow:none!important; padding-left:0!important; padding-right:0!important;
    }
    .linklike div[data-testid="stButton"] > button[kind="secondary"]:hover{ text-decoration:underline; background:transparent!important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.title("Navigation")

# Target section stored in session (NOT bound to any widget key)
if "nav_target" not in st.session_state:
    st.session_state["nav_target"] = "Home Page"

_options = ["Home Page", "Structuring Desk", "Intelligence Desk"]
_default_idx = _options.index(st.session_state["nav_target"])

# Radio with a DYNAMIC key so it re-inits when nav_target changes
_radio_key = f"nav_radio_{st.session_state['nav_target'].replace(' ', '_')}"
section = st.sidebar.radio("Go to:", _options, index=_default_idx, key=_radio_key, label_visibility="collapsed")

# Keep nav_target in sync with the user's manual click on the radio
if section != st.session_state["nav_target"]:
    st.session_state["nav_target"] = section

st.sidebar.markdown("---")
st.sidebar.markdown("by [Maxime Eneau](https://www.linkedin.com/in/maxime-eneau/)", help="Author")

# Two columns inside the sidebar (no captions)
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
    """Home banner from hero_path."""
    if not hero_path.exists():
        return
    b = hero_path.read_bytes()
    src = "data:image/jpg;base64," + base64.b64encode(b).decode("utf-8")
    st.markdown(f'<div class="hero" style="background-image:url({src});"></div>', unsafe_allow_html=True)

def _render_hero_from(path: Path):
    """Generic banner renderer for other sections."""
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
    # HERO
    _render_hero()

    # Title & subtitle
    st.title("Debt Capital Markets Workbench")
    st.markdown(
        '<div class="subtitle">Your all-in-one platform for Debt Capital Markets analytics & execution · Master’s Thesis – Paris I Panthéon-Sorbonne</div>',
        unsafe_allow_html=True
    )

    # Overview CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Overview")
    st.markdown(
        """
        <p class="justify">
        The Debt Capital Markets Workbench is a modular Streamlit application that consolidates core DCM workflows into a single, reproducible interface.
        It supports rapid pricing scenarios, primary & secondary market analytics, execution utilities, and learning integrations. The roadmap includes market
        data integrations (rates, credit curves, ESG frameworks), a copilot agent for repetitive tasks, and audit-ready exports to streamline transaction
        preparation, governance materials, and performance tracking.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Two desk CARDS
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Structuring Desk</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Pricing, analytics and utilities for primary DCM workflows.</div>', unsafe_allow_html=True)
        st.markdown('<div class="pillbox">', unsafe_allow_html=True)
        if st.button("Pricer", type="secondary", key="home_sd_pricer"):
            _goto("Structuring Desk", "Pricer")
        if st.button("Data Visualisation", type="secondary", key="home_sd_dataviz"):
            _goto("Structuring Desk", "Data Visualisation")
        if st.button("Tools", type="secondary", key="home_sd_tools"):
            _goto("Structuring Desk", "Tools")
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Intelligence Desk</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">AI agent and curated news & insights.</div>', unsafe_allow_html=True)
        st.markdown('<div class="pillbox">', unsafe_allow_html=True)
        if st.button("AI Agent", type="secondary", key="home_id_agent"):
            _goto("Intelligence Desk", "AI Agent")
        if st.button("Latest News and Insights", type="secondary", key="home_id_news"):
            _goto("Intelligence Desk", "Latest News and Insights")
        st.markdown('</div></div>', unsafe_allow_html=True)

    # What's inside CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### What’s inside (quick overview)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "- **Pricer**: multi-product pricing (fixed, FRN, callable/puttable, zero, convertibles, sukuk), duration/convexity, cash flows.\n"
            "- **Data Visualisation**: mid-swaps, iBoxx spreads, forward curves, sector trackers, correlations.\n"
            "- **Tools**: **Term Sheet Builder**, **Fees & Net Proceeds**, **Amortization Builder**."
        )
    with c2:
        st.markdown(
            "- **AI Agent**: chatbot with memory & local RAG on uploaded files (Mistral via secrets).\n"
            "- **Latest News and Insights**: RSS aggregator, rates dashboard, central banks feeds, manual deal watch."
        )
    st.markdown('</div>', unsafe_allow_html=True)

elif section == "Structuring Desk":
    # HERO spécifique Structuring Desk
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
    # HERO spécifique Intelligence Desk
    _render_hero_from(hero_intel_path)

    st.title("Intelligence Desk")
    tab_agent, tab_news = st.tabs(["AI Agent", "Latest News and Insights"])
    with tab_agent:
        safe_tab("page2.agent", "Intelligence Desk — AI Agent")
    with tab_news:
        safe_tab("page2.news", "Intelligence Desk — Latest News and Insights")
    _maybe_select_pending_tab()
