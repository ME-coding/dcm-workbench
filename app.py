import streamlit as st
import importlib
from pathlib import Path
import base64

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
      --border:rgba(255,255,255,.18);
      --text-muted:rgba(255,255,255,.72);
    }

    .main .block-container{max-width:1200px;}

    .hero{
      width:100%;
      height:220px;
      border-radius:18px;
      margin:.5rem 0 1.25rem 0;
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

    .card{ border:2px solid var(--border); border-radius:16px; padding:16px; background:var(--surface); }
    .section-head{ font-weight:700; font-size:1.25rem; margin-bottom:.5rem; }
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
section = st.sidebar.radio(
    "Go to:", _options, index=_default_idx,
    key=f"nav_radio_{st.session_state['nav_target'].replace(' ', '_')}",
    label_visibility="collapsed"
)
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

# ---------- Hero helpers ----------
def _render_hero():
    if hero_path.exists():
        b = hero_path.read_bytes()
        src = "data:image/jpg;base64," + base64.b64encode(b).decode("utf-8")
        st.markdown(f'<div class="hero" style="background-image:url({src});"></div>', unsafe_allow_html=True)

def _render_hero_from(path: Path):
    if path.exists():
        b = path.read_bytes()
        src = "data:image/jpg;base64," + base64.b64encode(b).decode("utf-8")
        st.markdown(f'<div class="hero" style="background-image:url({src});"></div>', unsafe_allow_html=True)

# ---------- Routing ----------
if section == "Home Page":
    _render_hero()

    st.title("Debt Capital Markets Workbench")
    st.markdown(
        '<div class="subtitle">Your all-in-one platform for Debt Capital Markets analytics & execution · End-of-year Project – Paris I Panthéon-Sorbonne</div>',
        unsafe_allow_html=True
    )

    # ---- Overview : plein largeur ----
    st.markdown("#### Overview")
    st.markdown(
        """
        <p class="justify overview-text">
        The Debt Capital Markets Workbench is a Streamlit application that consolidates core DCM workflows into a single interface.
        It supports the entire activity cycle of the profession, from bond pricing and analysis to monitoring the data room and the competitive
        DCM market, while also integrating learning tools to help newcomers understand. Below is a page-by-page explanation.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Page-by-page : deux colonnes ----
    st.markdown("#### Page-by-page")
    col_left, col_right = st.columns(2, vertical_alignment="top")

    with col_left:
        st.markdown("**Structuring Desk**")
        st.markdown(
            """
- **Pricer**: A comprehensive bond pricing platform (vanilla bonds, zero-coupon bonds, convertibles, and sustainability-linked bonds). Simply input your parameters to access key metrics and tailored charts, accompanied by clear explanations.
- **Tools**: Two essential deal tools — a customizable term sheet generator (Word format) and a cash flow amortization schedule builder.
- **Glossary & Learn More**: Since this application is also aimed at newcomers to DCM, you’ll find a complete glossary, key formulas, and additional information on exotic bond types.
            """.strip()
        )

    with col_right:
        st.markdown("**Intelligence Desk**")
        st.markdown(
            """
- **AI Agent**: Explore the capabilities of an AI agent designed to support your data room. With active memory and advanced document-search functionalities, it is the ideal companion.
- **Latest News and Insights**: This final section of the application provides a synthesis of the latest news, macroeconomic data from central banks, and a tracker of recent DCM deals with the banks involved.
            """.strip()
        )

elif section == "Structuring Desk":
    _render_hero_from(hero_struct_path)
    st.title("Structuring Desk")
    tab_pricer, tab_tools, tab_glossary = st.tabs(["Pricer", "Tools", "Glossary & Learn More"])
    with tab_pricer:
        safe_tab("Structuring.pricer", "Structuring Desk — Pricer")
    with tab_tools:
        safe_tab("Structuring.tools", "Structuring Desk — Tools")
    with tab_glossary:
        safe_tab("Structuring.glossary", "Structuring Desk — Glossary & Learn More")

elif section == "Intelligence Desk":
    _render_hero_from(hero_intel_path)
    st.title("Intelligence Desk")
    tab_agent, tab_news = st.tabs(["AI Agent", "Latest News and Insights"])
    with tab_agent:
        safe_tab("page2.agent", "Intelligence Desk — AI Agent")
    with tab_news:
        safe_tab("page2.news", "Intelligence Desk — Latest News and Insights")
