"""
Streamlit UI for the Toronto Health-Tech Ecosystem RAG demo.

Run:
    streamlit run src/app.py

Visual design: University of Toronto navy on white, medical/academic feel.
Locked to light theme via .streamlit/config.toml.

The UI exposes the internals of the RAG pipeline so the demo viewer can see
WHY a particular answer was given — every claim is traced back to a retrieved
chunk, every chunk shows its freshness date. This implements all four pillars
from Ng, Matsuba & Zhang (NEJM AI, 2024): recency, private data, traceability,
accuracy.
"""

import re
from pathlib import Path

import streamlit as st

from retriever import Retriever
from generator import generate
from ingest import ingest as build_index, CHROMA_DIR


# =========================================================================
# Auto-build the index on first run (e.g. fresh deploy on Streamlit Cloud).
# Locally this is a no-op because chroma_store/ already exists from when
# you ran `python src/ingest.py`. On a cloud deploy the store is gitignored,
# so we build it the first time the app boots.
# =========================================================================

@st.cache_resource
def _bootstrap_index():
    if not Path(CHROMA_DIR).exists() or not any(Path(CHROMA_DIR).iterdir()):
        with st.spinner("First-time setup — building the ecosystem index…"):
            build_index()
    return True


_bootstrap_index()


# =========================================================================
# Page config
# =========================================================================

st.set_page_config(
    page_title="Toronto Health-Tech Ecosystem RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================
# U of T palette — defined once, referenced everywhere
#
#   --uoft-navy:       #002A5C   primary brand
#   --uoft-navy-deep:  #001D40   hero gradient end
#   --uoft-blue:       #1A4480   secondary accent
#   --uoft-sky:        #0077C8   links + highlights
#   --surface:         #FFFFFF   page background
#   --surface-soft:    #F4F7FB   cards, panels
#   --surface-line:    #DEE5EF   borders
#   --ink:             #1A2332   primary text
#   --ink-soft:        #5A6B7C   secondary text
#   --ink-muted:       #8898A8   captions, labels
#   --medical-green:   #2E7D5C   verified pill
#   --medical-amber:   #B8860B   offline banner
# =========================================================================

st.markdown(
    """
    <style>
    /* =================================================================
       NAVY PALETTE
       --bg              #001A38   page background (deep navy)
       --bg-card         #042851   raised surface (cards, hero)
       --bg-soft         #07315F   subtle surface
       --bg-input        #051F3F   input fields
       --border          rgba(255,255,255,0.08)
       --border-strong   rgba(255,255,255,0.16)
       --text            #F2F6FB   primary text
       --text-soft       #BDD4ED   secondary text
       --text-muted      #7E9AC0   captions
       --accent          #5FB0E8   sky blue (interactive)
       --accent-soft     #3D89C2
       --verified        #5FD49A   medical green
       --amber           #F4C26B   offline banner
       --error           #FF8585
    ================================================================= */

    /* ---- Global ---------------------------------------------------- */
    html, body, [class*="css"], .stApp {
        background-color: #001A38 !important;
        color: #F2F6FB !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    }
    header[data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer { visibility: hidden; }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1080px;
    }

    h1, h2, h3, h4, h5, p, span, label, div { color: #F2F6FB; }
    p, label { color: #BDD4ED; }

    /* ---- Hero — raised navy card with subtle inner glow ------------ */
    .hero {
        display: flex;
        align-items: center;
        gap: 0.95rem;
        padding: 0.95rem 1.25rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #042851 0%, #053068 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25),
                    inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    .hero-mark {
        width: 46px;
        height: 46px;
        border-radius: 10px;
        background: linear-gradient(135deg, #5FB0E8 0%, #3D89C2 100%);
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        font-weight: 700;
        flex-shrink: 0;
        letter-spacing: -0.02em;
        box-shadow: 0 2px 8px rgba(95, 176, 232, 0.35);
    }
    .hero-body { flex: 1; min-width: 0; }
    .hero-eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.76rem;
        font-weight: 600;
        color: #7E9AC0 !important;
        margin-bottom: 0.3rem;
    }
    .hero-title {
        font-size: 1.4rem;
        font-weight: 700;
        line-height: 1.25;
        margin: 0;
        color: #FFFFFF !important;
        letter-spacing: -0.01em;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #BDD4ED !important;
        line-height: 1.5;
        margin-top: 0.35rem;
    }
    .hero-citation {
        font-size: 0.85rem;
        color: #7E9AC0 !important;
        font-style: italic;
        white-space: nowrap;
        flex-shrink: 0;
        padding-left: 1.1rem;
        border-left: 1px solid rgba(255, 255, 255, 0.1);
        max-width: 280px;
        text-align: right;
    }
    @media (max-width: 900px) {
        .hero { flex-wrap: wrap; }
        .hero-citation {
            border-left: none;
            padding-left: 0;
            white-space: normal;
            text-align: left;
            max-width: none;
            width: 100%;
            margin-top: 0.4rem;
        }
    }

    /* ---- Section labels -------------------------------------------- */
    .section-label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.76rem;
        font-weight: 600;
        color: #7E9AC0 !important;
        margin: 0.9rem 0 0.45rem 0;
    }

    /* ---- Input field ----------------------------------------------- */
    div[data-baseweb="input"] > div,
    .stTextInput input {
        background-color: #051F3F !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 9px !important;
        color: #F2F6FB !important;
        font-size: 1.02rem !important;
        padding: 0.75rem 1rem !important;
    }
    .stTextInput input:focus,
    div[data-baseweb="input"]:focus-within > div {
        border-color: #5FB0E8 !important;
        box-shadow: 0 0 0 3px rgba(95, 176, 232, 0.15) !important;
    }
    .stTextInput input::placeholder {
        color: #7E9AC0 !important;
        font-weight: 400;
    }

    /* ---- Example chips (Streamlit buttons) ------------------------- */
    div[data-testid="stButton"] {
        margin-bottom: 0.4rem !important;
    }
    div[data-testid="stButton"] > button {
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        text-align: left !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 0 !important;
        padding: 0.7rem 1rem !important;
        line-height: 1.4 !important;
        background: #042851 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 9px !important;
        color: #BDD4ED !important;
        font-weight: 400 !important;
        font-size: 0.95rem !important;
        transition: all 140ms ease !important;
        box-shadow: none !important;
        position: relative !important;
    }
    div[data-testid="stButton"] > button::after {
        content: "→";
        margin-left: auto;
        padding-left: 0.7rem;
        color: #4A6A8E;
        font-size: 0.95rem;
        transition: color 140ms ease, transform 140ms ease;
        flex-shrink: 0;
    }
    div[data-testid="stButton"] > button:hover::after {
        color: #5FB0E8;
        transform: translateX(2px);
    }
    div[data-testid="stButton"] > button > *,
    div[data-testid="stButton"] > button div,
    div[data-testid="stButton"] > button p,
    div[data-testid="stButton"] > button span,
    div[data-testid="stButton"] > button [data-testid="stMarkdownContainer"] {
        text-align: left !important;
        justify-content: flex-start !important;
        margin: 0 !important;
        padding: 0 !important;
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
        color: inherit !important;
        font-weight: 400 !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: #07315F !important;
        border-color: rgba(95, 176, 232, 0.4) !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stButton"] > button:hover * {
        color: #FFFFFF !important;
    }
    div[data-testid="stButton"] > button:focus,
    div[data-testid="stButton"] > button:focus:not(:active) {
        border-color: #5FB0E8 !important;
        box-shadow: 0 0 0 2px rgba(95, 176, 232, 0.18) !important;
        color: #FFFFFF !important;
        outline: none !important;
    }

    /* ---- How-it-works card ----------------------------------------- */
    .info-card {
        padding: 0.95rem 1.15rem;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.95rem;
        line-height: 1.6;
        color: #BDD4ED;
    }
    .info-card strong { color: #FFFFFF; font-weight: 600; }

    /* ---- Examples header ------------------------------------------- */
    .examples-header {
        font-size: 0.92rem;
        font-weight: 600;
        color: #BDD4ED;
        margin: 1rem 0 0.6rem 0;
        letter-spacing: -0.005em;
    }

    /* ---- How-it-works footer caption ------------------------------- */
    .how-it-works {
        margin-top: 1.35rem;
        padding: 0.7rem 0.95rem;
        font-size: 0.9rem;
        color: #7E9AC0;
        background: transparent;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
        line-height: 1.6;
    }
    .how-it-works strong { color: #FFFFFF; font-weight: 600; }
    .how-dot {
        color: #5FB0E8;
        font-size: 0.6rem;
        margin-right: 0.35rem;
        vertical-align: middle;
    }

    /* ---- Mode banners ---------------------------------------------- */
    .banner {
        padding: 0.85rem 1.1rem;
        border-radius: 10px;
        font-size: 0.9rem;
        margin-bottom: 1.1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        line-height: 1.5;
    }
    .banner-offline {
        background: rgba(244, 194, 107, 0.1);
        border: 1px solid rgba(244, 194, 107, 0.3);
        border-left: 4px solid #F4C26B;
        color: #F4C26B;
    }
    .banner-live {
        background: rgba(95, 212, 154, 0.1);
        border: 1px solid rgba(95, 212, 154, 0.3);
        border-left: 4px solid #5FD49A;
        color: #5FD49A;
    }
    .banner-error {
        background: rgba(255, 133, 133, 0.1);
        border: 1px solid rgba(255, 133, 133, 0.3);
        border-left: 4px solid #FF8585;
        color: #FF8585;
    }
    .banner-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: currentColor;
        flex-shrink: 0;
        margin-top: 0.45rem;
    }
    .banner code {
        background: rgba(255, 255, 255, 0.08);
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #5FB0E8;
    }

    /* ---- Program cards --------------------------------------------- */
    .program-card {
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        background: #042851;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #5FB0E8;
        margin-bottom: 0.85rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: all 150ms ease;
    }
    .program-card:hover {
        border-left-color: #5FD49A;
        background: #07315F;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    .program-card-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin-bottom: 0.35rem;
        gap: 0.75rem;
    }
    .program-rank {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7E9AC0;
        font-weight: 700;
    }
    .program-score {
        font-size: 0.86rem;
        color: #7E9AC0;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
    }
    .program-name {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        line-height: 1.25;
        color: #FFFFFF;
    }
    .program-org {
        font-size: 0.96rem;
        color: #BDD4ED;
        margin-bottom: 0.85rem;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.85rem;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        line-height: 1.3;
    }
    .badge-stage {
        background: rgba(95, 176, 232, 0.15);
        color: #9CCBF0;
        border: 1px solid rgba(95, 176, 232, 0.3);
    }
    .badge-sector {
        background: rgba(157, 200, 240, 0.1);
        color: #BDD4ED;
        border: 1px solid rgba(157, 200, 240, 0.22);
    }
    .badge-funding {
        background: rgba(255, 255, 255, 0.06);
        color: #BDD4ED;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .badge-verified {
        background: rgba(95, 212, 154, 0.12);
        color: #5FD49A;
        border: 1px solid rgba(95, 212, 154, 0.3);
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.8rem;
    }
    .program-desc {
        font-size: 1rem;
        line-height: 1.65;
        margin-bottom: 0.9rem;
        color: #DCE6F1;
    }
    .program-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
        font-size: 0.92rem;
    }
    .program-footer a {
        color: #5FB0E8;
        text-decoration: none;
        font-weight: 500;
    }
    .program-footer a:hover {
        text-decoration: underline;
        color: #9CCBF0;
    }
    .program-footer-meta {
        color: #7E9AC0;
        font-size: 0.86rem;
    }

    /* ---- Inline citation pills (LLM answers) ----------------------- */
    .citation-pill {
        display: inline-block;
        padding: 0.1rem 0.55rem;
        border-radius: 6px;
        background: rgba(95, 176, 232, 0.15);
        color: #5FB0E8;
        font-size: 0.86rem;
        font-weight: 600;
        margin: 0 2px;
        border: 1px solid rgba(95, 176, 232, 0.3);
    }

    /* ---- Metadata grid in chunk expanders -------------------------- */
    .meta-grid {
        display: grid;
        grid-template-columns: max-content 1fr;
        column-gap: 1.2rem;
        row-gap: 0.45rem;
        font-size: 0.95rem;
        padding: 0.35rem 0 0.55rem;
    }
    .meta-key {
        color: #7E9AC0;
        font-weight: 600;
    }
    .meta-val { color: #F2F6FB; }
    .meta-val code {
        background: rgba(255, 255, 255, 0.06);
        color: #5FB0E8;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .meta-val a { color: #5FB0E8; text-decoration: none; }
    .meta-val a:hover { text-decoration: underline; color: #9CCBF0; }

    /* ---- Streamlit expander styling -------------------------------- */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        background: #042851 !important;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600;
        color: #BDD4ED !important;
    }
    div[data-testid="stExpander"] summary:hover {
        background: rgba(255, 255, 255, 0.03);
    }
    div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: #DCE6F1 !important;
    }

    /* ---- Sidebar styling ------------------------------------------- */
    section[data-testid="stSidebar"] {
        background: #001530 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] * { color: #F2F6FB; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #BDD4ED !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #7E9AC0 !important;
    }
    /* Sidebar select boxes */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #042851 !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: #F2F6FB !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        font-size: 0.9rem;
    }

    /* ---- Slider color ---------------------------------------------- */
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
        background: #5FB0E8 !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background: #5FB0E8 !important;
        border-color: #5FB0E8 !important;
    }

    /* ---- Captions -------------------------------------------------- */
    [data-testid="stCaptionContainer"], .stCaption {
        color: #7E9AC0 !important;
        font-size: 0.9rem;
    }

    /* ---- Spinners -------------------------------------------------- */
    .stSpinner > div { border-top-color: #5FB0E8 !important; }

    /* ---- Footer ---------------------------------------------------- */
    .footer-caption {
        text-align: center;
        font-size: 0.88rem;
        color: #7E9AC0;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    .footer-caption code {
        background: rgba(255, 255, 255, 0.05);
        color: #5FB0E8;
        padding: 0.12rem 0.45rem;
        border-radius: 4px;
        font-size: 0.84rem;
    }

    /* ---- Inline code in markdown ----------------------------------- */
    .stMarkdown code {
        background: rgba(255, 255, 255, 0.06) !important;
        color: #5FB0E8 !important;
        padding: 0.1rem 0.4rem !important;
        border-radius: 4px !important;
    }

    /* ---- Empty state (off-topic query) ----------------------------- */
    .empty-state {
        padding: 2rem 2.25rem;
        border-radius: 12px;
        background: #042851;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #F4C26B;
        text-align: left;
        margin-top: 0.5rem;
    }
    .empty-state-icon {
        font-size: 1.6rem;
        color: #F4C26B;
        margin-bottom: 0.6rem;
        line-height: 1;
    }
    .empty-state-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }
    .empty-state-body {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #BDD4ED;
        max-width: 720px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# Session state — for clickable example chips
# =========================================================================

if "query_input" not in st.session_state:
    st.session_state.query_input = ""


def set_example(text: str):
    st.session_state.query_input = text


# =========================================================================
# Cached retriever
# =========================================================================

@st.cache_resource
def get_retriever():
    return Retriever()


# =========================================================================
# Hero
# =========================================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-mark">T</div>
        <div class="hero-body">
            <div class="hero-eyebrow">Temerty Faculty of Medicine · Health Nexus MVP</div>
            <div class="hero-title">Toronto Health-Tech Ecosystem RAG</div>
            <div class="hero-subtitle">
                A grounded, citation-first assistant for navigating Toronto&rsquo;s
                health-tech funding landscape.
            </div>
        </div>
        <div class="hero-citation">
            Architecture per Ng, Matsuba &amp; Zhang<br>NEJM AI, 2024
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# Sidebar controls
# =========================================================================

with st.sidebar:
    st.markdown("### Retrieval settings")

    mode = st.radio(
        "Retrieval mode",
        ["Naïve (vector only)", "Hybrid (vector + BM25 + metadata)"],
        index=1,
        help=(
            "Naïve = pure cosine similarity. "
            "Hybrid = dense + BM25 fused with Reciprocal Rank Fusion, with "
            "optional metadata filtering. Zhang's NEJM AI paper recommends "
            "hybrid over pure vector for complex regulated information — "
            "entrepreneurship data has the same structural complexity."
        ),
    )
    is_hybrid = mode.startswith("Hybrid")

    top_k = st.slider("Top-k retrieved chunks", 1, 6, 3)

    stage_filter = None
    sector_filter = None
    if is_hybrid:
        st.markdown("#### Metadata filters")
        st.caption("Optional — leave as *Any* to skip.")
        stage_filter = st.selectbox(
            "Stage",
            [None, "idea", "pre-seed", "seed", "Series A", "growth"],
            format_func=lambda x: "Any" if x is None else x,
        )
        sector_filter = st.selectbox(
            "Sector",
            [
                None,
                "digital health",
                "medical devices",
                "biotech",
                "regenerative medicine",
            ],
            format_func=lambda x: "Any" if x is None else x,
        )

    st.divider()
    st.markdown("### About")
    st.caption(
        "Built for the Temerty Health Nexus Chair undergraduate research "
        "interview. The retriever and generator are intentionally separable "
        "so the comparison between Naïve and Hybrid retrieval can be measured "
        "directly."
    )
    st.caption("**Stack:** ChromaDB · MiniLM embeddings · BM25 · Streamlit")


# =========================================================================
# Query input + example chips
# =========================================================================

query = st.text_input(
    "Your question",
    placeholder="Ask the ecosystem — e.g., I'm a clinician with a digital health idea, where do I start?",
    label_visibility="collapsed",
    key="query_input",
)

if not query:
    st.markdown(
        '<div class="examples-header">Suggested questions</div>',
        unsafe_allow_html=True,
    )

    examples = [
        "I'm a U of T med student with a digital health idea. Where do I start?",
        "Where can I get non-dilutive funding for a Canadian health-tech R&D project?",
        "I'm building a cell therapy company. Who can help with manufacturing?",
        "Wet lab space without giving up equity?",
        "I need an accelerator that takes pre-seed digital health founders.",
        "MaRS vs. JLABS for an early-stage device company?",
    ]

    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            st.button(
                ex,
                key=f"ex_{i}",
                on_click=set_example,
                args=(ex,),
                use_container_width=True,
            )

    st.markdown(
        """
        <div class="how-it-works">
            <span class="how-dot">●</span>
            Questions are embedded and matched against a curated index of 12
            Toronto health-tech programs. Toggle <strong>Naïve</strong> vs.
            <strong>Hybrid</strong> retrieval in the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# Helpers
# =========================================================================

# Match [Program Name] citations but NOT markdown links like [text](url)
CITATION_RE = re.compile(r"(?<!\]) \[([^\[\]]+?)\](?!\()", flags=re.VERBOSE)


def stylize_citations(text: str) -> str:
    """Wrap [Program Name] tokens in a navy pill. Preserve markdown links."""
    return CITATION_RE.sub(
        lambda m: f'<span class="citation-pill">{m.group(1)}</span>',
        text,
    )


def render_program_card(hit: dict, rank: int):
    m = hit["metadata"]
    desc_preview = (hit["document"][:240] + "…") if len(hit["document"]) > 240 else hit["document"]
    eligibility = ""
    for line in hit["document"].splitlines():
        low = line.lower()
        if low.startswith("eligibility") or low.startswith("who it's for"):
            eligibility = line.split(":", 1)[-1].strip()
            break
    body = eligibility if eligibility else desc_preview

    st.markdown(
        f"""
        <div class="program-card">
            <div class="program-card-header">
                <span class="program-rank">Match #{rank}</span>
                <span class="program-score">score · {hit['score']:.3f}</span>
            </div>
            <div class="program-name">{m['name']}</div>
            <div class="program-org">{m['organization']}</div>
            <div class="badge-row">
                <span class="badge badge-stage">{m['stage']}</span>
                <span class="badge badge-sector">{m['sector']}</span>
                <span class="badge badge-funding">{m['funding_type']}</span>
                <span class="badge badge-verified">✓ verified {m['last_verified']}</span>
            </div>
            <div class="program-desc">{body}</div>
            <div class="program-footer">
                <a href="{m['url']}" target="_blank">↗ {m['url']}</a>
                <span class="program-footer-meta">{m['funding_type']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mode_banner(mode_str: str):
    if mode_str == "anthropic_claude":
        st.markdown(
            '<div class="banner banner-live"><span class="banner-dot"></span>'
            "<div><strong>Live mode.</strong>&nbsp;Answer generated by Claude, "
            "grounded in the retrieved context.</div></div>",
            unsafe_allow_html=True,
        )
    elif mode_str == "fallback_after_error":
        st.markdown(
            '<div class="banner banner-error"><span class="banner-dot"></span>'
            "<div><strong>API error — showing retrieval output.</strong>&nbsp;"
            "The retrieval pipeline (the part being researched) is unaffected.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="banner banner-offline"><span class="banner-dot"></span>'
            "<div><strong>Offline mode.</strong>&nbsp;No API key configured. "
            "The retrieval layer — the focus of the comparison — runs "
            "identically with or without a generator. Set "
            "<code>ANTHROPIC_API_KEY</code> in <code>.env</code> to enable "
            "live generation.</div></div>",
            unsafe_allow_html=True,
        )


# =========================================================================
# Main interaction
# =========================================================================

if query:
    retriever = get_retriever()

    with st.spinner("Retrieving relevant programs…"):
        if is_hybrid:
            hits = retriever.retrieve_hybrid(
                query,
                k=top_k,
                stage_filter=stage_filter,
                sector_filter=sector_filter,
            )
        else:
            hits = retriever.retrieve_naive(query, k=top_k)

    if not hits:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">○</div>
                <div class="empty-state-title">Nothing in the corpus matches this query.</div>
                <div class="empty-state-body">
                    The system refused to answer because the closest indexed
                    program scored below the relevance threshold
                    (cosine similarity &lt; 0.15). This is deliberate — naive
                    RAG would have returned the least-irrelevant chunk and
                    written confident nonsense. Try a question about Toronto
                    health-tech programs, accelerators, grants, or incubators.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Generating grounded answer…"):
            result = generate(query, hits)

        col_answer, col_chunks = st.columns([3, 2], gap="large")

        with col_answer:
            st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
            render_mode_banner(result["mode"])

            if result["mode"] == "anthropic_claude":
                styled = stylize_citations(result["answer"])
                st.markdown(styled, unsafe_allow_html=True)
            else:
                for i, hit in enumerate(hits, 1):
                    render_program_card(hit, i)

        with col_chunks:
            st.markdown(
                f'<div class="section-label">Retrieved chunks · {len(hits)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Every claim above is grounded in these records. "
                "This panel exposes the traceability pillar from Zhang's framework."
            )
            for i, hit in enumerate(hits, 1):
                m = hit["metadata"]
                with st.expander(
                    f"{i}. {m['name']}  ·  score {hit['score']:.3f}",
                    expanded=(i == 1),
                ):
                    st.markdown(
                        f"""
                        <div class="meta-grid">
                            <div class="meta-key">Organization</div>
                            <div class="meta-val">{m['organization']}</div>
                            <div class="meta-key">URL</div>
                            <div class="meta-val"><a href="{m['url']}" target="_blank">{m['url']}</a></div>
                            <div class="meta-key">Stage</div>
                            <div class="meta-val">{m['stage']}</div>
                            <div class="meta-key">Sector</div>
                            <div class="meta-val">{m['sector']}</div>
                            <div class="meta-key">Funding type</div>
                            <div class="meta-val">{m['funding_type']}</div>
                            <div class="meta-key">Last verified</div>
                            <div class="meta-val"><code>{m['last_verified']}</code></div>
                            <div class="meta-key">Source note</div>
                            <div class="meta-val">{m['source_note']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Indexed text**")
                    st.text(hit["document"])

        st.divider()
        with st.expander("Pipeline trace — what just happened", expanded=False):
            filters_str = (
                f"stage=`{stage_filter}`, sector=`{sector_filter}`"
                if is_hybrid and (stage_filter or sector_filter)
                else "none"
            )
            st.markdown(
                f"""
**1 · Document Ingestion** *(once, at index-build time)*
- 12 programs from `data/seed_programs.json` chunked using structure-aware chunking (one program = one chunk, preserving program-name → eligibility → deadlines as one unit).
- Each chunk embedded with ChromaDB's MiniLM default model.
- Metadata (stage, sector, funding_type, last_verified, source_note) stored alongside each vector for filtering and freshness tracking.

**2 · Retriever** *(mode: `{mode}`)*
- Your query was embedded and compared against the index.
- {"Hybrid mode also ran BM25 over the same corpus and fused both rankings with Reciprocal Rank Fusion (RRF, k=60)." if is_hybrid else "Naïve mode used cosine similarity only — pure dense vector search, no sparse signal."}
- Metadata filters: {filters_str}
- Top-{top_k} chunks passed to the generator.

**3 · Generator** *(mode: `{result['mode']}`)*
- The prompt instructed the model to ground every claim in the retrieved chunks, cite each claim inline with the program name, flag time-sensitive claims with a "verify on website" note, and refuse to guess when context is missing.
- This implements all four of Zhang's pillars: **recency** (verified dates), **private data** (curated corpus), **traceability** (citations), and **accuracy** (anti-hallucination prompt).
                """
            )

