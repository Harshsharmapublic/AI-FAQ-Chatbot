"""
app.py - AI-Powered FAQ Chatbot - Main Streamlit Application
=============================================================
Author: AI-Powered FAQ Chatbot System
Description:
    Main entry point for the Streamlit web application.
    Provides a professional, modern UI with four navigation tabs:
      1. 💬 Chat Portal     - AI-powered FAQ chat interface
      2. 📁 Document Index  - PDF upload and indexing
      3. 📊 Analytics       - Dashboard with usage metrics and charts
      4. ⚙️ FAQ Manager     - Admin panel for CRUD operations

Run:
    streamlit run app.py
"""

import os
import sys
import logging
import html as html_module
import textwrap
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows to handle unicode characters (₹, emojis, etc.)
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import streamlit as st

# ---------------------------------------------------------------------------
# Logging — deferred FileHandler (logs/ dir may not exist at import time)
# ---------------------------------------------------------------------------
Path("logs").mkdir(exist_ok=True)

_log_handlers = [logging.StreamHandler(sys.stdout)]
try:
    _fh = logging.FileHandler(Path("logs") / "app.log", mode="a", encoding="utf-8")
    _log_handlers.append(_fh)
except OSError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=_log_handlers,
)
logging.getLogger("streamlit").setLevel(logging.WARNING)
logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuraFAQ · AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "**NeuraFAQ** — AI-Powered FAQ Chatbot | Built with Python, Streamlit & NLP",
    },
)

# ---------------------------------------------------------------------------
# Imports (after page config)
# ---------------------------------------------------------------------------
from chatbot import FAQEngine
from faq_manager import FAQManager
from pdf_processor import PDFProcessor
from voice_handler import VoiceHandler
from analytics import AnalyticsManager
import pandas as pd


# ---------------------------------------------------------------------------
# Custom CSS — Premium Dark Theme
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown("""
    <style>
    /* ── Google Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ──────────────────────────────────── */
    :root {
        --bg-primary:     #0A0A0F;
        --bg-secondary:   #111118;
        --bg-card:        #16161F;
        --bg-glass:       rgba(255,255,255,0.03);
        --border:         rgba(255,255,255,0.06);
        --border-hover:   rgba(124,58,237,0.50);
        --purple-500:     #7C3AED;
        --purple-400:     #A78BFA;
        --purple-300:     #C4B5FD;
        --indigo-500:     #4F46E5;
        --teal-400:       #00D4AA;
        --pink-500:       #EC4899;
        --red-500:        #FF4B4B;
        --text-primary:   #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted:     #64748B;
        --radius-lg:      14px;
        --radius-xl:      20px;
        --shadow:         0 4px 24px rgba(0,0,0,0.40);
        --shadow-purple:  0 4px 24px rgba(124,58,237,0.25);
        --transition:     all 0.25s cubic-bezier(0.4,0,0.2,1);
    }

    /* ── Global Reset ────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── Hide Streamlit Branding ─────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }

    /* ── Tabs ────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 4px;
        gap: 4px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 8px 20px !important;
        transition: var(--transition) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--purple-500), var(--indigo-500)) !important;
        color: white !important;
        box-shadow: var(--shadow-purple) !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

    /* ── Inputs ──────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        transition: var(--transition) !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--purple-500) !important;
        box-shadow: 0 0 0 2px rgba(124,58,237,0.20) !important;
    }

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, var(--purple-500), var(--indigo-500)) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-lg) !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.55rem 1.4rem !important;
        transition: var(--transition) !important;
        box-shadow: var(--shadow-purple) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 32px rgba(124,58,237,0.40) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── File Uploader ───────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 2px dashed var(--border) !important;
        border-radius: var(--radius-xl) !important;
        padding: 1.5rem !important;
        transition: var(--transition) !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--purple-500) !important;
        background: rgba(124,58,237,0.05) !important;
    }

    /* ── Metrics ─────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-xl) !important;
        padding: 1.2rem 1.4rem !important;
        transition: var(--transition) !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--border-hover) !important;
        box-shadow: var(--shadow-purple) !important;
    }
    [data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700 !important; }

    /* ── DataFrames ──────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        background: var(--bg-card) !important;
        border-radius: var(--radius-lg) !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Alerts ──────────────────────────────────────────── */
    .stSuccess {
        background: rgba(0,212,170,0.10) !important;
        border: 1px solid rgba(0,212,170,0.30) !important;
        border-radius: var(--radius-lg) !important;
        color: var(--teal-400) !important;
    }
    .stError {
        background: rgba(255,75,75,0.10) !important;
        border: 1px solid rgba(255,75,75,0.30) !important;
        border-radius: var(--radius-lg) !important;
    }
    .stWarning {
        background: rgba(251,191,36,0.10) !important;
        border: 1px solid rgba(251,191,36,0.30) !important;
        border-radius: var(--radius-lg) !important;
    }
    .stInfo {
        background: rgba(79,70,229,0.10) !important;
        border: 1px solid rgba(79,70,229,0.30) !important;
        border-radius: var(--radius-lg) !important;
    }

    /* ── Chat Message Bubbles ────────────────────────────── */
    .user-bubble {
        background: linear-gradient(135deg, #7C3AED, #4F46E5);
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 0.85rem 1.2rem;
        margin: 0.4rem 0;
        max-width: 78%;
        margin-left: auto;
        box-shadow: 0 4px 16px rgba(124,58,237,0.30);
        animation: slideInRight 0.3s ease;
        font-size: 0.935rem;
        line-height: 1.6;
    }
    .bot-bubble {
        background: var(--bg-card);
        border: 1px solid var(--border);
        color: var(--text-primary);
        border-radius: 18px 18px 18px 4px;
        padding: 0.85rem 1.2rem;
        margin: 0.4rem 0;
        max-width: 82%;
        box-shadow: var(--shadow);
        animation: slideInLeft 0.3s ease;
        font-size: 0.935rem;
        line-height: 1.6;
    }
    .bot-bubble.fallback {
        border-color: rgba(255,75,75,0.25);
        background: rgba(255,75,75,0.05);
    }
    .chat-meta {
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }
    .chat-meta.user-meta { text-align: right; }
    .confidence-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 100px;
        margin-left: 6px;
    }
    .badge-high   { background: rgba(0,212,170,0.15); color: #00D4AA; }
    .badge-mod    { background: rgba(251,191,36,0.15); color: #FBBF24; }
    .badge-low    { background: rgba(255,75,75,0.15);  color: #FF4B4B; }
    .badge-pdf    { background: rgba(79,70,229,0.15);  color: #818CF8; }
    .badge-faq    { background: rgba(124,58,237,0.15); color: #A78BFA; }

    /* ── Typing Indicator ────────────────────────────────── */
    .typing-dots span {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--purple-400);
        margin: 0 2px;
        animation: bounce 1.2s infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

    /* ── Card ────────────────────────────────────────────── */
    .ng-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        padding: 1.4rem;
        transition: var(--transition);
    }
    .ng-card:hover { border-color: var(--border-hover); box-shadow: var(--shadow-purple); }

    /* ── Section Header ──────────────────────────────────── */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    .section-sub {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 1.2rem;
    }

    /* ── Divider ─────────────────────────────────────────── */
    hr { border-color: var(--border) !important; }

    /* ── Scrollbar ───────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb { background: var(--border-hover); border-radius: 100px; }

    /* ── Suggestion Chips ────────────────────────────────── */
    div[data-testid="stHorizontalBlock"] > div > div > button {
        background: rgba(124,58,237,0.08) !important;
        border: 1px solid rgba(124,58,237,0.30) !important;
        border-radius: 100px !important;
        color: var(--purple-300) !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        padding: 0.3rem 0.8rem !important;
        transition: var(--transition) !important;
        box-shadow: none !important;
    }
    div[data-testid="stHorizontalBlock"] > div > div > button:hover {
        background: rgba(124,58,237,0.22) !important;
        border-color: var(--purple-500) !important;
        color: #fff !important;
        transform: translateY(-1px) !important;
    }

    /* ── Animations ──────────────────────────────────────── */
    @keyframes slideInRight {
        from { opacity:0; transform: translateX(20px); }
        to   { opacity:1; transform: translateX(0); }
    }
    @keyframes slideInLeft {
        from { opacity:0; transform: translateX(-20px); }
        to   { opacity:1; transform: translateX(0); }
    }
    @keyframes bounce {
        0%, 80%, 100% { transform: translateY(0); }
        40%            { transform: translateY(-8px); }
    }
    @keyframes pulse {
        0%, 100% { opacity:1; }
        50%       { opacity:0.6; }
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        "chat_messages": [],          # List of dicts: {role, content, meta}
        "faq_engine": None,
        "faq_manager": None,
        "pdf_processor": None,
        "voice_handler": None,
        "analytics_manager": None,
        "pdf_indexed": False,
        "pdf_filename": "",
        "pdf_chunk_count": 0,
        "voice_tts_enabled": True,
        "voice_stt_enabled": True,
        "confidence_threshold": 0.25,
        "greeting_shown": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Lazy-loaded Singletons
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_faq_manager() -> FAQManager:
    return FAQManager()


@st.cache_resource(show_spinner=False)
def get_analytics_manager() -> AnalyticsManager:
    return AnalyticsManager()


@st.cache_resource(show_spinner=False)
def get_voice_handler() -> VoiceHandler:
    return VoiceHandler()


@st.cache_resource(show_spinner=False)
def get_pdf_processor() -> PDFProcessor:
    return PDFProcessor()


def get_faq_engine() -> FAQEngine:
    """Return the cached FAQEngine (rebuilt when threshold changes)."""
    threshold = st.session_state.get("confidence_threshold", 0.25)
    if (
        "faq_engine_instance" not in st.session_state
        or st.session_state.get("engine_threshold") != threshold
    ):
        faq_manager = get_faq_manager()
        engine = FAQEngine(faq_manager=faq_manager, confidence_threshold=threshold)
        st.session_state["faq_engine_instance"] = engine
        st.session_state["engine_threshold"] = threshold
    return st.session_state["faq_engine_instance"]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        # Logo / Brand
        st.markdown(textwrap.dedent("""
            <div style="text-align:center; padding: 0.5rem 0 1.5rem;">
                <div style="font-size:2.8rem;">🤖</div>
                <div style="font-size:1.4rem; font-weight:800; background: linear-gradient(135deg,#A78BFA,#60A5FA); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">NeuraFAQ</div>
                <div style="font-size:0.75rem; color:#64748B; margin-top:2px;">AI-Powered FAQ Chatbot</div>
            </div>
        """), unsafe_allow_html=True)

        st.markdown("---")

        # System Status
        engine = get_faq_engine()
        voice = get_voice_handler()
        st.markdown("**⚡ System Status**")
        col1, col2 = st.columns(2)
        with col1:
            faq_count = engine.faq_count
            st.metric("FAQs", faq_count)
        with col2:
            pdf_status = "✅" if st.session_state.get("pdf_indexed") else "—"
            st.metric("PDF", pdf_status)

        # Voice status
        voice_status = []
        if voice.stt_available:
            voice_status.append("🎤 STT")
        if voice.tts_available:
            voice_status.append("🔊 TTS")
        if voice_status:
            st.success(" · ".join(voice_status) + " Available")
        else:
            st.warning("Voice features unavailable")

        st.markdown("---")

        # Settings
        st.markdown("**⚙️ Configuration**")
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.80,
            value=st.session_state["confidence_threshold"],
            step=0.05,
            help="Minimum similarity score to return an answer. Lower = more lenient.",
            key="threshold_slider",
        )
        if threshold != st.session_state["confidence_threshold"]:
            st.session_state["confidence_threshold"] = threshold

        # Voice toggles — use checkbox for broad Streamlit version support
        st.session_state["voice_tts_enabled"] = st.checkbox(
            "🔊 Text-to-Speech",
            value=st.session_state["voice_tts_enabled"],
            key="tts_checkbox",
        )
        st.session_state["voice_stt_enabled"] = st.checkbox(
            "🎤 Voice Input",
            value=st.session_state["voice_stt_enabled"],
            key="stt_checkbox",
        )

        st.markdown("---")

        # Quick Stats
        analytics = get_analytics_manager()
        stats = analytics.get_summary_stats()
        st.markdown("**📈 Quick Stats**")
        st.caption(f"Total Queries: **{stats['total_queries']}**")
        st.caption(f"Success Rate: **{stats['success_rate']}%**")
        st.caption(f"Avg Confidence: **{stats['avg_confidence']}%**")

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#64748B; font-size:0.72rem;'>Built with ❤️ using Python & Streamlit</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Greeting Detection
# ---------------------------------------------------------------------------
_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings", "good morning",
    "good afternoon", "good evening", "good day", "sup", "what's up",
    "namaste", "helo", "hii", "helloo", "heyy",
}

_GREETING_RESPONSES = [
    (
        "👋 Hello there! I'm **NeuraFAQ**, your AI-powered FAQ assistant.\n\n"
        "I can help you with:\n"
        "- 🎓 **Admissions** — fees, eligibility, deadlines, entrance exams\n"
        "- 📚 **Academics** — courses, duration, grading, attendance\n"
        "- 🏠 **Hostel** — accommodation, fees, facilities\n"
        "- 💰 **Scholarships** — merit, need-based, sports\n"
        "- 💼 **Placements** — packages, companies, career support\n"
        "- 🚌 **Facilities** — transport, library, sports, Wi-Fi\n\n"
        "Try clicking one of the suggested questions below, or just ask away! 😊"
    ),
]

_THANKS_WORDS = {"thanks", "thank you", "thank", "thx", "ty", "great", "awesome", "perfect", "nice"}

_THANKS_RESPONSE = (
    "😊 You're welcome! Is there anything else I can help you with?\n\n"
    "Feel free to ask about admissions, courses, fees, hostel, placements, or any other queries!"
)

def _detect_special_query(query: str):
    """
    Returns (type, response) if the query is a greeting/thanks/social.
    Returns (None, None) if it should go through the FAQ engine.
    """
    q = query.strip().lower().rstrip("!?.,")
    if q in _GREETINGS or any(q.startswith(g) for g in _GREETINGS):
        return "greeting", _GREETING_RESPONSES[0]
    if q in _THANKS_WORDS or any(q.startswith(t) for t in _THANKS_WORDS):
        return "thanks", _THANKS_RESPONSE
    return None, None


# Suggested questions to show below the chat input
_SUGGESTED_QUESTIONS = [
    "What is the admission fee?",
    "What courses are offered?",
    "What is the hostel fee?",
    "Are scholarships available?",
    "What is the placement rate?",
    "What is the attendance requirement?",
    "Is there a transport facility?",
    "Can fees be paid in installments?",
]


# ---------------------------------------------------------------------------
# Tab 1: Chat Portal
# ---------------------------------------------------------------------------
def render_chat_tab() -> None:
    st.markdown(textwrap.dedent("""
        <div class='section-title'>💬 Chat Portal</div>
        <div class='section-sub'>Ask any question — I'll find the most relevant answer from the knowledge base.</div>
    """), unsafe_allow_html=True)

    engine = get_faq_engine()
    voice = get_voice_handler()
    analytics = get_analytics_manager()

    # --- Initial greeting ---
    if not st.session_state["greeting_shown"]:
        st.session_state["chat_messages"].append({
            "role": "assistant",
            "content": _GREETING_RESPONSES[0],
            "meta": {"confidence": 1.0, "source": "System", "category": "Greeting", "is_successful": True},
        })
        st.session_state["greeting_shown"] = True

    # --- Chat History ---
    for msg in st.session_state["chat_messages"]:
        _render_message(msg)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input Form (Enter key submits) ─────────────────────────
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_send, col_voice = st.columns([8, 1.2, 1.2])
        with col_input:
            user_input = st.text_input(
                label="Ask your question",
                placeholder="Type your question and press Enter…",
                label_visibility="collapsed",
                key="chat_input_field",
            )
        with col_send:
            send_clicked = st.form_submit_button(
                "Send ➤",
                use_container_width=True,
            )
        with col_voice:
            voice_clicked = st.form_submit_button(
                "🎤",
                use_container_width=True,
                help="Click to speak",
            )

    # ── Action buttons row ─────────────────────────────────────
    col_clear, col_spacer = st.columns([1.5, 8])
    with col_clear:
        if st.button("🗑️ Clear Chat", key="clear_btn"):
            st.session_state["chat_messages"] = []
            st.session_state["greeting_shown"] = False
            engine.clear_history()
            st.rerun()

    # ── Suggested Questions ────────────────────────────────────
    st.markdown(textwrap.dedent("""
        <div style='margin-top:0.8rem; margin-bottom:0.4rem;
                    font-size:0.78rem; color:#64748B; font-weight:600; letter-spacing:0.04em;'>
            💡 SUGGESTED QUESTIONS
        </div>
    """), unsafe_allow_html=True)

    # Render suggestion chips in rows of 4
    suggestion_cols = st.columns(4)
    for i, suggestion in enumerate(_SUGGESTED_QUESTIONS):
        with suggestion_cols[i % 4]:
            if st.button(
                suggestion,
                key=f"suggest_{i}",
                use_container_width=True,
            ):
                st.session_state["pending_query"] = suggestion
                st.rerun()

    # ── Handle Voice Input ─────────────────────────────────────
    if voice_clicked and st.session_state.get("voice_stt_enabled", True):
        with st.spinner("🎙️ Listening for 5 seconds… Speak now!"):
            success, text = voice.listen()
        if success:
            st.session_state["pending_query"] = text
            st.rerun()
        else:
            st.warning(f"🎙️ {text}")

    # ── Resolve pending query (from suggestion or voice) ───────
    pending = st.session_state.pop("pending_query", None)
    if pending:
        user_input = pending
        send_clicked = True

    # ── Process query ──────────────────────────────────────────
    if send_clicked and user_input and user_input.strip():
        query = user_input.strip()

        # Add user message to history
        st.session_state["chat_messages"].append({
            "role": "user",
            "content": query,
            "meta": {},
        })

        # Check for greeting / thanks first
        special_type, special_response = _detect_special_query(query)

        if special_type:
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": special_response,
                "meta": {"confidence": 1.0, "source": "System",
                         "category": "Greeting", "is_successful": True,
                         "response_time_ms": 0},
            })
        else:
            # Re-attach PDF index if available
            if st.session_state.get("pdf_indexed") and st.session_state.get("pdf_chunks"):
                if not engine.has_pdf:
                    engine.index_pdf_chunks(st.session_state["pdf_chunks"])

            with st.spinner("🤔 Thinking…"):
                response = engine.query(query)

            analytics.log_chat_response(response)

            bot_meta = {
                "confidence": response.confidence,
                "source": response.source,
                "category": response.category,
                "matched_question": response.matched_question,
                "response_time_ms": response.response_time_ms,
                "is_successful": response.is_successful,
            }
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": response.answer,
                "meta": bot_meta,
            })

            # TTS for successful responses
            if st.session_state.get("voice_tts_enabled") and voice.tts_available and response.is_successful:
                voice.speak(response.answer[:300])

        st.rerun()


def _render_message(msg: dict) -> None:
    """Render a single chat message bubble with properly escaped content."""
    role = msg.get("role", "user")
    raw_content = msg.get("content", "")
    meta = msg.get("meta", {})
    ts = datetime.now().strftime("%H:%M")

    # Escape HTML to prevent raw HTML tags from rendering as markup.
    # Then convert newlines and **bold** markdown to safe HTML.
    def _safe_html(text: str) -> str:
        escaped = html_module.escape(text)
        # Convert **bold** to <strong>
        import re
        escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
        # Convert *italic* to <em>
        escaped = re.sub(r'\*(.+?)\*', r'<em>\1</em>', escaped)
        # Convert newlines to <br>
        escaped = escaped.replace('\n', '<br>')
        # Convert markdown list items (- item) to bullet points
        escaped = re.sub(r'(?m)^- (.+)', r'&bull; \1<br>', escaped)
        return escaped

    content = _safe_html(raw_content)

    if role == "user":
        safe_user = html_module.escape(raw_content).replace('\n', '<br>')
        st.markdown(textwrap.dedent(f"""
            <div class="user-bubble">{safe_user}</div>
            <div class="chat-meta user-meta">You &middot; {ts}</div>
        """), unsafe_allow_html=True)
    else:
        conf = meta.get("confidence", 0)
        source = meta.get("source", "")
        is_successful = meta.get("is_successful", True)
        matched_q = meta.get("matched_question", "")
        resp_time = meta.get("response_time_ms", 0)

        # Confidence badge
        badge_class = "badge-low"
        conf_label = ""
        if conf >= 0.70:
            badge_class = "badge-high"
            conf_label = f"{conf*100:.0f}%"
        elif conf >= 0.40:
            badge_class = "badge-mod"
            conf_label = f"{conf*100:.0f}%"
        elif conf > 0:
            badge_class = "badge-low"
            conf_label = f"{conf*100:.0f}%"

        # Source badge
        src_badge = ""
        if source == "FAQ":
            src_badge = '<span class="confidence-badge badge-faq">FAQ</span>'
        elif source == "PDF":
            src_badge = '<span class="confidence-badge badge-pdf">PDF</span>'
        elif source == "System":
            src_badge = '<span class="confidence-badge badge-mod">System</span>'

        conf_badge = f'<span class="confidence-badge {badge_class}">{conf_label}</span>' if conf_label else ""
        bubble_class = "bot-bubble" if is_successful else "bot-bubble fallback"

        extra_info = ""
        if matched_q and matched_q not in ("[From uploaded PDF]", ""):
            safe_matched = html_module.escape(matched_q)
            extra_info = f'<div style="font-size:0.72rem;color:#64748B;margin-top:0.5rem;">&#128206; Matched: <em>{safe_matched}</em></div>'

        timing = f'&nbsp;&middot; <span style="color:#64748B;">{resp_time:.0f}ms</span>' if resp_time else ''

        st.markdown(textwrap.dedent(f"""
            <div class="{bubble_class}">
                <div style="font-size:1.1rem;">&#129302;</div>
                <div style="margin-top:0.4rem;">{content}</div>
                {extra_info}
                <div class="chat-meta" style="margin-top:0.5rem;">
                    NeuraFAQ &middot; {ts} &nbsp; {src_badge}{conf_badge}{timing}
                </div>
            </div>
        """), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 2: Document Indexer
# ---------------------------------------------------------------------------
def render_pdf_tab() -> None:
    st.markdown(textwrap.dedent("""
        <div class='section-title'>📁 Document Indexer</div>
        <div class='section-sub'>Upload a PDF document and ask questions based on its content.</div>
    """), unsafe_allow_html=True)

    pdf_processor = get_pdf_processor()
    engine = get_faq_engine()

    # Current PDF status
    if st.session_state.get("pdf_indexed"):
        st.success(
            f"✅ **{st.session_state['pdf_filename']}** is indexed · "
            f"{st.session_state['pdf_chunk_count']} chunks ready"
        )
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Remove PDF", key="remove_pdf_btn"):
                engine.clear_pdf_index()
                st.session_state["pdf_indexed"] = False
                st.session_state["pdf_filename"] = ""
                st.session_state["pdf_chunk_count"] = 0
                st.session_state.pop("pdf_chunks", None)
                st.success("PDF index cleared.")
                st.rerun()

    st.markdown("---")

    # Upload
    uploaded_file = st.file_uploader(
        "📄 Upload a PDF document",
        type=["pdf"],
        help="Supported format: PDF. Maximum size: 50 MB.",
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"📄 File: **{uploaded_file.name}** · Size: {uploaded_file.size / 1024:.1f} KB")
        with col2:
            if st.button("⚡ Index Document", key="index_pdf_btn", use_container_width=True):
                with st.spinner(f"📖 Processing {uploaded_file.name}..."):
                    file_bytes = uploaded_file.read()
                    chunks = pdf_processor.process_bytes(file_bytes, uploaded_file.name)

                if chunks:
                    engine.index_pdf_chunks(chunks)
                    st.session_state["pdf_indexed"] = True
                    st.session_state["pdf_filename"] = uploaded_file.name
                    st.session_state["pdf_chunk_count"] = len(chunks)
                    st.session_state["pdf_chunks"] = chunks
                    st.success(f"✅ Indexed **{len(chunks)} chunks** from **{uploaded_file.name}**")
                    st.balloons()
                else:
                    st.error("❌ Could not extract text from this PDF. Ensure it contains selectable text (not scanned image).")

    st.markdown("---")

    # Preview indexed chunks
    if st.session_state.get("pdf_indexed") and st.session_state.get("pdf_chunks"):
        with st.expander("🔍 Preview Indexed Chunks", expanded=False):
            chunks = st.session_state["pdf_chunks"]
            for i, chunk in enumerate(chunks[:10], 1):
                st.markdown(textwrap.dedent(f"""
                    <div class="ng-card" style="margin-bottom:0.75rem;">
                        <div style="font-size:0.7rem;color:#64748B;margin-bottom:0.4rem;">Chunk {i}</div>
                        <div style="font-size:0.85rem;color:#CBD5E1;">{chunk[:300]}...</div>
                    </div>
                """), unsafe_allow_html=True)
            if len(chunks) > 10:
                st.caption(f"... and {len(chunks) - 10} more chunks.")


# ---------------------------------------------------------------------------
# Tab 3: Analytics Dashboard
# ---------------------------------------------------------------------------
def render_analytics_tab() -> None:
    st.markdown(textwrap.dedent("""
        <div class='section-title'>📊 Analytics Dashboard</div>
        <div class='section-sub'>Real-time insights into chatbot performance and usage patterns.</div>
    """), unsafe_allow_html=True)

    analytics = get_analytics_manager()
    stats = analytics.get_summary_stats()

    # ── KPI Metrics ─────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🗣️ Total Queries", stats["total_queries"])
    c2.metric("✅ Successful", stats["successful_queries"])
    c3.metric("❌ Failed", stats["failed_queries"])
    c4.metric("🎯 Avg Confidence", f"{stats['avg_confidence']}%")
    c5.metric("⚡ Avg Response", f"{stats['avg_response_time_ms']}ms")

    st.markdown("---")

    if stats["total_queries"] == 0:
        st.info("📭 No data yet. Start chatting to see analytics appear here!")
        return

    # ── Charts Row 1 ─────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig = analytics.plot_query_volume()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = analytics.plot_success_rate()
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ── Charts Row 2 ─────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        fig = analytics.plot_confidence_distribution()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = analytics.plot_source_breakdown()
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ── Top Questions Chart ──────────────────────────────────────
    fig = analytics.plot_top_questions(10)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # ── Category Breakdown ───────────────────────────────────────
    fig = analytics.plot_category_breakdown()
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Search History Table ─────────────────────────────────────
    st.markdown("### 📋 Full Query History")
    df = analytics.load_data()
    if not df.empty:
        search_term = st.text_input("🔍 Filter history…", placeholder="Search queries…", key="history_search")
        if search_term:
            mask = (
                df["query"].str.contains(search_term, case=False, na=False) |
                df["category"].str.contains(search_term, case=False, na=False)
            )
            df = df[mask]

        display_cols = ["timestamp", "query", "confidence", "source", "category", "is_successful", "response_time_ms"]
        st.dataframe(
            df[display_cols].sort_values("timestamp", ascending=False).head(200),
            use_container_width=True,
            hide_index=True,
        )
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear All Logs", key="clear_logs_btn"):
                analytics.clear_logs()
                st.success("Logs cleared.")
                st.rerun()
    else:
        st.info("No history records available yet.")


# ---------------------------------------------------------------------------
# Tab 4: FAQ Manager (Admin Panel)
# ---------------------------------------------------------------------------
def render_faq_manager_tab() -> None:
    st.markdown(textwrap.dedent("""
        <div class='section-title'>⚙️ FAQ Manager</div>
        <div class='section-sub'>Add, edit, or delete FAQ entries. Changes take effect immediately without restarting.</div>
    """), unsafe_allow_html=True)

    faq_manager = get_faq_manager()
    engine = get_faq_engine()

    # ── Sub-tabs ──────────────────────────────────────────────────
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["➕ Add FAQ", "📋 Browse & Edit", "📤 Import / Export", "🔍 Search"])

    # ── Add FAQ ───────────────────────────────────────────────────
    with sub_tab1:
        st.markdown("#### Add New FAQ Entry")
        with st.form("add_faq_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                new_category = st.selectbox(
                    "Category",
                    options=faq_manager.get_categories() + ["Add New…"],
                    key="add_category_select",
                )
                if new_category == "Add New…":
                    new_category = st.text_input("New Category Name", key="new_category_input")
            with col2:
                new_tags = st.text_input("Tags (comma-separated)", placeholder="fee, admission, cost", key="add_tags")
            new_question = st.text_area("Question *", placeholder="What is the admission fee?", key="add_question", height=80)
            new_answer = st.text_area("Answer *", placeholder="The admission fee is ₹50,000…", key="add_answer", height=100)
            submitted = st.form_submit_button("➕ Add FAQ", use_container_width=True)

            if submitted:
                tags_list = [t.strip() for t in new_tags.split(",") if t.strip()]
                success, msg = faq_manager.add_faq(new_question, new_answer, new_category, tags_list)
                if success:
                    engine.rebuild_faq_index()
                    st.success(f"✅ FAQ added successfully! (ID: `{msg}`)")
                else:
                    st.error(f"❌ {msg}")

    # ── Browse & Edit ─────────────────────────────────────────────
    with sub_tab2:
        st.markdown("#### Browse FAQs")
        faqs = faq_manager.get_all()
        categories = ["All"] + faq_manager.get_categories()
        selected_cat = st.selectbox("Filter by Category", categories, key="browse_cat_filter")

        if selected_cat != "All":
            faqs = [f for f in faqs if f.category == selected_cat]

        st.caption(f"Showing **{len(faqs)}** FAQ entries")

        for faq in faqs:
            with st.expander(f"**[{faq.category}]** {faq.question[:70]}…", expanded=False):
                col1, col2, col3 = st.columns([5, 1, 1])
                with col1:
                    st.markdown(f"**ID:** `{faq.id}`")
                    st.markdown(f"**Category:** {faq.category}")

                edit_q = st.text_area(f"Question", value=faq.question, key=f"edit_q_{faq.id}", height=70)
                edit_a = st.text_area(f"Answer", value=faq.answer, key=f"edit_a_{faq.id}", height=90)
                edit_cat = st.text_input("Category", value=faq.category, key=f"edit_cat_{faq.id}")

                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button("💾 Save Changes", key=f"save_{faq.id}", use_container_width=True):
                        success, msg = faq_manager.edit_faq(faq.id, question=edit_q, answer=edit_a, category=edit_cat)
                        if success:
                            engine.rebuild_faq_index()
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")
                with bcol2:
                    if st.button("🗑️ Delete", key=f"del_{faq.id}", use_container_width=True):
                        success, msg = faq_manager.delete_faq(faq.id)
                        if success:
                            engine.rebuild_faq_index()
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

    # ── Import / Export ───────────────────────────────────────────
    with sub_tab3:
        st.markdown("#### Import FAQs from File")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📥 Import from JSON**")
            json_file = st.file_uploader("Upload JSON", type=["json"], key="import_json")
            if json_file and st.button("Import JSON", key="do_import_json"):
                tmp_path = Path("uploads") / json_file.name
                tmp_path.parent.mkdir(exist_ok=True)
                tmp_path.write_bytes(json_file.read())
                added, skipped = faq_manager.import_from_json(str(tmp_path))
                engine.rebuild_faq_index()
                st.success(f"✅ Imported {added} FAQs, skipped {skipped} duplicates.")
        with col2:
            st.markdown("**📥 Import from CSV**")
            csv_file = st.file_uploader("Upload CSV", type=["csv"], key="import_csv")
            if csv_file and st.button("Import CSV", key="do_import_csv"):
                tmp_path = Path("uploads") / csv_file.name
                tmp_path.parent.mkdir(exist_ok=True)
                tmp_path.write_bytes(csv_file.read())
                added, skipped = faq_manager.import_from_csv(str(tmp_path))
                engine.rebuild_faq_index()
                st.success(f"✅ Imported {added} FAQs, skipped {skipped} duplicates.")

        st.markdown("---")
        st.markdown("**📤 Export All FAQs**")
        if st.button("⬇️ Download FAQ as JSON", key="export_json_btn"):
            export_path = Path("data") / "faq_export.json"
            if faq_manager.export_to_json(str(export_path)):
                with open(export_path, "rb") as f:
                    st.download_button(
                        label="📥 Click to Download",
                        data=f,
                        file_name="faq_export.json",
                        mime="application/json",
                        key="download_json_btn",
                    )

    # ── Search ────────────────────────────────────────────────────
    with sub_tab4:
        st.markdown("#### Search Knowledge Base")
        search_kw = st.text_input("Search by keyword", placeholder="hostel, fee, scholarship…", key="faq_search_kw")
        if search_kw:
            results = faq_manager.search(search_kw)
            st.caption(f"Found **{len(results)}** matches for *'{search_kw}'*")
            for faq in results:
                st.markdown(textwrap.dedent(f"""
                    <div class="ng-card" style="margin-bottom:0.75rem;">
                        <div style="font-size:0.7rem;color:#A78BFA;">{faq.category} · {faq.id}</div>
                        <div style="font-weight:600;margin:0.3rem 0;">{faq.question}</div>
                        <div style="font-size:0.85rem;color:#94A3B8;">{faq.answer}</div>
                    </div>
                """), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main() -> None:
    inject_css()
    init_session_state()
    render_sidebar()

    # ── Hero Header ───────────────────────────────────────────────
    st.markdown(textwrap.dedent("""
        <div style="text-align:center; padding: 1.5rem 0 1rem;">
            <h1 style="font-size:2.4rem; font-weight:800; background:linear-gradient(135deg,#A78BFA,#60A5FA,#34D399); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.3rem;">
                🤖 NeuraFAQ
            </h1>
            <p style="color:#94A3B8; font-size:1.0rem;">AI-Powered Semantic FAQ Chatbot · TF-IDF · Cosine Similarity · NLP</p>
        </div>
    """), unsafe_allow_html=True)

    # ── Navigation Tabs ───────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬  Chat Portal",
        "📁  Document Indexer",
        "📊  Analytics",
        "⚙️  FAQ Manager",
    ])

    with tab1:
        render_chat_tab()
    with tab2:
        render_pdf_tab()
    with tab3:
        render_analytics_tab()
    with tab4:
        render_faq_manager_tab()


if __name__ == "__main__":
    main()
