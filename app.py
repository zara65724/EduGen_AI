"""
EduGen_AI – Smart Study Assistant
Main application entry point | Powered by Groq API
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from utils import (
    extract_text_from_pdfs,
    create_vector_store,
    get_qa_chain,
    summarize_text,
    generate_mcqs,
    extract_key_points,
    format_chat_history,
)

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduGen_AI – Smart Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "chat_history":  [],
        "vector_store":  None,
        "raw_text":      "",
        "pdf_processed": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_api_key() -> str:
    return (
        os.getenv("EduGen_AI")
        or st.secrets.get("EduGen_AI", "")
    )

def add_chat_message(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})

def render_chat_bubble(role: str, content: str):
    if role == "user":
        st.markdown(
            f'<div class="chat-bubble user-bubble">'
            f'<span class="bubble-icon">👤</span>'
            f'<div class="bubble-text">{content}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-bubble ai-bubble">'
            f'<span class="bubble-icon">🤖</span>'
            f'<div class="bubble-text">{content}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">🎓</div>
            <div class="brand-text">
                <span class="brand-title">EduGen_AI</span>
                <span class="brand-sub">Smart Study Assistant</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── API Key ───────────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">🔑 Groq API Key</p>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=get_api_key(),
        placeholder="gsk_...",
        label_visibility="collapsed",
    )
    if api_key_input:
        os.environ["EduGen_AI"] = api_key_input

    st.markdown(
        '<a class="api-link" href="https://console.groq.com/keys" target="_blank">'
        '🔗 Get Free Groq API Key</a>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

    # ── File Upload ───────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">📄 Upload Study Material</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(
            f'<div class="file-count-badge">'
            f'{len(uploaded_files)} PDF{"s" if len(uploaded_files) > 1 else ""} selected'
            f'</div>',
            unsafe_allow_html=True,
        )

    process_btn = st.button(
        "⚡ Process Documents",
        use_container_width=True,
        disabled=not uploaded_files or not api_key_input,
    )

    if process_btn and uploaded_files and api_key_input:
        with st.spinner("⚡ Extracting & indexing your PDFs…"):
            try:
                raw_text = extract_text_from_pdfs(uploaded_files)
                if not raw_text.strip():
                    st.error("❌ Could not extract text. Check your PDFs.")
                else:
                    vector_store = create_vector_store(raw_text, api_key_input)
                    st.session_state.vector_store  = vector_store
                    st.session_state.raw_text      = raw_text
                    st.session_state.pdf_processed = True
                    st.session_state.chat_history  = []
                    st.success(f"✅ {len(uploaded_files)} PDF(s) indexed!")
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")

    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

    # ── Status ────────────────────────────────────────────────────────────────
    if st.session_state.pdf_processed:
        word_count = len(st.session_state.raw_text.split())
        st.markdown(
            f"""
            <div class="status-card active">
                <div class="status-dot"></div>
                <div>
                    <div class="status-title">Knowledge Base Ready</div>
                    <div class="status-sub">~{word_count:,} words indexed</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-card inactive">
                <div class="status-dot-off"></div>
                <div>
                    <div class="status-title">No Documents Loaded</div>
                    <div class="status-sub">Upload PDFs to begin</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-footer">Built with ❤️ using Groq + LangChain</p>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="main-header">
        <h1 class="main-title">EduGen<span class="accent">_AI</span></h1>
        <p class="main-subtitle">
            Your intelligent study companion — ask questions, generate MCQs,
            summarize documents, and extract key insights instantly.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_summary, tab_mcq, tab_keypoints = st.tabs([
    "💬 Ask Questions",
    "📋 Summarize",
    "📝 Generate MCQs",
    "🔍 Key Points",
])

# ════════════════════════════════
# TAB 1 – ASK QUESTIONS
# ════════════════════════════════
with tab_chat:
    st.markdown(
        '<div class="tab-intro">Ask anything from your uploaded documents.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="empty-chat">
                <div class="empty-icon">💡</div>
                <div class="empty-title">Start a conversation</div>
                <div class="empty-sub">Upload a PDF and ask your first question below.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.chat_history:
            render_chat_bubble(msg["role"], msg["content"])

    st.markdown('<div class="chat-input-spacer"></div>', unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            "Your question",
            placeholder="e.g. What is the main topic of this document?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_btn:
        ask_btn = st.button("Ask ✦", use_container_width=True)

    if ask_btn and user_question:
        if not st.session_state.pdf_processed:
            st.warning("⚠️ Please upload and process a PDF first.")
        elif not api_key_input:
            st.warning("⚠️ Please enter your Groq API key.")
        else:
            add_chat_message("user", user_question)
            with st.spinner("⚡ Thinking…"):
                try:
                    chain = get_qa_chain(st.session_state.vector_store, api_key_input)
                    history_pairs = format_chat_history(st.session_state.chat_history[:-1])
                    response = chain.invoke({
                        "question": user_question,
                        "chat_history": history_pairs,
                    })
                    answer = response.get("answer", str(response))
                    add_chat_message("assistant", answer)
                except Exception as e:
                    add_chat_message("assistant", f"❌ Error: {e}")
            st.rerun()

# ════════════════════════════════
# TAB 2 – SUMMARIZE
# ════════════════════════════════
with tab_summary:
    st.markdown(
        '<div class="tab-intro">Generate a structured summary of your uploaded documents.</div>',
        unsafe_allow_html=True,
    )
    detail_level = st.select_slider(
        "Summary detail level",
        options=["Brief", "Standard", "Detailed"],
        value="Standard",
    )
    if st.button("📋 Generate Summary", use_container_width=True):
        if not st.session_state.pdf_processed:
            st.warning("⚠️ Please upload and process a PDF first.")
        elif not api_key_input:
            st.warning("⚠️ Please enter your Groq API key.")
        else:
            with st.spinner("⚡ Generating summary…"):
                try:
                    summary = summarize_text(
                        st.session_state.raw_text, api_key_input, detail_level.lower()
                    )
                    st.markdown(
                        f'<div class="result-card"><h3>📋 Document Summary</h3>'
                        f'<div class="result-body">{summary}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.download_button("⬇️ Download Summary", summary, "summary.txt", "text/plain")
                except Exception as e:
                    st.error(f"❌ {e}")

# ════════════════════════════════
# TAB 3 – MCQs
# ════════════════════════════════
with tab_mcq:
    st.markdown(
        '<div class="tab-intro">Auto-generate multiple choice questions for self-testing.</div>',
        unsafe_allow_html=True,
    )
    col_n, col_diff = st.columns(2)
    with col_n:
        num_mcqs = st.number_input("Number of questions", min_value=3, max_value=20, value=5)
    with col_diff:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

    if st.button("📝 Generate MCQs", use_container_width=True):
        if not st.session_state.pdf_processed:
            st.warning("⚠️ Please upload and process a PDF first.")
        elif not api_key_input:
            st.warning("⚠️ Please enter your Groq API key.")
        else:
            with st.spinner("⚡ Crafting questions…"):
                try:
                    mcqs_text = generate_mcqs(
                        st.session_state.raw_text, api_key_input,
                        int(num_mcqs), difficulty.lower()
                    )
                    st.markdown(
                        f'<div class="result-card"><h3>📝 Multiple Choice Questions</h3>'
                        f'<div class="result-body">{mcqs_text}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.download_button("⬇️ Download MCQs", mcqs_text, "mcqs.txt", "text/plain")
                except Exception as e:
                    st.error(f"❌ {e}")

# ════════════════════════════════
# TAB 4 – KEY POINTS
# ════════════════════════════════
with tab_keypoints:
    st.markdown(
        '<div class="tab-intro">Extract the most important concepts from your material.</div>',
        unsafe_allow_html=True,
    )
    num_points = st.slider("Number of key points", min_value=5, max_value=20, value=10)

    if st.button("🔍 Extract Key Points", use_container_width=True):
        if not st.session_state.pdf_processed:
            st.warning("⚠️ Please upload and process a PDF first.")
        elif not api_key_input:
            st.warning("⚠️ Please enter your Groq API key.")
        else:
            with st.spinner("⚡ Extracting insights…"):
                try:
                    points = extract_key_points(
                        st.session_state.raw_text, api_key_input, num_points
                    )
                    st.markdown(
                        f'<div class="result-card"><h3>🔍 Key Points</h3>'
                        f'<div class="result-body">{points}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.download_button(
                        "⬇️ Download Key Points", points, "key_points.txt", "text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-footer">
        EduGen_AI &nbsp;|&nbsp; Powered by Groq &amp; LangChain
        &nbsp;|&nbsp; Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
