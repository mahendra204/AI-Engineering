"""
ui.py
Streamlit chat interface for the RAG application.

Run:
    streamlit run ui.py
"""

import sys
from pathlib import Path
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp { background-color: #0f1117; color: #e8e8e8; }
    
    /* Chat messages */
    .user-bubble {
        background: #1e3a5f;
        border-left: 3px solid #4a9eff;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 15px;
    }
    .assistant-bubble {
        background: #1a1f2e;
        border-left: 3px solid #50fa7b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 15px;
    }
    .source-card {
        background: #12151e;
        border: 1px solid #2a2d3a;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 4px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #8892b0;
    }
    .score-badge {
        background: #1e3a2a;
        color: #50fa7b;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .stat-box {
        background: #12151e;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .stat-number { font-size: 28px; font-weight: 600; color: #4a9eff; }
    .stat-label  { font-size: 12px; color: #8892b0; margin-top: 4px; }
    
    /* Input */
    .stTextInput input, .stTextArea textarea {
        background: #1a1f2e !important;
        color: #e8e8e8 !important;
        border: 1px solid #2a2d3a !important;
        border-radius: 6px !important;
    }
    .stButton button {
        background: #4a9eff;
        color: #0f1117;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        width: 100%;
    }
    .stButton button:hover { background: #6ab4ff; }
    
    div[data-testid="stSidebarContent"] { background: #12151e; }
    
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline init ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising RAG pipeline...")
def load_pipeline():
    from app.rag_pipeline import RAGPipeline
    return RAGPipeline()


def safe_load_pipeline():
    try:
        return load_pipeline()
    except Exception as e:
        st.error(f"❌ Failed to initialise pipeline: {e}")
        st.info("Make sure your `.env` file has valid API keys.")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(pipeline):
    with st.sidebar:
        st.markdown("## 🔍 RAG Assistant")
        st.markdown("---")

        # Stats
        if pipeline:
            sources = pipeline.list_sources()
            chunks = pipeline.count()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-number">{chunks}</div>
                    <div class="stat-label">Chunks</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-number">{len(sources)}</div>
                    <div class="stat-label">Documents</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("")

        # Upload
        st.markdown("### 📁 Add Documents")
        uploaded = st.file_uploader(
            "Drop files here",
            type=["pdf", "txt", "md", "docx", "html", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded and pipeline:
            if st.button("⬆️ Ingest Selected Files"):
                with st.spinner("Ingesting..."):
                    total = 0
                    for f in uploaded:
                        try:
                            n = pipeline.ingest_bytes(f.read(), f.name)
                            total += n
                            st.success(f"✅ {f.name} → {n} chunks")
                        except Exception as e:
                            st.error(f"❌ {f.name}: {e}")
                if total > 0:
                    st.cache_resource.clear()
                    st.rerun()

        # Settings
        st.markdown("### ⚙️ Retrieval Settings")
        top_k = st.slider("Top-K results", 1, 20, 5)
        mode = st.selectbox("Mode", ["hybrid", "semantic", "keyword"], index=0)

        # Sources list
        if pipeline and pipeline.list_sources():
            st.markdown("### 📚 Knowledge Base")
            for src in pipeline.list_sources():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"<small>📄 {src['filename']}<br><span style='color:#8892b0'>{src['chunks']} chunks</span></small>",
                                unsafe_allow_html=True)
                with col2:
                    if st.button("🗑", key=f"del_{src['filename']}"):
                        pipeline.delete_source(src["filename"])
                        st.rerun()

        # Clear chat
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    return top_k, mode


# ── Chat rendering ────────────────────────────────────────────────────────────

def render_message(role: str, content: str, sources=None):
    if role == "user":
        st.markdown(f'<div class="user-bubble">👤 {content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">🤖 {content}</div>', unsafe_allow_html=True)
        if sources:
            with st.expander(f"📎 {len(sources)} source(s) used"):
                for src in sources:
                    badge = f'<span class="score-badge">{src["score"]:.2f}</span>'
                    page = f" · p.{src['page']}" if src.get("page") else ""
                    st.markdown(
                        f"""<div class="source-card">
                            {badge} <strong>{src['filename']}</strong>{page}<br>
                            <span style="color:#6b7280">{src['content'][:300]}…</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pipeline = safe_load_pipeline()
    top_k, mode = render_sidebar(pipeline)

    # Title
    st.markdown("# 🔍 RAG Knowledge Assistant")
    st.markdown("<p style='color:#8892b0;margin-top:-12px'>Ask questions against your ingested documents</p>",
                unsafe_allow_html=True)

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"], msg.get("sources"))

    # Empty state
    if pipeline and pipeline.count() == 0:
        st.info("📂 **No documents ingested yet.** Use the sidebar to upload PDF, TXT, DOCX, or other supported files.")

    # Chat input
    if question := st.chat_input("Ask a question about your documents…"):
        if not pipeline:
            st.error("Pipeline not available. Check your API keys.")
            return

        if pipeline.count() == 0:
            st.warning("Please ingest some documents first using the sidebar.")
            return

        # Save + render user message
        st.session_state.messages.append({"role": "user", "content": question})
        render_message("user", question)

        # Stream answer
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            with st.spinner("Thinking…"):
                try:
                    # Override retrieval mode from sidebar
                    pipeline.retriever.mode = mode
                    result = pipeline.query(question, top_k=top_k)
                    full_answer = result.answer
                    sources = [
                        {
                            "content": s.content,
                            "filename": s.filename,
                            "page": s.page,
                            "score": s.score,
                        }
                        for s in result.sources
                    ]
                    placeholder.markdown(f'<div class="assistant-bubble">🤖 {full_answer}</div>',
                                         unsafe_allow_html=True)
                    # Show sources
                    if sources:
                        with st.expander(f"📎 {len(sources)} source(s) used"):
                            for src in sources:
                                badge = f'<span class="score-badge">{src["score"]:.2f}</span>'
                                page = f" · p.{src['page']}" if src.get("page") else ""
                                st.markdown(
                                    f"""<div class="source-card">
                                        {badge} <strong>{src['filename']}</strong>{page}<br>
                                        <span style="color:#6b7280">{src['content'][:300]}…</span>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )
                    # Latency
                    st.caption(f"⚡ {result.latency_ms:.0f}ms · {result.tokens_used} tokens · {result.retrieval_mode} retrieval · {result.model}")

                except Exception as e:
                    full_answer = f"Error: {e}"
                    placeholder.error(full_answer)
                    sources = []

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
        })


if __name__ == "__main__":
    main()
