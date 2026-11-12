"""WAVE: Streamlit interface for the local Website RAG Assistant.

Replace the project's app.py with this file. Run: streamlit run app.py
"""

from urllib.parse import urlparse

import streamlit as st

from scrapper.crawler import iter_website_pages, normalize_url
from RAG.chunker import chunk_pages
from RAG.Vectorstore import VectorStore
from RAG.graph import RAGGraph

COLLECTION = "website_chunks_v3"  # Must match RAG/graph.py
BATCH_SIZE = 20
CRAWL_DELAY = 0.5
NOT_FOUND = "I could not find this information on the website."

st.set_page_config(page_title="WAVE | Explore beyond the page", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root {color-scheme:dark;}
.stApp {background: radial-gradient(ellipse at 85% 0%, #292347 0%, #171626 37%, #131320 76%); color:#f3f0ff;}
.block-container {max-width: 1020px; padding-top: 1.4rem; padding-bottom: 6rem;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background:#201d34;}
[data-testid="stSidebar"] * {color:#f1edff;}
[data-testid="stChatMessage"] {background: #242139; border: 1px solid #393451; border-radius:18px; padding:1rem 1.2rem; margin:.7rem 0;}
[data-testid="stChatMessage"] p {line-height:1.7;}
[data-testid="stChatInput"] {border-radius:18px; border:1px solid #62568d; background:#28243f;}
[data-testid="stChatInput"] textarea {color:#f3efff;}
.stButton > button {border-radius:999px; border:1px solid #49415f; background:#302b49; color:#eee9ff;}
.stButton > button:hover {border-color:#ac9afa; background:#41365f; color:white;}
.stButton > button[kind="primary"] {background:#a99af1; color:#1c1732; border-color:#a99af1; font-weight:700;}
.stButton > button[kind="primary"]:hover {background:#c0b6ff; color:#1c1732;}
[data-testid="stExpander"] {border:1px solid #423b61; border-radius:14px; background:#25223a;}
[data-testid="stMetric"] {background:#28243e; padding:.8rem; border-radius:12px;}
.wave-brand {display:flex; align-items:center; gap:12px; font-size:1.55rem; font-weight:750; letter-spacing:-.06em;}
.wave-logo {background:#ab9cf2; color:#201936; border-radius:13px; width:42px; height:42px; display:flex; align-items:center; justify-content:center; font-size:1.4rem;}
.wave-eyebrow {font-size:.75rem; color:#b6a6ef; font-weight:750; letter-spacing:.16em; margin-top:2.2rem;}
.wave-title {font-size:clamp(2rem,5vw,3.4rem); font-weight:750; letter-spacing:-.055em; line-height:1.14; margin:.65rem 0;}
.wave-subtitle {color:#bcb4d3; font-size:1.02rem; max-width:620px; line-height:1.65;}
.wave-section {color:#b9aed9; font-size:.83rem; margin:.8rem 0 .5rem;}
.wave-footer {color:#807895; font-size:.78rem; text-align:center; margin-top:2.2rem;}
/* Keep text inputs legible even with light browser theme preferences */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {border-radius:12px;}
@media (max-width:640px) {.block-container {padding-left:1rem;padding-right:1rem;} .wave-eyebrow{margin-top:1rem;}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    return VectorStore(collection_name=COLLECTION), RAGGraph()


try:
    vector_store, rag_graph = load_system()
except Exception as error:
    st.error(f"Unable to start WAVE: {error}")
    st.stop()

for key, default in (
    ("conversations", {}),
    ("thread_ids", {}),
    ("selected_domain", None),
    ("index_result", None),
):
    if key not in st.session_state:
        st.session_state[key] = default


def index_website(url: str, max_pages: int, scope: str):
    """Index in batches and report progress. The percentage is an upper-bound estimate."""
    url = url.strip()
    if not url or not url.startswith(("https://", "http://")):
        st.error("Enter a website URL starting with https:// or http://")
        return
    parsed = urlparse(url)
    if not parsed.netloc:
        st.error("Enter a valid website address.")
        return

    progress = st.progress(0, text="Preparing to index...")
    status = st.empty()
    pages_processed, chunks_processed = 0, 0
    batch = []
    initial_count = vector_store.collection.count()
    try:
        for page in iter_website_pages(url, max_pages=max_pages, delay=CRAWL_DELAY, scope=scope):
            batch.append(page)
            pages_processed += 1
            if len(batch) >= BATCH_SIZE:
                chunks = chunk_pages(batch)
                if chunks:
                    vector_store.add_chunks(chunks)
                    chunks_processed += len(chunks)
                batch.clear()
            progress.progress(min(pages_processed / max_pages, 1.0), text=f"Explored {pages_processed} of up to {max_pages} pages")
            status.caption(f"Current page: {page.get('url', 'Unknown')}")

        if batch:
            chunks = chunk_pages(batch)
            if chunks:
                vector_store.add_chunks(chunks)
                chunks_processed += len(chunks)

        if not pages_processed:
            progress.empty()
            status.empty()
            st.warning("No accessible pages were found at that address.")
            return

        domain = urlparse(normalize_url(url)).netloc.lower()
        st.session_state.selected_domain = domain
        st.session_state.index_result = {
            "domain": domain,
            "pages": pages_processed,
            "chunks": chunks_processed,
            "net_new": vector_store.collection.count() - initial_count,
        }
        progress.progress(1.0, text="Website ready")
        status.empty()
        st.success(f"{domain} is ready to explore!")
        st.rerun()
    except Exception as error:
        st.error(f"Indexing failed: {error}")
        if pages_processed:
            st.info(f"{pages_processed} pages were processed before the error. Completed batches may already be saved.")


# Header
header_left, header_right = st.columns([5, 1])
with header_left:
    st.markdown('<div class="wave-brand"><span class="wave-logo">≈</span><span>wave</span></div>', unsafe_allow_html=True)
with header_right:
    if st.button("＋ Website", use_container_width=True, help="Open the website indexing panel"):
        st.session_state.show_indexer = not st.session_state.get("show_indexer", False)
        st.rerun()

# Indexer in a collapsible sidebar, with an inline fallback panel when explicitly requested.
with st.sidebar:
    st.title("🌐 Add a website")
    st.caption("Turn a website into a searchable knowledge space.")
    with st.form("sidebar_index_form"):
        sidebar_url = st.text_input("Website URL", placeholder="https://docs.python.org/3/tutorial/")
        sidebar_pages = st.number_input("Maximum pages", min_value=1, max_value=10000, value=100, step=25)
        sidebar_mode = st.radio("Crawling scope", ["Entire website", "Current section only"], help="Crawling respects robots.txt.")
        sidebar_submit = st.form_submit_button("Index website", type="primary", use_container_width=True)
    if sidebar_submit:
        index_website(sidebar_url, int(sidebar_pages), "website" if sidebar_mode == "Entire website" else "section")
    st.divider()
    st.caption("Local AI · LangGraph · ChromaDB · Ollama")

if st.session_state.get("show_indexer", False):
    with st.expander("Add a new knowledge space", expanded=True):
        with st.form("inline_index_form"):
            website_url = st.text_input("Website URL", placeholder="https://docs.python.org/3/tutorial/")
            max_pages = st.number_input("Maximum pages to index", min_value=1, max_value=10000, value=100, step=25)
            crawl_mode = st.radio("Crawling mode", ["Entire website", "Current section only"], horizontal=True)
            submitted = st.form_submit_button("Start indexing", type="primary")
        if submitted:
            index_website(website_url, int(max_pages), "website" if crawl_mode == "Entire website" else "section")

try:
    domains = vector_store.get_domains()
except Exception as error:
    st.error(f"Could not load indexed websites: {error}")
    st.stop()

if domains:
    if st.session_state.selected_domain not in domains:
        st.session_state.selected_domain = domains[0]
    st.markdown('<div class="wave-section">YOUR SPACES</div>', unsafe_allow_html=True)
    # Native buttons provide keyboard navigation and a clear selected state.
    tabs = st.columns(min(len(domains), 4) + 1)
    for idx, domain in enumerate(domains):
        # Wrap after every four domains without truncating the list.
        if idx and idx % 4 == 0:
            tabs = st.columns(min(len(domains) - idx, 4) + 1)
        with tabs[idx % 4]:
            active = domain == st.session_state.selected_domain
            if st.button(("● " if active else "◦ ") + domain, key=f"domain_{idx}", type="primary" if active else "secondary", use_container_width=True):
                st.session_state.selected_domain = domain
                st.rerun()
else:
    st.markdown('<div class="wave-eyebrow">YOUR KNOWLEDGE SPACE</div>', unsafe_allow_html=True)
    st.markdown('<div class="wave-title">Explore beyond the page.</div>', unsafe_allow_html=True)
    st.markdown('<div class="wave-subtitle">Add a website, then ask questions and explore answers grounded in its content.</div>', unsafe_allow_html=True)
    st.info("No websites indexed yet. Select **＋ Website** above to create your first space.")
    st.stop()

selected_domain = st.session_state.selected_domain
if selected_domain not in st.session_state.conversations:
    st.session_state.conversations[selected_domain] = []
if selected_domain not in st.session_state.thread_ids:
    st.session_state.thread_ids[selected_domain] = rag_graph.new_thread_id()
messages = st.session_state.conversations[selected_domain]
thread_id = st.session_state.thread_ids[selected_domain]

if st.session_state.index_result:
    result = st.session_state.index_result
    if result["domain"] == selected_domain:
        st.success(f"Indexed {result['pages']} pages and generated {result['chunks']} chunks. Net new stored chunks: {result['net_new']}.")
        st.session_state.index_result = None

st.markdown('<div class="wave-eyebrow">EXPLORING ' + selected_domain.replace("<", "&lt;").replace(">", "&gt;") + '</div>', unsafe_allow_html=True)
if not messages:
    st.markdown('<div class="wave-title">Curiosity starts here.</div>', unsafe_allow_html=True)
    st.markdown('<div class="wave-subtitle">Ask anything about this website. Discover answers and trace them back to their sources.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="wave-title">Follow the conversation.</div>', unsafe_allow_html=True)

control_left, control_right = st.columns([3, 1])
with control_left:
    st.caption(f"🌐 {selected_domain} · {len(messages) // 2} answered question(s)")
with control_right:
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.conversations[selected_domain] = []
        st.session_state.thread_ids[selected_domain] = rag_graph.new_thread_id()
        st.rerun()


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📖 Explore sources ({len(sources)})"):
        for idx, source in enumerate(sources, 1):
            if isinstance(source, str):
                st.markdown(f"**{idx}.** {source}")
            elif isinstance(source, dict):
                url = source.get("source") or source.get("url")
                if url:
                    st.markdown(f"**{idx}.** {url}")
                preview = source.get("text") or source.get("content") or source.get("page_content")
                if preview:
                    st.caption(str(preview)[:500])
            else:
                st.write(source)


for message in messages:
    with st.chat_message(message["role"], avatar="🌊" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

question = st.chat_input(f"Ask anything about {selected_domain}...")
if question and question.strip():
    question = question.strip()
    messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🌊"):
        try:
            with st.spinner("Exploring your sources..."):
                result = rag_graph.ask(question=question, domain=selected_domain, thread_id=thread_id)
            answer = result["answer"]
            context = result.get("context", [])
            sources = []
            seen = set()
            if context and NOT_FOUND.lower() not in answer.lower():
                for chunk in context:
                    source = chunk.get("source") if isinstance(chunk, dict) else None
                    if source and source not in seen:
                        seen.add(source)
                        sources.append(source)
            st.markdown(answer)
            show_sources(sources)
            messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as error:
            st.error(f"Could not answer the question: {error}")
            # Remove the failed user question so UI and graph history do not diverge.
            messages.pop()

st.markdown('<div class="wave-footer">WAVE · Explore websites with local AI</div>', unsafe_allow_html=True)
