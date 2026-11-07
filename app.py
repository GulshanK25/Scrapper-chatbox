
import streamlit as st
from urllib.parse import urlparse

from scrapper.crawler import (
    iter_website_pages,
    normalize_url,
)
from RAG.chunker import chunk_pages
from RAG.Vectorstore import VectorStore
from RAG.graph import RAGGraph


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Website RAG Assistant",
    page_icon="🔎",
    layout="centered",
)

st.title("🔎 Website RAG Assistant")

st.write(
    "Index website content and chat with it using "
    "LangGraph, ChromaDB, and locally hosted Llama 3."
)


# --------------------------------
# CONFIGURATION
# --------------------------------

BATCH_SIZE = 20
CRAWL_DELAY = 0.5

NOT_FOUND = (
    "I could not find this information on the website."
)


@st.cache_resource
def load_system():
    vector_store = VectorStore()
    rag_graph = RAGGraph()

    return vector_store, rag_graph


vector_store, rag_graph = load_system()


# --------------------------------
# SESSION STATE
# --------------------------------

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "thread_ids" not in st.session_state:
    st.session_state.thread_ids = {}

if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = None


# --------------------------------
# WEBSITE INDEXING
# --------------------------------

st.subheader("1. Index a Website")

website_url = st.text_input(
    "Website URL",
    placeholder="https://docs.python.org/3/tutorial/",
)

max_pages = st.number_input(
    "Maximum pages to index",
    min_value=1,
    max_value=10000,
    value=100,
    step=50,
)


# --------------------------------
# CRAWLING MODE
# --------------------------------

crawl_mode = st.radio(
    "Crawling Mode",
    options=[
        "Current section only",
        "Entire website",
    ],
    index=1,
    help=(
        "Current section only indexes pages within "
        "the starting URL path. Entire website explores "
        "other permitted sections of the website."
    ),
)

crawl_scope = (
    "website"
    if crawl_mode == "Entire website"
    else "section"
)

st.caption(
    "Pages are discovered from sitemaps and internal links. "
    "Choose whether to index the current section or explore "
    "the wider website. Crawling respects robots.txt."
)


# --------------------------------
# INDEXING PROCESS
# --------------------------------

if st.button("Index Website", type="primary"):

    website_url = website_url.strip()

    if not website_url:
        st.warning("Please enter a website URL.")

    elif not website_url.startswith(
        ("http://", "https://")
    ):
        st.error(
            "Please enter a valid URL beginning "
            "with http:// or https://"
        )

    else:

        progress_bar = st.progress(0)
        status = st.empty()
        stats = st.empty()

        pages_processed = 0
        chunks_processed = 0
        batch = []

        initial_chunk_count = (
            vector_store.collection.count()
        )

        try:

            with st.spinner("Indexing website..."):

                for page in iter_website_pages(
                    website_url,
                    max_pages=int(max_pages),
                    delay=CRAWL_DELAY,
                    scope=crawl_scope,
                ):

                    batch.append(page)
                    pages_processed += 1

                    # Save completed batches
                    if len(batch) >= BATCH_SIZE:

                        chunks = chunk_pages(batch)

                        if chunks:
                            vector_store.add_chunks(chunks)
                            chunks_processed += len(chunks)

                        batch.clear()

                    progress = min(
                        pages_processed / int(max_pages),
                        1.0,
                    )

                    progress_bar.progress(progress)

                    status.info(
                        f"Processing: {page['url']}"
                    )

                    stats.write(
                        f"**Pages processed:** {pages_processed}  "
                        f"| **Chunks generated:** {chunks_processed}"
                    )

                # Save remaining pages
                if batch:

                    chunks = chunk_pages(batch)

                    if chunks:
                        vector_store.add_chunks(chunks)
                        chunks_processed += len(chunks)

                    batch.clear()

                # --------------------------------
                # FINAL INDEXING RESULTS
                # --------------------------------

                final_chunk_count = (
                    vector_store.collection.count()
                )

                chunk_difference = (
                    final_chunk_count
                    - initial_chunk_count
                )

                if pages_processed == 0:

                    st.warning(
                        "No accessible website pages "
                        "were found."
                    )

                else:

                    domain = urlparse(
                        normalize_url(website_url)
                    ).netloc.lower()

                    st.session_state[
                        "selected_domain"
                    ] = domain

                    progress_bar.progress(1.0)

                    st.success(
                        "Website indexing completed successfully!"
                    )

                    st.write(
                        f"**Crawling mode:** {crawl_mode}"
                    )

                    st.write(
                        f"**Pages processed:** {pages_processed}"
                    )

                    st.write(
                        f"**Chunks generated:** {chunks_processed}"
                    )

                    st.write(
                        f"**New net chunks in ChromaDB:** "
                        f"{chunk_difference}"
                    )

                    st.write(
                        f"**Total chunks stored:** "
                        f"{final_chunk_count}"
                    )

                    if (
                        chunks_processed > 0
                        and chunk_difference == 0
                    ):
                        st.warning(
                            "Chunks were generated, but the "
                            "collection count did not increase. "
                            "Existing records may have been "
                            "overwritten or deduplicated. "
                            "Check Vectorstore.py."
                        )

                    stats.write(
                        f"**Final result:** "
                        f"{pages_processed} pages processed, "
                        f"{chunks_processed} chunks generated."
                    )

                    status.empty()

        except Exception as error:

            st.error(
                f"Website indexing failed: {error}"
            )

            if pages_processed:
                st.info(
                    f"{pages_processed} pages were processed "
                    "before the error. Previously completed "
                    "batches may already be stored."
                )


# --------------------------------
# WEBSITE SELECTION
# --------------------------------

st.divider()

st.subheader("2. Select Website")

domains = vector_store.get_domains()

if not domains:

    st.info(
        "Index a website above to start chatting."
    )

    st.stop()


default_domain = st.session_state.get(
    "selected_domain"
)

default_index = (
    domains.index(default_domain)
    if default_domain in domains
    else 0
)

selected_domain = st.selectbox(
    "Indexed website",
    domains,
    index=default_index,
)

st.session_state[
    "selected_domain"
] = selected_domain


# --------------------------------
# CONVERSATION MANAGEMENT
# --------------------------------

if selected_domain not in st.session_state.conversations:
    st.session_state.conversations[selected_domain] = []

if selected_domain not in st.session_state.thread_ids:
    st.session_state.thread_ids[
        selected_domain
    ] = rag_graph.new_thread_id()


messages = st.session_state.conversations[
    selected_domain
]

thread_id = st.session_state.thread_ids[
    selected_domain
]


# --------------------------------
# CHAT INTERFACE
# --------------------------------

st.divider()

st.subheader("3. Chat with Website")

st.caption(
    f"Currently chatting with: {selected_domain}"
)


if st.button("New Conversation"):

    st.session_state.conversations[
        selected_domain
    ] = []

    st.session_state.thread_ids[
        selected_domain
    ] = rag_graph.new_thread_id()

    st.rerun()


# --------------------------------
# DISPLAY CONVERSATION HISTORY
# --------------------------------

for message in messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("Sources"):

                for source in message["sources"]:
                    st.write(source)


# --------------------------------
# QUESTION ANSWERING
# --------------------------------

question = st.chat_input(
    "Ask a question about this website..."
)

if question:

    messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner("Thinking..."):

                result = rag_graph.ask(
                    question=question,
                    domain=selected_domain,
                    thread_id=thread_id,
                )

            answer = result["answer"]
            context = result.get("context", [])

            st.markdown(answer)

            sources = []

            if (
                context
                and NOT_FOUND.lower()
                not in answer.lower()
            ):

                for chunk in context:

                    source = chunk.get("source")

                    if source and source not in sources:
                        sources.append(source)

                if sources:

                    with st.expander("Sources"):

                        for source in sources:
                            st.write(source)

            messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:

            st.error(
                f"Question answering failed: {error}"
            )
