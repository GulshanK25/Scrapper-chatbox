import streamlit as st
from urllib.parse import urlparse

from scrapper.crawler import crawl_website
from RAG.chunker import chunk_pages
from RAG.Vectorstore import VectorStore
from RAG.generator import Generator


st.set_page_config(
    page_title="Website RAG Assistant",
    page_icon="🔎",
    layout="centered",
)

st.title("🔎 Website RAG Assistant")

st.write(
    "Enter a website, index its content, and ask questions "
    "using a fully local RAG pipeline powered by Llama 3."
)


@st.cache_resource
def load_system():
    vector_store = VectorStore()
    generator = Generator()

    return vector_store, generator


vector_store, generator = load_system()


# -----------------------------
# WEBSITE INDEXING
# -----------------------------

st.subheader("1. Index a Website")

website_url = st.text_input(
    "Website URL",
    placeholder="https://docs.python.org/3/tutorial/",
)


if st.button("Index Website"):

    if not website_url.strip():

        st.warning(
            "Please enter a website URL."
        )

    elif not website_url.startswith(
        ("http://", "https://")
    ):

        st.error(
            "Please enter a valid URL beginning "
            "with http:// or https://"
        )

    else:

        try:

            with st.spinner(
                "Crawling and indexing website..."
            ):

                pages = crawl_website(
                    website_url,
                    max_pages=10,
                )

                if not pages:

                    st.error(
                        "No website content could be retrieved."
                    )

                else:

                    chunks = chunk_pages(pages)

                    vector_store.add_chunks(
                        chunks
                    )

                    domain = urlparse(
                        website_url
                    ).netloc

                    st.session_state[
                        "selected_domain"
                    ] = domain

                    st.success(
                        f"Website indexed successfully. "
                        f"{len(pages)} pages and "
                        f"{len(chunks)} chunks processed."
                    )

        except Exception as error:

            st.error(
                f"Website indexing failed: {error}"
            )


# -----------------------------
# WEBSITE SELECTION
# -----------------------------

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


if default_domain in domains:

    default_index = domains.index(
        default_domain
    )

else:

    default_index = 0


selected_domain = st.selectbox(
    "Indexed website",
    domains,
    index=default_index,
)

st.session_state[
    "selected_domain"
] = selected_domain


# -----------------------------
# QUESTION ANSWERING
# -----------------------------

st.divider()

st.subheader("3. Ask Questions")

question = st.text_input(
    "Question",
    placeholder="What does this website say about...?",
)


if st.button("Ask Question"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching website..."
        ):

            results = vector_store.search(
                question,
                top_k=3,
                domain=selected_domain,
            )

            if not results:

                st.warning(
                    "No relevant information was found."
                )

            else:

                answer = generator.generate(
                    question,
                    results,
                )

                st.subheader("Answer")

                st.write(answer)

                not_found = (
                    "I could not find this "
                    "information on the website."
                )

                if (
                    not_found.lower()
                    not in answer.lower()
                ):

                    st.subheader("Sources")

                    sources = []

                    for result in results:

                        source = result[
                            "source"
                        ]

                        if source not in sources:
                            sources.append(
                                source
                            )

                    for source in sources:
                        st.write(source)