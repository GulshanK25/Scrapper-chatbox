
import requests

from scrapper.crawler import fetch_page
from RAG.chunker import chunk_pages
from RAG.Vectorstore import VectorStore


URL = "https://docs.python.org/3/builtins/functions.html"
COLLECTION = "website_chunks_v3"


def main():
    print("Fetching missing documentation page...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "WebsiteRAGAssistant/1.0"
    })

    try:
        page = fetch_page(session, URL)

        if not page:
            print("Failed to fetch page.")
            return

        # Expected return format: (text, links)
        if (
            not isinstance(page, tuple)
            or len(page) != 2
            or not isinstance(page[0], str)
            or not isinstance(page[1], list)
        ):
            print("Unexpected fetch_page return format.")
            return

        text, links = page

        if not text.strip():
            print("Page contains no extracted text.")
            return

        print("Page fetched successfully.")
        print("Extracted characters:", len(text))
        print("Discovered links:", len(links))

        # Convert to the dictionary format
        # expected by chunk_pages()
        page_data = {
            "url": URL,
            "text": text,
            "title": "Built-in Functions",
        }

        chunks = chunk_pages([page_data])

        if not chunks:
            print("No chunks generated.")
            return

        print(f"Generated {len(chunks)} chunks.")

        # Verify that the expected topic was extracted
        matching_chunks = [
            chunk for chunk in chunks
            if "classmethod" in chunk["text"].lower()
        ]

        print(
            "Generated chunks mentioning classmethod:",
            len(matching_chunks)
        )

        if not matching_chunks:
            print(
                "Warning: No classmethod content found. "
                "Skipping indexing."
            )
            return

        store = VectorStore(
            collection_name=COLLECTION
        )

        print("Current chunk count:", store.count())

        store.add_chunks(chunks)

        print("Updated chunk count:", store.count())

        # Verify stored content
        results = store.collection.get(
            where={"source": URL},
            include=["documents"],
        )

        documents = results.get("documents", [])

        matches = [
            doc for doc in documents
            if "classmethod" in doc.lower()
        ]

        print("\nVerification")
        print("Indexed page chunks:", len(documents))
        print("Chunks mentioning classmethod:", len(matches))

        for match in matches[:2]:
            print("\n--- Matching chunk ---")
            print(match[:700])

    finally:
        session.close()


if __name__ == "__main__":
    main()
