
from RAG.Vectorstore import VectorStore

COLLECTIONS = {
    "V2": "website_chunks_v2",
    "V3": "website_chunks_v3",
}

PAGES = [
    "https://docs.python.org/3/library/functions.html",
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/glossary.html",
]


def main():
    for version, collection in COLLECTIONS.items():
        store = VectorStore(collection_name=collection)

        print(f"\n{'=' * 60}")
        print(f"{version}: {store.count()} chunks")
        print("=" * 60)

        for url in PAGES:
            results = store.collection.get(
                where={"source": url},
                include=["documents"],
            )

            documents = results.get("documents", [])

            print(f"\nPage: {url}")
            print(f"Chunks: {len(documents)}")

            if not documents:
                print("STATUS: NOT INDEXED")
                continue

            terms = ["classmethod", "def introduces"]

            for term in terms:
                matches = [
                    doc for doc in documents
                    if term.lower() in doc.lower()
                ]

                print(
                    f"Contains '{term}': "
                    f"{len(matches)} chunks"
                )


if __name__ == "__main__":
    main()
