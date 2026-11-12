
from RAG.Vectorstore import VectorStore
from RAG.reranker import Reranker


QUESTIONS = [
    "What is a Python decorator?",
    "How do I define a function in Python?",
    "What does @classmethod do?",
    "What is the difference between a list and a tuple?",
    "How does Python handle exceptions?",
]

COLLECTIONS = {
    "V2": "website_chunks_v2",
    "V3": "website_chunks_v3",
}


def evaluate_collection(store, reranker, question):
    candidates = store.hybrid_search(
        query=question,
        top_k=30,
        candidate_k=50,
    )

    return reranker.rerank(
        query=question,
        chunks=candidates,
        top_k=5,
    )


def display_results(version, results):
    print(f"\n{version} TOP 5 RESULTS")

    if not results:
        print("No results.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("Source:", result.get("source", "Unknown"))
        print(
            "Rerank score:",
            round(result.get("rerank_score", 0), 4),
        )
        print(
            "Text:",
            result.get("text", "")[:650],
        )


def main():
    print("Loading collections...")

    stores = {
        name: VectorStore(collection_name=collection)
        for name, collection in COLLECTIONS.items()
    }

    print("\nLoading reranker...")
    reranker = Reranker()

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        for version, store in stores.items():
            results = evaluate_collection(
                store,
                reranker,
                question,
            )

            display_results(version, results)


if __name__ == "__main__":
    main()
