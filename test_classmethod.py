
from RAG.Vectorstore import VectorStore
from RAG.reranker import Reranker

QUERY = "What does @classmethod do in Python?"
COLLECTION = "website_chunks_v3"

def main():
    store = VectorStore(collection_name=COLLECTION)

    print("Collection:", COLLECTION)
    print("Total chunks:", store.count())

    print("\nRunning hybrid retrieval...")

    candidates = store.hybrid_search(
        query=QUERY,
        top_k=30,
        candidate_k=50,
    )

    print("Retrieved candidates:", len(candidates))

    reranker = Reranker()

    results = reranker.rerank(
        QUERY,
        candidates,
        top_k=5,
    )

    print("\nTop 5 reranked results:")

    found_rank = None

    for rank, result in enumerate(results, start=1):
        source = result.get("source", "")
        content = result.get("text", "")

        is_relevant = (
            "builtins/functions.html" in source
            and "class method receives the class" in content.lower()
        )

        if is_relevant:
            found_rank = rank

        print(f"\nRank {rank}")
        print("Source:", source)
        print("Relevant:", is_relevant)
        print("Preview:", content[:350])

    print("\nEvaluation")

    if found_rank:
        print("Relevant documentation found at rank:", found_rank)
        print("Hit@5: 1")
        print("Reciprocal Rank:", round(1 / found_rank, 3))
    else:
        print("Relevant documentation not found in top 5.")
        print("Hit@5: 0")
        print("Reciprocal Rank: 0")

if __name__ == "__main__":
    main()
