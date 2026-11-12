
from RAG.Vectorstore import VectorStore
from RAG.reranker import Reranker

COLLECTIONS = {
    "V2": "website_chunks_v2",
    "V3": "website_chunks_v3",
}

TEST_CASES = [
    (
        "Decorators",
        "What is a Python decorator?",
        "glossary.html",
        ["A function returning another function"],
    ),
    (
        "Function definitions",
        "How do I define a function in Python?",
        "tutorial/controlflow.html",
        ["The keyword def introduces"],
    ),
    (
        "Lists and tuples",
        "What is the difference between a list and a tuple?",
        "tutorial/datastructures.html",
        ["Tuples are immutable", "Lists are mutable"],
    ),
    (
        "Exceptions",
        "How does Python handle exceptions?",
        "reference/executionmodel.html",
        ["Exception handlers are specified"],
    ),
]

STRATEGIES = {
    "Semantic only": (1.0, 0.0),
    "Hybrid 50/50": (0.5, 0.5),
    "Hybrid 70/30": (0.7, 0.3),
    "Hybrid 80/20": (0.8, 0.2),
}

CANDIDATE_K = 50
RERANK_K = 5
RRF_K = 60


def is_relevant(chunk, case):
    _, _, source, phrases = case

    if not chunk.get("source", "").endswith(source):
        return False

    text = chunk.get("text", "").lower()

    return all(phrase.lower() in text for phrase in phrases)


def weighted_fusion(semantic, keyword, ws, wk):
    fused = {}

    for results, weight in ((semantic, ws), (keyword, wk)):
        if weight <= 0:
            continue

        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **chunk,
                    "rrf_score": 0.0,
                }

            fused[chunk_id]["rrf_score"] += (
                weight / (RRF_K + rank)
            )

    return sorted(
        fused.values(),
        key=lambda c: c["rrf_score"],
        reverse=True,
    )[:CANDIDATE_K]


def reciprocal_rank(results, case):
    for rank, chunk in enumerate(results, start=1):
        if is_relevant(chunk, case):
            return 1.0 / rank
    return 0.0


def main():
    reranker = Reranker()

    for version, collection in COLLECTIONS.items():
        store = VectorStore(collection_name=collection)

        print("\n" + "=" * 65)
        print(f"COLLECTION: {version}")
        print("=" * 65)

        scores = {name: [] for name in STRATEGIES}

        for case in TEST_CASES:
            name, question, _, _ = case
            print(f"\nQuestion: {name}")

            semantic = store.search(
                query=question,
                top_k=CANDIDATE_K,
            )

            keyword = store.keyword_search(
                query=question,
                top_k=CANDIDATE_K,
            )

            for strategy, (ws, wk) in STRATEGIES.items():
                candidates = weighted_fusion(
                    semantic, keyword, ws, wk
                )

                results = reranker.rerank(
                    query=question,
                    chunks=candidates,
                    top_k=RERANK_K,
                )

                rr = reciprocal_rank(results, case)
                scores[strategy].append(rr)

                rank = int(1 / rr) if rr else "Not found"
                print(f"  {strategy:16} Rank: {rank}")

        print("\nSUMMARY")
        for strategy, values in scores.items():
            mrr = sum(values) / len(values)
            hits = sum(value > 0 for value in values)

            print(
                f"{strategy:16} "
                f"Hit@5: {hits}/{len(values)} "
                f"MRR@5: {mrr:.3f}"
            )


if __name__ == "__main__":
    main()
