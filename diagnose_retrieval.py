
from RAG.Vectorstore import VectorStore
from RAG.reranker import Reranker


COLLECTIONS = {
    "V2": "website_chunks_v2",
    "V3": "website_chunks_v3",
}

TEST_CASES = [
    {
        "name": "Decorators",
        "question": "What is a Python decorator?",
        "source": "glossary.html",
        "phrases": [
            "A function returning another function",
        ],
    },
    {
        "name": "Function definitions",
        "question": "How do I define a function in Python?",
        "source": "tutorial/controlflow.html",
        "phrases": [
            "The keyword def introduces",
        ],
    },
    {
        "name": "Class methods",
        "question": "What does @classmethod do?",
        "source": "glossary.html",
        "phrases": [
            "class method",
        ],
    },
    {
        "name": "Lists and tuples",
        "question": "What is the difference between a list and a tuple?",
        "source": "tutorial/datastructures.html",
        "phrases": [
            "Tuples are immutable",
            "Lists are mutable",
        ],
    },
    {
        "name": "Exceptions",
        "question": "How does Python handle exceptions?",
        "source": "reference/executionmodel.html",
        "phrases": [
            "Exception handlers are specified",
        ],
    },
]


def is_relevant(chunk, case):
    source = chunk.get("source", "")
    text = chunk.get("text", "").lower()

    if not source.endswith(case["source"]):
        return False

    return all(
        phrase.lower() in text
        for phrase in case["phrases"]
    )


def find_rank(results, case):
    for index, chunk in enumerate(results, start=1):
        if is_relevant(chunk, case):
            return index

    return None


def show_stage(label, results, case):
    rank = find_rank(results, case)

    print(f"{label:20} ", end="")

    if rank is None:
        print("NOT FOUND")
    else:
        print(f"FOUND at rank {rank}")


def main():
    stores = {
        version: VectorStore(collection_name=collection)
        for version, collection in COLLECTIONS.items()
    }

    reranker = Reranker()

    for case in TEST_CASES:
        print("\n" + "=" * 75)
        print(case["name"])
        print("Question:", case["question"])
        print("=" * 75)

        for version, store in stores.items():
            print(f"\n{version}")

            semantic = store.search(
                query=case["question"],
                top_k=50,
            )

            keyword = store.keyword_search(
                query=case["question"],
                top_k=50,
            )

            hybrid = store.hybrid_search(
                query=case["question"],
                top_k=30,
                candidate_k=50,
            )

            reranked = reranker.rerank(
                query=case["question"],
                chunks=hybrid,
                top_k=30,
            )

            show_stage("Semantic top 50", semantic, case)
            show_stage("BM25 top 50", keyword, case)
            show_stage("Hybrid top 30", hybrid, case)
            show_stage("Reranked top 30", reranked, case)

            final_rank = find_rank(reranked, case)

            if final_rank is not None:
                print(
                    "Final top 5:",
                    "YES" if final_rank <= 5 else "NO",
                )
            else:
                print("Final top 5: NO")


if __name__ == "__main__":
    main()
