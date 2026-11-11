
from RAG.Vectorstore import VectorStore
from RAG.reranker import Reranker


store = VectorStore()
reranker = Reranker()

questions = [
    "What is a Python decorator?",
    "How do I define a function in Python?",
    "What does @classmethod do?",
]

domain = "docs.python.org"


def display_results(title, chunks, limit=5):
    print(f"\n{title}")
    print("-" * 60)

    for index, chunk in enumerate(chunks[:limit], 1):
        print(f"\n{index}. {chunk['source']}")
        print("Chunk ID:", chunk.get("chunk_id"))
        print("Text:", chunk["text"][:500])


for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    semantic = store.search(
        query=question,
        top_k=5,
        domain=domain,
    )

    hybrid = store.hybrid_search(
        query=question,
        top_k=30,
        candidate_k=50,
        domain=domain,
    )

    reranked = reranker.rerank(
        query=question,
        chunks=hybrid,
        top_k=5,
    )

    display_results(
        "SEMANTIC SEARCH",
        semantic,
    )

    display_results(
        "HYBRID SEARCH",
        hybrid,
    )

    display_results(
        "HYBRID + RERANKING",
        reranked,
    )
