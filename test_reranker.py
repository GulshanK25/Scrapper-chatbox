
from RAG.Vectorstore import VectorStore
from RAG.reranker import Reranker


store = VectorStore()
reranker = Reranker()

questions = [
    "What is a Python decorator?",
    "How do I define a function in Python?",
    "What does @classmethod do?",
]

for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    candidates = store.hybrid_search(
        query=question,
        top_k=20,
        candidate_k=30,
        domain="docs.python.org",
    )

    results = reranker.rerank(
        query=question,
        chunks=candidates,
        top_k=5,
    )

    for index, chunk in enumerate(
        results,
        start=1,
    ):
        print(f"\nResult {index}")
        print("Source:", chunk["source"])
        print(
            "Rerank score:",
            round(chunk["rerank_score"], 4),
        )
        print("Text:", chunk["text"][:350])
