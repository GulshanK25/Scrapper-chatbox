
from RAG.Vectorstore import VectorStore


store = VectorStore()

questions = [
    "What is a Python decorator?",
    "How do I define a function in Python?",
    "What does @classmethod do?",
]

for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    results = store.hybrid_search(
        query=question,
        top_k=5,
        domain="docs.python.org",
    )

    for index, chunk in enumerate(
        results,
        start=1,
    ):

        print(f"\nResult {index}")
        print("Source:", chunk["source"])
        print(
            "RRF score:",
            round(chunk["rrf_score"], 5),
        )
        print(
            "Methods:",
            chunk["retrieval_methods"],
        )
        print(
            "Text:",
            chunk["text"][:350],
        )
