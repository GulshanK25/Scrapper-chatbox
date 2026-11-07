
from RAG.Vectorstore import VectorStore

vector_store = VectorStore()

questions = [
    "What is a Python decorator?",
    "How do I define a function in Python?",
]

for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    results = vector_store.search(
        question,
        top_k=10,
        domain="docs.python.org",
    )

    print("Chunks retrieved:", len(results))

    for index, chunk in enumerate(results, start=1):

        print(f"\n--- RESULT {index} ---")

        print("Source:", chunk.get("source"))

        text = chunk.get(
            "text",
            chunk.get("document", "")
        )

        print("Content:", str(text)[:600])
