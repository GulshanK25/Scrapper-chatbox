
from RAG.Vectorstore import VectorStore
from RAG.reranker import Reranker

QUERY = "How do you define a function in Python?"

store = VectorStore(
    collection_name="website_chunks_v3"
)

candidates = store.hybrid_search(
    query=QUERY,
    top_k=50,
    candidate_k=75,
)

reranker = Reranker()

results = reranker.rerank(
    QUERY,
    candidates,
    top_k=15,
)

print("\nFUNCTION DEFINITION RETRIEVAL")
print("=" * 50)

for rank, result in enumerate(results, start=1):
    text = result.get("text", "")
    source = result.get("source", "")

    is_target = (
        "tutorial/controlflow.html" in source
        and "the keyword def introduces" in text.lower()
    )

    print(f"\nRank {rank}")
    print("Target passage:", is_target)
    print("Source:", source)
    print("Content:")
    print(text[:1200])
    print("-" * 60)

    if is_target:
        print("*** EXACT TARGET FOUND ***")
