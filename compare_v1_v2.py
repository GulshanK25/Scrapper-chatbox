
from RAG.Vectorstore import VectorStore
from RAG.reranker import Reranker


DOMAIN = "docs.python.org"

v1 = VectorStore(
    collection_name="website_chunks"
)

v2 = VectorStore(
    collection_name="website_chunks_v2"
)

reranker = Reranker()


QUESTIONS = [
    "What is a Python decorator?",
    "How do I define a function in Python?",
    "What does @classmethod do?",
    "What is the difference between a list and a tuple?",
    "How does Python handle exceptions?",
]


def get_sources(store):
    sources = set()
    offset = 0

    while True:
        result = store.collection.get(
            where={"domain": DOMAIN},
            include=["metadatas"],
            limit=500,
            offset=offset,
        )

        if not result["ids"]:
            break

        for metadata in result["metadatas"]:
            if metadata:
                sources.add(metadata["source"])

        offset += len(result["ids"])

    return sources


sources_v1 = get_sources(v1)
sources_v2 = get_sources(v2)

print("\n=== COLLECTION COMPARISON ===")
print("V1 chunks:", v1.count())
print("V2 chunks:", v2.count())
print("V1 pages:", len(sources_v1))
print("V2 pages:", len(sources_v2))

print("\nPages only in V1:")
for source in sorted(sources_v1 - sources_v2):
    print(source)

print("\nPages only in V2:")
for source in sorted(sources_v2 - sources_v1):
    print(source)


def test_collection(name, store, question):
    candidates = store.hybrid_search(
        query=question,
        top_k=30,
        candidate_k=50,
        domain=DOMAIN,
    )

    results = reranker.rerank(
        query=question,
        chunks=candidates,
        top_k=5,
    )

    print(f"\n{name}")

    for index, chunk in enumerate(results, 1):
        print(f"\n{index}. {chunk['source']}")
        print("Chunk ID:", chunk["chunk_id"])
        print(
            "Rerank score:",
            round(chunk["rerank_score"], 3),
        )
        print("Text:", chunk["text"][:450])


for question in QUESTIONS:
    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    test_collection("V1 RESULTS", v1, question)
    test_collection("V2 RESULTS", v2, question)
