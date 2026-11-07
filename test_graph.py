
from RAG.graph import RAGGraph

rag = RAGGraph()

tests = [
    "How do I define a function in Python?",
    "Who is the president of France?",
]

for question in tests:
    print("\n" + "=" * 60)
    print("QUESTION:", question)

    result = rag.ask(
        question=question,
        domain="docs.python.org",
    )

    print("RELEVANCE:", result["relevance"])
    print("RETRIES:", result["retry_count"])
    print("SEARCH QUERY:", result["search_query"])
    print("ANSWER:", result["answer"])

    if result["relevance"] == "yes":
        print("SOURCES:")
        for source in sorted({
            chunk["source"]
            for chunk in result["context"]
        }):
            print(source)
