from RAG.graph import RAGGraph


rag = RAGGraph()

result = rag.ask(
    question="How do I define a function in Python?",
    domain="docs.python.org",
)

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")

sources = set()

for chunk in result["context"]:
    sources.add(
        chunk["source"]
    )

for source in sources:
    print(source)