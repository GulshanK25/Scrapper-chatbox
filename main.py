from RAG.Vectorstore import VectorStore
from RAG.generator import Generator


vector_store = VectorStore()
generator = Generator()

print("\n-------------------------")
print("LOCAL WEBSITE RAG CHATBOT")
print("-------------------------")

while True:

    question = input(
        "\nAsk a question (or type 'exit'): "
    )

    if question.lower() == "exit":
        break

    results = vector_store.search(
        question,
        top_k=3,
    )

    answer = generator.generate(
        question,
        results,
    )

    print("\n-------------------------")
    print("ANSWER")
    print("-------------------------")

    print(answer)

    # Only display sources if the model found an answer.
    if (
        "I could not find this information on the website."
        not in answer
    ):
        print("\nSources:")

        unique_sources = []

        for result in results:

            source = result["source"]

            if source not in unique_sources:
                unique_sources.append(source)

        for source in unique_sources:
            print("-", source)