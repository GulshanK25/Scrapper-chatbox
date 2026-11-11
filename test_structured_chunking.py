
from scrapper.crawler import scrape_page
from RAG.chunker import chunk_text


TEST_PAGES = {
    "Functions":
        "https://docs.python.org/3/tutorial/controlflow.html",
    "Decorators":
        "https://docs.python.org/3/glossary.html",
    "Data structures":
        "https://docs.python.org/3/tutorial/datastructures.html",
}


def inspect_chunks(topic, url):
    print("\n" + "=" * 70)
    print("TOPIC:", topic)
    print("=" * 70)

    text, _ = scrape_page(url)

    if not text:
        print("No content extracted.")
        return

    chunks = chunk_text(text)

    print("Original characters:", len(text))
    print("Total chunks:", len(chunks))

    oversized = [
        chunk for chunk in chunks
        if len(chunk) > 1000
    ]

    print("Oversized chunks:", len(oversized))

    if topic == "Functions":
        search_terms = [
            "The keyword def introduces",
            "Defining Functions",
        ]

    elif topic == "Decorators":
        search_terms = [
            "### decorator",
            "A function returning another function",
        ]

    else:
        search_terms = [
            "Tuples are immutable",
            "Lists are mutable",
        ]

    for term in search_terms:
        matching = [
            (index, chunk)
            for index, chunk in enumerate(chunks)
            if term.lower() in chunk.lower()
        ]

        print("\nSearch term:", term)
        print("Matching chunks:", len(matching))

        for index, chunk in matching[:2]:
            print(f"\nCHUNK {index}")
            print("Length:", len(chunk))
            print(chunk[:1200])

    broken_fences = [
        index
        for index, chunk in enumerate(chunks)
        if chunk.count("```") % 2 != 0
    ]

    print(
        "\nChunks with unmatched code fences:",
        len(broken_fences),
    )

    if broken_fences:
        print("Chunk indices:", broken_fences[:10])


for topic, url in TEST_PAGES.items():
    inspect_chunks(topic, url)
