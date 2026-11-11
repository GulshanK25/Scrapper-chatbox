
from scrapper.crawler import scrape_page


TEST_PAGES = {
    "Function definitions":
        "https://docs.python.org/3/tutorial/controlflow.html",

    "Decorators":
        "https://docs.python.org/3/glossary.html",

    "Lists and tuples":
        "https://docs.python.org/3/tutorial/datastructures.html",
}


for topic, url in TEST_PAGES.items():
    print("\n" + "=" * 70)
    print("TEST:", topic)
    print("URL:", url)
    print("=" * 70)

    text, links = scrape_page(url)

    print("\nTotal characters:", len(text))
    print("Discovered links:", len(links))

    if not text:
        print("No content extracted.")
        continue

    if topic == "Function definitions":
        keywords = [
            "Defining Functions",
            "def ",
        ]

    elif topic == "Decorators":
        keywords = [
            "decorator",
            "function returning another function",
        ]

    else:
        keywords = [
            "Tuples and Sequences",
            "immutable",
        ]

    for keyword in keywords:
        position = text.lower().find(
            keyword.lower()
        )

        print(
            f"\nKeyword: {keyword}"
        )

        if position == -1:
            print("NOT FOUND")
            continue

        start = max(0, position - 150)
        end = min(
            len(text),
            position + 650,
        )

        print(text[start:end])

    print("\nFIRST 500 CHARACTERS:")
    print(text[:500])
