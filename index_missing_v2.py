

import time

import requests

from RAG.Vectorstore import VectorStore
from RAG.chunker import chunk_pages
from scrapper.crawler import (
    HEADERS,
    USER_AGENT,
    fetch_page,
    load_robots,
    normalize_url,
)


DOMAIN = "docs.python.org"
DELAY = 0.5

v1 = VectorStore(
    collection_name="website_chunks"
)

v2 = VectorStore(
    collection_name="website_chunks_v2"
)


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


def main():
    sources_v1 = get_sources(v1)
    sources_v2 = get_sources(v2)

    missing_urls = sorted(
        sources_v1 - sources_v2
    )

    print("V1 pages:", len(sources_v1))
    print("V2 pages:", len(sources_v2))
    print("Missing pages:", len(missing_urls))

    if not missing_urls:
        print("No missing pages to index.")
        return

    pages = []

    with requests.Session() as session:
        session.headers.update(HEADERS)

        robots, _ = load_robots(
            session,
            "https://docs.python.org/3/",
        )

        if robots is None:
            print(
                "Cannot verify robots.txt. "
                "Stopping safely."
            )
            return

        for index, url in enumerate(
            missing_urls,
            start=1,
        ):
            url = normalize_url(url)

            if not robots.can_fetch(
                USER_AGENT,
                url,
            ):
                print("Disallowed:", url)
                continue

            print(
                f"[{index}/{len(missing_urls)}] "
                f"Fetching {url}"
            )

            try:
                text, _ = fetch_page(
                    session,
                    url,
                )

                if not text:
                    print("No content:", url)
                    continue

                pages.append({
                    "url": url,
                    "text": text,
                })

            except requests.RequestException as error:
                print(
                    f"Failed: {url}: {error}"
                )

            time.sleep(DELAY)

    if not pages:
        print("No pages available to index.")
        return

    chunks = chunk_pages(pages)

    print("\nCreating embeddings...")
    print("Pages fetched:", len(pages))
    print("Chunks generated:", len(chunks))

    v2.add_chunks(chunks)

    print("\nIndexing completed.")
    print("V2 total chunks:", v2.count())

    updated_sources = get_sources(v2)

    still_missing = sources_v1 - updated_sources

    print(
        "Pages still missing:",
        len(still_missing),
    )

    for url in sorted(still_missing):
        print("Still missing:", url)


if __name__ == "__main__":
    main()
