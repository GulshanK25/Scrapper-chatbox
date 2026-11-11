
import time

import requests

from scrapper.crawler import (
    fetch_page,
    load_robots,
    normalize_url,
    HEADERS,
    USER_AGENT,
)
from RAG.chunker import chunk_pages
from RAG.Vectorstore import VectorStore


SOURCE_COLLECTION = "website_chunks_v2"
TARGET_COLLECTION = "website_chunks_v3"

BATCH_SIZE = 10
REQUEST_DELAY = 0.5


def get_indexed_urls(store):
    """
    Get the distinct source URLs from V2.
    """
    results = store.collection.get(
        include=["metadatas"]
    )

    urls = {
        metadata["source"]
        for metadata in results["metadatas"]
        if metadata and metadata.get("source")
    }

    return sorted(urls)


def main():
    print("Loading V2 collection...")
    v2 = VectorStore(
        collection_name=SOURCE_COLLECTION
    )

    urls = get_indexed_urls(v2)

    print("V2 source pages:", len(urls))

    if not urls:
        print("No source URLs found. Stopping.")
        return

    print("\nPreparing V3 collection...")
    v3 = VectorStore(
        collection_name=TARGET_COLLECTION
    )

    print("Existing V3 chunks:", v3.count())

    indexed_urls = get_indexed_urls(v3)
    indexed_set = set(indexed_urls)

    remaining_urls = [
        url for url in urls
        if url not in indexed_set
    ]

    print("Pages remaining:", len(remaining_urls))

    with requests.Session() as session:
        session.headers.update(HEADERS)

        robots_cache = {}

        for index, url in enumerate(
            remaining_urls,
            start=1,
        ):
            parsed_origin = (
                url.split("/", 3)[0]
                + "//"
                + url.split("/", 3)[2]
            )

            if parsed_origin not in robots_cache:
                robots, _ = load_robots(
                    session,
                    url,
                )
                robots_cache[parsed_origin] = robots

            robots = robots_cache[parsed_origin]

            if robots is None:
                print(
                    "Skipping: robots.txt unavailable",
                    url,
                )
                continue

            if not robots.can_fetch(
                USER_AGENT,
                url,
            ):
                print("Disallowed by robots.txt:", url)
                continue

            print(
                f"\n[{index}/{len(remaining_urls)}] "
                f"Fetching {url}"
            )

            try:
                text, _ = fetch_page(
                    session,
                    normalize_url(url),
                )

                if not text:
                    print("No content extracted.")
                    continue

                chunks = chunk_pages([
                    {
                        "url": url,
                        "text": text,
                    }
                ])

                if not chunks:
                    print("No chunks generated.")
                    continue

                v3.add_chunks(chunks)

                print(
                    "Indexed chunks:",
                    len(chunks),
                )

            except requests.RequestException as error:
                print("Request failed:", error)

            except Exception as error:
                print("Indexing failed:", error)

            time.sleep(REQUEST_DELAY)

    print("\n" + "=" * 60)
    print("V3 indexing finished.")
    print("Total V3 chunks:", v3.count())
    print(
        "Total V3 pages:",
        len(get_indexed_urls(v3)),
    )


if __name__ == "__main__":
    main()
