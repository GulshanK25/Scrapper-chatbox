
from RAG.Vectorstore import VectorStore
from RAG.chunker import chunk_pages
from scrapper.crawler import iter_website_pages


WEBSITE_URL = "https://docs.python.org/3/tutorial/"
MAX_PAGES = 100
BATCH_SIZE = 20

store = VectorStore(
    collection_name="website_chunks_v2"
)

print("Starting experimental indexing...")
print("Existing V2 chunks:", store.count())

batch = []
pages_processed = 0
chunks_processed = 0


def store_batch():
    global chunks_processed

    if not batch:
        return

    chunks = chunk_pages(batch)

    if chunks:
        store.add_chunks(chunks)
        chunks_processed += len(chunks)

    batch.clear()


for page in iter_website_pages(
    WEBSITE_URL,
    max_pages=MAX_PAGES,
    delay=0.5,
    scope="website",
):
    batch.append(page)
    pages_processed += 1

    print(
        f"Processed page {pages_processed}: "
        f"{page['url']}"
    )

    if len(batch) >= BATCH_SIZE:
        store_batch()

store_batch()

print("\nIndexing completed.")
print("Pages processed:", pages_processed)
print("Chunks generated:", chunks_processed)
print("Total V2 chunks:", store.count())
