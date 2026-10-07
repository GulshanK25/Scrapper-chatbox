from scrapper.crawler import crawl_website
from RAG.chunker import chunk_pages
from RAG.Vectorstore import VectorStore


website = input(
    "Enter website URL to index: "
)

pages = crawl_website(
    website,
    max_pages=5,
)

chunks = chunk_pages(pages)

print("\n-------------------------")
print("INDEXING")
print("-------------------------")

print(
    f"Pages scraped: {len(pages)}"
)

print(
    f"Chunks created: {len(chunks)}"
)

vector_store = VectorStore()

vector_store.add_chunks(chunks)

print(
    f"Total chunks in database: "
    f"{vector_store.count()}"
)

print("\nWebsite indexing complete.")