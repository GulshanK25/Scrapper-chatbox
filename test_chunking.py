
from RAG.chunker import chunk_text


sample_text = """
Python functions are defined using the def keyword.
A function can accept parameters and return values.

Decorators are functions that modify or extend
the behavior of other functions.

A decorator is commonly applied using the
@decorator syntax above a function definition.

Decorators are useful for logging, authentication,
caching, and other reusable functionality.
"""


chunks = chunk_text(
    sample_text,
    chunk_size=130,
    overlap=30,
)

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk)
    print("Length:", len(chunk))