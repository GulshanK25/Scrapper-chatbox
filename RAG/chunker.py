def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[str]:
    """
    Split text into overlapping chunks.

    chunk_size and overlap are measured in characters
    for this simple first implementation.
    """

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Convert scraped website pages into chunks while
    preserving the source URL for every chunk.
    """

    all_chunks = []

    for page in pages:

        chunks = chunk_text(page["text"])

        for index, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "text": chunk,
                    "source": page["url"],
                    "chunk_id": index,
                }
            )

    return all_chunks