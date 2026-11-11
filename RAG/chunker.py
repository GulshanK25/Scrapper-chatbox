import re

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

HEADING_PATTERN = re.compile(
    r"^#{1,6}\s+.+$"
)

FENCE_PATTERN = re.compile(
    r"^```"
)


def split_into_sections(text: str) -> list[str]:
    """
    Split structured text by headings.

    Headings inside code blocks are ignored.
    """
    sections = []
    current = []
    inside_code = False

    for line in text.splitlines(keepends=True):
        stripped = line.strip()

        if FENCE_PATTERN.match(stripped):
            inside_code = not inside_code

        is_heading = (
            not inside_code
            and bool(HEADING_PATTERN.fullmatch(stripped))
        )

        if is_heading and current:
            section = "".join(current).strip()

            if section:
                sections.append(section)

            current = []

        current.append(line)

    if current:
        section = "".join(current).strip()

        if section:
            sections.append(section)

    return sections


def split_into_blocks(section: str) -> list[str]:
    """
    Split a section into paragraph and code blocks.

    Fenced code blocks remain intact.
    """
    blocks = []
    current = []
    inside_code = False

    for line in section.splitlines(keepends=True):
        stripped = line.strip()

        if FENCE_PATTERN.match(stripped):
            if not inside_code and current:
                block = "".join(current).strip()

                if block:
                    blocks.append(block)

                current = []

            current.append(line)
            inside_code = not inside_code

            if not inside_code:
                block = "".join(current).strip()

                if block:
                    blocks.append(block)

                current = []

            continue

        if not inside_code and not stripped:
            if current:
                block = "".join(current).strip()

                if block:
                    blocks.append(block)

                current = []
        else:
            current.append(line)

    if current:
        block = "".join(current).strip()

        if block:
            blocks.append(block)

    return blocks


def split_text_block(
    text: str,
    max_size: int,
    overlap: int,
) -> list[str]:
    """
    Split oversized normal text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=min(
            overlap,
            max(0, max_size - 1),
        ),
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        keep_separator="end",
        strip_whitespace=True,
    )

    return splitter.split_text(text)


def split_code_block(
    block: str,
    max_size: int,
) -> list[str]:
    """
    Split oversized code blocks by lines.

    Each output retains valid Markdown fences.
    """
    lines = block.splitlines()

    if len(lines) < 2:
        return [block]

    opening = lines[0]
    closing = "```"

    code_lines = lines[1:-1]

    pieces = []
    current = []

    def build_piece(lines_to_include):
        return (
            opening
            + "\n"
            + "\n".join(lines_to_include)
            + "\n"
            + closing
        )

    for line in code_lines:
        candidate = current + [line]

        if (
            current
            and len(build_piece(candidate)) > max_size
        ):
            pieces.append(build_piece(current))
            current = [line]
        else:
            current = candidate

    if current:
        pieces.append(build_piece(current))

    return pieces


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Structure-aware chunking with code protection.

    - Preserves headings
    - Preserves short code blocks
    - Splits oversized code blocks by lines
    - Keeps glossary definitions together when possible
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    sections = split_into_sections(text)
    all_chunks = []

    for section in sections:
        lines = section.splitlines()

        heading = ""

        if lines and HEADING_PATTERN.fullmatch(
            lines[0].strip()
        ):
            heading = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            body = section

        if not body:
            all_chunks.append(section)
            continue

        prefix = heading + "\n\n" if heading else ""

        available = chunk_size - len(prefix)

        if available <= 20:
            all_chunks.extend(
                split_text_block(
                    section,
                    chunk_size,
                    overlap,
                )
            )
            continue

        blocks = split_into_blocks(body)

        current = ""

        def flush():
            nonlocal current

            if current.strip():
                all_chunks.append(
                    prefix + current.strip()
                )
                current = ""

        for block in blocks:
            if len(block) > available:
                flush()

                if block.startswith("```"):
                    parts = split_code_block(
                        block,
                        available,
                    )
                else:
                    parts = split_text_block(
                        block,
                        available,
                        overlap,
                    )

                for part in parts:
                    all_chunks.append(
                        prefix + part
                    )

                continue

            candidate = (
                current + "\n\n" + block
                if current
                else block
            )

            if len(candidate) <= available:
                current = candidate
            else:
                flush()
                current = block

        flush()

    return [
        chunk
        for chunk in all_chunks
        if chunk.strip()
    ]


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Convert crawled pages to vector-store chunks.
    """
    all_chunks = []

    for page in pages:
        text = page.get("text", "")
        source = page.get("url", "")

        if not text or not source:
            continue

        for index, chunk in enumerate(
            chunk_text(text)
        ):
            all_chunks.append({
                "text": chunk,
                "source": source,
                "chunk_id": index,
            })

    return all_chunks
