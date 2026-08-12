from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text, chunk_size=None, chunk_overlap=None):
    """
    Split text into overlapping chunks.

    Example:
    chunk_size = 1000
    chunk_overlap = 200

    The next chunk starts 200 characters
    before the previous chunk ends.
    """

    if chunk_size is None:
        chunk_size = CHUNK_SIZE

    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def chunk_documents(documents, chunk_size=None, chunk_overlap=None):
    """
    Chunk loaded documents while preserving metadata.
    """

    chunked_documents = []

    for document in documents:

        chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for index, chunk in enumerate(chunks):

            chunked_documents.append({
                "text": chunk,
                "source": document["source"],
                "page": document.get("page"),
                "file_type": document["file_type"],
                "chunk_id": index
            })

    return chunked_documents