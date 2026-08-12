from pathlib import Path
import sys

# Allow importing modules from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.loaders import load_document
from app.chunking import chunk_documents
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore


DATA_DIR = PROJECT_ROOT / "data"


def collect_files():
    """
    Find all supported documents inside:
    data/pdf
    data/html
    data/md
    """

    files = []

    for extension in ["*.pdf", "*.html", "*.htm", "*.md", "*.markdown"]:

        files.extend(
            DATA_DIR.rglob(extension)
        )

    return files


def main():

    print("=" * 60)
    print("RAG DOCUMENT INGESTION")
    print("=" * 60)

    files = collect_files()

    if not files:

        print("No documents found.")

        print(
            "Add PDF, HTML, or Markdown files "
            "inside the data/ folder."
        )

        return

    print(f"\nFound {len(files)} document(s).")

    all_documents = []

    # ---------------------------------
    # 1. Load documents
    # ---------------------------------

    for file_path in files:

        print(
            f"\nLoading: {file_path}"
        )

        try:

            documents = load_document(
                str(file_path)
            )

            print(
                f"Loaded {len(documents)} document section(s)."
            )

            all_documents.extend(
                documents
            )

        except Exception as error:

            print(
                f"Error loading {file_path}: {error}"
            )

    if not all_documents:

        print("\nNo text could be extracted.")

        return

    print(
        f"\nTotal loaded sections: "
        f"{len(all_documents)}"
    )

    # ---------------------------------
    # 2. Chunk documents
    # ---------------------------------

    print("\nCreating chunks...")

    chunks = chunk_documents(
        all_documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    if not chunks:

        print("No chunks were created.")

        return

    # ---------------------------------
    # 3. Create embeddings
    # ---------------------------------

    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Creating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = embedding_model.encode(
        texts
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # ---------------------------------
    # 4. Store in ChromaDB
    # ---------------------------------

    print("\nStoring vectors in ChromaDB...")

    vector_store = VectorStore()

    vector_store.add_documents(
        chunks,
        embeddings
    )

    print(
        f"ChromaDB now contains "
        f"{vector_store.count()} chunks."
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()