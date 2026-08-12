import chromadb

from app.config import CHROMA_PATH, COLLECTION_NAME


class VectorStore:
    """
    ChromaDB vector store for storing document chunks
    and their embeddings.
    """

    def __init__(
        self,
        path=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    ):

        self.client = chromadb.PersistentClient(
            path=path
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine"
            }
        )

    def add_documents(
        self,
        documents,
        embeddings
    ):
        """
        Add document chunks and embeddings to ChromaDB.
        """

        ids = []

        texts = []

        metadatas = []

        for index, document in enumerate(documents):

            document_id = (
                f"{document['source']}_"
                f"{document.get('page', 'na')}_"
                f"{document['chunk_id']}_"
                f"{index}"
            )

            ids.append(document_id)

            texts.append(
                document["text"]
            )

            metadatas.append({
                "source": document["source"],
                "page": str(
                    document.get("page", "")
                ),
                "file_type": document["file_type"],
                "chunk_id": str(
                    document["chunk_id"]
                )
            })

        if not ids:
            return

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )

    def search(
        self,
        query_embedding,
        top_k=5
    ):
        """
        Search ChromaDB for the most similar chunks.
        """

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        return results

    def count(self):
        """
        Return number of stored chunks.
        """

        return self.collection.count()