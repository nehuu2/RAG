from app.config import TOP_K, SIMILARITY_THRESHOLD
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore


class Retriever:
    """
    Retrieves the most relevant document chunks
    for a user query.
    """

    def __init__(
        self,
        top_k=None,
        similarity_threshold=None
    ):
        self.top_k = (
            top_k
            if top_k is not None
            else TOP_K
        )

        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else SIMILARITY_THRESHOLD
        )

        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    def retrieve(self, query):

        if not query or not query.strip():
            return []

        # -----------------------------------
        # Convert question into embedding
        # -----------------------------------

        query_embedding = (
            self.embedding_model.encode_query(query)
        )

        # -----------------------------------
        # Search ChromaDB
        # -----------------------------------

        results = self.vector_store.search(
            query_embedding,
            top_k=self.top_k
        )

        # DEBUG: Show complete ChromaDB result
        print("\n========== CHROMA RESULTS ==========")
        print(results)
        print("====================================\n")

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        print("Number of documents:", len(documents))
        print("Number of distances:", len(distances))
        print(
            "Similarity threshold:",
            self.similarity_threshold
        )

        retrieved_chunks = []

        # -----------------------------------
        # Process retrieved documents
        # -----------------------------------

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            # ChromaDB cosine distance
            # similarity = 1 - distance
            similarity = 1 - distance

            print("\n---------- CHUNK ----------")
            print("Distance:", distance)
            print("Similarity:", similarity)
            print("Source:", metadata.get("source"))
            print("Page:", metadata.get("page"))
            print("---------------------------")

            if similarity >= self.similarity_threshold:

                retrieved_chunks.append({
                    "text": document,
                    "metadata": metadata,
                    "similarity": similarity
                })

        print(
            "\nRetrieved chunks after threshold:",
            len(retrieved_chunks)
        )

        return retrieved_chunks