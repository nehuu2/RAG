from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


class EmbeddingModel:
    """
    Wrapper around the Sentence Transformer embedding model.
    """

    def __init__(self, model_name=None):

        if model_name is None:
            model_name = EMBEDDING_MODEL

        self.model_name = model_name

        print(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

    def encode(self, texts):

        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    def encode_query(self, query):

        return self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]