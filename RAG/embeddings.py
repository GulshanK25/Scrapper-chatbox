from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed_documents(
        self,
        texts: list[str],
    ):
        """
        Convert multiple text chunks into embeddings.
        """

        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )

    def embed_query(
        self,
        query: str,
    ):
        """
        Convert a user's question into an embedding.
        """

        return self.model.encode(
            query,
            normalize_embeddings=True,
        )