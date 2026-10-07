import numpy as np

from RAG.embeddings import EmbeddingModel


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        print("Creating embeddings for website content...")

        self.model = EmbeddingModel()

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        self.embeddings = self.model.embed_documents(texts)

        print(
            f"Created embeddings for {len(chunks)} chunks."
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        query_embedding = self.model.embed_query(query)

        scores = np.dot(
            self.embeddings,
            query_embedding,
        )

        best_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in best_indices:

            chunk = self.chunks[index].copy()

            chunk["score"] = float(scores[index])

            results.append(chunk)

        return results