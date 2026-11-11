
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        print("Loading cross-encoder reranker...")

        self.model = CrossEncoder(model_name)

        print("Reranker ready.")

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        if not chunks or top_k <= 0:
            return []

        pairs = [
            (query, chunk["text"])
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        scored_chunks = []

        for chunk, score in zip(chunks, scores):
            scored_chunks.append({
                **chunk,
                "rerank_score": float(score),
            })

        ranked = sorted(
            scored_chunks,
            key=lambda chunk: chunk["rerank_score"],
            reverse=True,
        )

        return ranked[:top_k]
