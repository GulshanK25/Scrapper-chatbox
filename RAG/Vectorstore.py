import hashlib
from urllib.parse import urlparse

import chromadb

from RAG.embeddings import EmbeddingModel


class VectorStore:
    def __init__(
        self,
        path: str = "./chroma_db",
        collection_name: str = "website_chunks",
    ):
        """
        Create or connect to a persistent ChromaDB collection.
        """

        self.client = chromadb.PersistentClient(
            path=path
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        self.embedding_model = EmbeddingModel()

    def add_chunks(
        self,
        chunks: list[dict],
    ):
        """
        Create embeddings and store website chunks
        permanently in ChromaDB.
        """

        if not chunks:
            print("No chunks to store.")
            return

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print("Creating embeddings...")

        embeddings = self.embedding_model.embed_documents(
            texts
        )

        ids = []
        metadatas = []

        for chunk in chunks:

            source = chunk["source"]

            domain = urlparse(
                source
            ).netloc

            # Create a unique ID using the URL
            # and the chunk number.
            unique_string = (
                f"{source}_{chunk['chunk_id']}"
            )

            unique_id = hashlib.sha256(
                unique_string.encode()
            ).hexdigest()

            ids.append(unique_id)

            metadatas.append(
                {
                    "source": source,
                    "domain": domain,
                    "chunk_id": chunk["chunk_id"],
                }
            )

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        print(
            f"Stored {len(chunks)} chunks in ChromaDB."
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        domain: str | None = None,
    ) -> list[dict]:
        """
        Search ChromaDB for chunks that are
        semantically similar to the user's question.

        If a domain is provided, search only
        chunks belonging to that website.
        """

        query_embedding = (
            self.embedding_model.embed_query(query)
        )

        query_arguments = {
            "query_embeddings": [
                query_embedding.tolist()
            ],
            "n_results": top_k,
        }

        # Restrict the search to one website.
        if domain:
            query_arguments["where"] = {
                "domain": domain
            }

        results = self.collection.query(
            **query_arguments
        )

        retrieved_chunks = []

        # No matching documents.
        if not results["documents"]:
            return retrieved_chunks

        if not results["documents"][0]:
            return retrieved_chunks

        for index in range(
            len(results["documents"][0])
        ):

            metadata = (
                results["metadatas"][0][index]
            )

            retrieved_chunks.append(
                {
                    "text": (
                        results["documents"][0][index]
                    ),
                    "source": metadata["source"],
                    "domain": metadata["domain"],
                    "chunk_id": metadata["chunk_id"],
                    "distance": (
                        results["distances"][0][index]
                    ),
                }
            )

        return retrieved_chunks

    def get_domains(self) -> list[str]:
        """
        Return all websites currently stored
        in the vector database.
        """

        results = self.collection.get(
            include=["metadatas"]
        )

        domains = set()

        for metadata in results["metadatas"]:

            if metadata and "domain" in metadata:
                domains.add(
                    metadata["domain"]
                )

        return sorted(domains)

    def count(self) -> int:
        """
        Return the number of stored chunks.
        """

        return self.collection.count()