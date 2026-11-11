
import hashlib
import re
from urllib.parse import urlparse

import chromadb
from rank_bm25 import BM25Okapi

from RAG.embeddings import EmbeddingModel


class VectorStore:

    def __init__(
        self,
        path: str = "./chroma_db",
        collection_name: str = "website_chunks",
    ):
        self.client = chromadb.PersistentClient(
            path=path
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )

        self.embedding_model = EmbeddingModel()

        # In-memory keyword index.
        self._keyword_cache = {}

    # --------------------------------
    # ADD DOCUMENT CHUNKS
    # --------------------------------

    def add_chunks(
        self,
        chunks: list[dict],
    ):
        if not chunks:
            print("No chunks to store.")
            return

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print("Creating embeddings...")

        embeddings = (
            self.embedding_model.embed_documents(
                texts
            )
        )

        ids = []
        metadatas = []

        for chunk in chunks:

            source = chunk["source"]

            domain = urlparse(
                source
            ).netloc.lower()

            unique_string = (
                f"{source}_{chunk['chunk_id']}"
            )

            unique_id = hashlib.sha256(
                unique_string.encode()
            ).hexdigest()

            ids.append(unique_id)

            metadatas.append({
                "source": source,
                "domain": domain,
                "chunk_id": chunk["chunk_id"],
            })

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        # Stored documents may have changed.
        self._keyword_cache.clear()

        print(
            f"Stored {len(chunks)} chunks in ChromaDB."
        )

    # --------------------------------
    # TOKENIZATION
    # --------------------------------

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Preserve technical identifiers such as
        classmethod, __init__, and asyncio.
        """
        return re.findall(
            r"[a-zA-Z_][a-zA-Z_0-9]*|\d+",
            text.lower(),
        )

    # --------------------------------
    # SEMANTIC SEARCH
    # --------------------------------

    def search(
        self,
        query: str,
        top_k: int = 3,
        domain: str | None = None,
    ) -> list[dict]:

        if not query.strip() or top_k <= 0:
            return []

        if self.collection.count() == 0:
            return []

        query_embedding = (
            self.embedding_model.embed_query(
                query
            )
        )

        arguments = {
            "query_embeddings": [
                query_embedding.tolist()
            ],
            "n_results": top_k,
        }

        if domain:
            arguments["where"] = {
                "domain": domain
            }

        results = self.collection.query(
            **arguments
        )

        retrieved = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        ids = results["ids"][0]
        distances = results["distances"][0]

        for index, document in enumerate(documents):

            metadata = metadatas[index]

            retrieved.append({
                "id": ids[index],
                "text": document,
                "source": metadata["source"],
                "domain": metadata["domain"],
                "chunk_id": metadata["chunk_id"],
                "distance": distances[index],
            })

        return retrieved

    # --------------------------------
    # BUILD KEYWORD INDEX
    # --------------------------------

    def _get_keyword_index(
        self,
        domain: str | None = None,
    ):
        """
        Build BM25 from documents already
        stored in ChromaDB.

        Cache the result for repeated queries.
        """

        cache_key = domain or "__all__"

        if cache_key in self._keyword_cache:
            return self._keyword_cache[cache_key]

        documents = []
        metadatas = []
        ids = []

        offset = 0
        batch_size = 500

        while True:

            arguments = {
                "limit": batch_size,
                "offset": offset,
                "include": [
                    "documents",
                    "metadatas",
                ],
            }

            if domain:
                arguments["where"] = {
                    "domain": domain
                }

            results = self.collection.get(
                **arguments
            )

            batch_ids = results["ids"]

            if not batch_ids:
                break

            ids.extend(batch_ids)
            documents.extend(
                results["documents"]
            )
            metadatas.extend(
                results["metadatas"]
            )

            offset += len(batch_ids)

        tokenized_documents = [
            self.tokenize(document or "")
            for document in documents
        ]

        bm25 = (
            BM25Okapi(tokenized_documents)
            if tokenized_documents
            else None
        )

        index = {
            "bm25": bm25,
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }

        self._keyword_cache[cache_key] = index

        return index

    # --------------------------------
    # KEYWORD SEARCH
    # --------------------------------

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        domain: str | None = None,
    ) -> list[dict]:

        if not query.strip() or top_k <= 0:
            return []

        index = self._get_keyword_index(domain)

        bm25 = index["bm25"]

        if bm25 is None:
            return []

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        retrieved = []

        for i in ranked_indices[:top_k]:

            score = float(scores[i])

            # Ignore documents without a match.
            if score <= 0:
                continue

            metadata = index["metadatas"][i]

            retrieved.append({
                "id": index["ids"][i],
                "text": index["documents"][i],
                "source": metadata["source"],
                "domain": metadata["domain"],
                "chunk_id": metadata["chunk_id"],
                "bm25_score": score,
            })

        return retrieved

    # --------------------------------
    # HYBRID SEARCH
    # --------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = 8,
        domain: str | None = None,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Combine semantic and BM25 rankings
        using Reciprocal Rank Fusion.
        """

        if not query.strip() or top_k <= 0:
            return []

        candidate_k = max(
            candidate_k,
            top_k,
        )

        semantic_results = self.search(
            query=query,
            top_k=candidate_k,
            domain=domain,
        )

        keyword_results = self.keyword_search(
            query=query,
            top_k=candidate_k,
            domain=domain,
        )

        fused = {}

        for method, results in (
            ("semantic", semantic_results),
            ("keyword", keyword_results),
        ):

            for rank, chunk in enumerate(
                results,
                start=1,
            ):

                chunk_id = chunk["id"]

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        **chunk,
                        "rrf_score": 0.0,
                        "retrieval_methods": [],
                    }

                fused[chunk_id]["rrf_score"] += (
                    1.0 / (rrf_k + rank)
                )

                fused[chunk_id][
                    "retrieval_methods"
                ].append(method)

                # Preserve scores from both searches.
                if "distance" in chunk:
                    fused[chunk_id]["distance"] = (
                        chunk["distance"]
                    )

                if "bm25_score" in chunk:
                    fused[chunk_id]["bm25_score"] = (
                        chunk["bm25_score"]
                    )

        ranked = sorted(
            fused.values(),
            key=lambda chunk: chunk["rrf_score"],
            reverse=True,
        )

        return ranked[:top_k]

    # --------------------------------
    # GET INDEXED DOMAINS
    # --------------------------------

    def get_domains(self) -> list[str]:

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

    # --------------------------------
    # COUNT STORED CHUNKS
    # --------------------------------

    def count(self) -> int:
        return self.collection.count()
