
import csv
import json
from pathlib import Path

from RAG.Vectorstore import VectorStore
from RAG.reranker import Reranker


COLLECTION = "website_chunks_v3"
QUESTIONS_FILE = Path("evaluation_questions.json")
OUTPUT_FILE = Path("evaluation_results_v3.csv")

TOP_K = 5
CANDIDATE_K = 50
RETRIEVAL_K = 30


def is_relevant(result, question):
    source = result.get("source", "")
    text = result.get("text", "")

    expected_source = question["expected_source"]
    expected_phrases = question["expected_phrases"]

    source_matches = expected_source in source

    text_lower = " ".join(text.lower().split())

    phrase_matches = any(
        " ".join(phrase.lower().split()) in text_lower
        for phrase in expected_phrases
    )

    return source_matches and phrase_matches


def evaluate_question(store, reranker, question):
    query = question["question"]

    candidates = store.hybrid_search(
        query=query,
        top_k=RETRIEVAL_K,
        candidate_k=CANDIDATE_K,
    )

    results = reranker.rerank(
        query,
        candidates,
        top_k=TOP_K,
    )

    relevant_rank = None

    for rank, result in enumerate(results, start=1):
        if is_relevant(result, question):
            relevant_rank = rank
            break

    hit = 1 if relevant_rank is not None else 0

    reciprocal_rank = (
        1 / relevant_rank
        if relevant_rank is not None
        else 0.0
    )

    return {
        "id": question["id"],
        "question": query,
        "hit_at_5": hit,
        "reciprocal_rank": reciprocal_rank,
        "relevant_rank": relevant_rank,
        "top_source": (
            results[0].get("source", "")
            if results else ""
        ),
        "top_preview": (
            results[0].get("text", "")[:200]
            .replace("\n", " ")
            if results else ""
        ),
    }


def main():
    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    print("Loading vector store...")
    store = VectorStore(collection_name=COLLECTION)

    print("Collection:", COLLECTION)
    print("Total chunks:", store.count())

    reranker = Reranker()

    evaluation_results = []

    for index, question in enumerate(questions, start=1):
        print(
            f"\n[{index}/{len(questions)}] "
            f"{question['question']}"
        )

        result = evaluate_question(
            store,
            reranker,
            question,
        )

        evaluation_results.append(result)

        print("Hit@5:", result["hit_at_5"])
        print("Relevant rank:", result["relevant_rank"])
        print(
            "Reciprocal rank:",
            round(result["reciprocal_rank"], 3),
        )

    total = len(evaluation_results)

    hit_at_5 = (
        sum(r["hit_at_5"] for r in evaluation_results)
        / total
        if total else 0.0
    )

    mrr_at_5 = (
        sum(r["reciprocal_rank"] for r in evaluation_results)
        / total
        if total else 0.0
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "question",
                "hit_at_5",
                "reciprocal_rank",
                "relevant_rank",
                "top_source",
                "top_preview",
            ],
        )

        writer.writeheader()
        writer.writerows(evaluation_results)

    print("\n" + "=" * 45)
    print("EVALUATION SUMMARY")
    print("=" * 45)

    print("Collection:", COLLECTION)
    print("Questions:", total)
    print("Hit@5:", round(hit_at_5, 3))
    print("MRR@5:", round(mrr_at_5, 3))
    print("Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
