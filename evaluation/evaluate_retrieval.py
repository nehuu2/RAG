import json
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore


TOP_K = 3


def evaluate_retrieval():

    print("\n# RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    # ---------------------------------------
    # Load evaluation questions
    # ---------------------------------------

    questions_file = ROOT_DIR / "evaluation" / "questions.json"

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\nFound {len(questions)} evaluation question(s).\n")

    # ---------------------------------------
    # Load embedding model
    # ---------------------------------------

    embedding_model = EmbeddingModel()

    # ---------------------------------------
    # Load ChromaDB
    # ---------------------------------------

    vector_store = VectorStore()

    print(f"ChromaDB contains {vector_store.count()} chunks.")
    print("\nRunning retrieval...\n")

    # ---------------------------------------
    # Metrics
    # ---------------------------------------

    total_questions = len(questions)
    successful_questions = 0

    similarity_scores = []
    reciprocal_ranks = []

    # ---------------------------------------
    # Evaluate questions
    # ---------------------------------------

    for i, item in enumerate(questions, start=1):

        question = item["question"]

        print(f"Question {i}: {question}")

        # Create query embedding
        query_embedding = embedding_model.encode_query(
            question
        )

        # Retrieve top K
        results = vector_store.search(
            query_embedding,
            top_k=TOP_K
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        print(f"Retrieved {len(documents)} chunks:")

        # ---------------------------------------
        # Hit Rate
        # ---------------------------------------

        if len(documents) > 0:
            successful_questions += 1

        # ---------------------------------------
        # Similarity
        # ---------------------------------------

        question_similarities = []

        for distance in distances:

            similarity = 1 - distance

            question_similarities.append(
                similarity
            )

            similarity_scores.append(
                similarity
            )

        # ---------------------------------------
        # MRR
        #
        # Without ground-truth labels,
        # we use the first retrieved result
        # as the retrieval rank.
        # ---------------------------------------

        if len(documents) > 0:
            reciprocal_ranks.append(1.0)
        else:
            reciprocal_ranks.append(0.0)

        # ---------------------------------------
        # Display results
        # ---------------------------------------

        for j, document in enumerate(
            documents,
            start=1
        ):

            preview = document[:200].replace(
                "\n",
                " "
            )

            source = "Unknown"

            if j - 1 < len(metadatas):
                source = metadatas[j - 1].get(
                    "source",
                    "Unknown"
                )

            distance = distances[j - 1]

            similarity = 1 - distance

            print(f"\n  Result {j}")
            print(f"  Source: {source}")
            print(f"  Distance: {distance:.4f}")
            print(f"  Similarity: {similarity:.4f}")
            print(f"  Text: {preview}...")

        if question_similarities:

            avg_similarity = (
                sum(question_similarities)
                / len(question_similarities)
            )

            print(
                f"\n  Average Similarity@{TOP_K}: "
                f"{avg_similarity:.4f}"
            )

        else:

            print(
                f"\n  Average Similarity@{TOP_K}: 0.0000"
            )

        print("\n" + "-" * 60)

    # ---------------------------------------
    # Final Metrics
    # ---------------------------------------

    hit_rate = (
        successful_questions
        / total_questions
        if total_questions > 0
        else 0
    )

    average_similarity = (
        sum(similarity_scores)
        / len(similarity_scores)
        if similarity_scores
        else 0
    )

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
        if reciprocal_ranks
        else 0
    )

    # ---------------------------------------
    # Evaluation Summary
    # ---------------------------------------

    print("\n")
    print("=" * 60)
    print("RAG RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"\nTotal Questions: "
        f"{total_questions}"
    )

    print(
        f"Questions with Retrieved Chunks: "
        f"{successful_questions}"
    )

    print(
        f"Hit Rate@{TOP_K}: "
        f"{hit_rate:.4f} "
        f"({hit_rate * 100:.2f}%)"
    )

    print(
        f"Average Similarity@{TOP_K}: "
        f"{average_similarity:.4f}"
    )

    print(
        f"MRR@{TOP_K}: "
        f"{mrr:.4f}"
    )

    print("\n" + "=" * 60)

    print(
        "\nNote:"
    )

    print(
        "These metrics are retrieval diagnostics. "
        "Formal Recall@K/MRR requires ground-truth "
        "relevant sources in questions.json."
    )


if __name__ == "__main__":
    evaluate_retrieval()