import sys

sys.path.append(".")

from day_7_rag.retrieve import (
    create_chroma_client,
    get_collection,
    retrieve_chunks
)

CHROMA_DIR = "day_7_rag/chroma_db"
COLLECTION_NAME = "employee_documents"

# Same 10-question retrieval set used for Day 9
QUESTIONS = [
    ("Q1", "What is the company's policy on employee expenses?", "DOC001"),
    ("Q2", "What are the rules for working from home?", "DOC002"),
    ("Q3", "What should employees do if they become unavailable while working remotely?", "DOC002"),
    ("Q4", "What should an employee do if remote work is unavailable?", "DOC002"),
    ("Q5", "What should an employee do when they have an unexpected absence?", "DOC003"),
    ("Q6", "What is the company's leave policy?", "DOC003"),
    ("Q7", "What are the company's working hours?", "DOC003"),
    ("Q8", "What training is required during employee onboarding?", "DOC004"),
    ("Q9", "What is the company's conflict of interest policy?", "DOC005"),
    ("Q10", "What should employees do about conflicts of interest?", "DOC005"),
]


def evaluate_retrieval(collection, top_k):
    results = []

    for question_id, question, expected_document in QUESTIONS:

        retrieval = retrieve_chunks(
            collection,
            question,
            top_k=top_k
        )

        documents = retrieval["documents"][0]
        metadatas = retrieval["metadatas"][0]
        distances = retrieval["distances"][0]

        expected_rank = None

        for rank, metadata in enumerate(metadatas, start=1):
            if metadata["document_id"] == expected_document:
                expected_rank = rank
                break

        hit = expected_rank is not None

        reciprocal_rank = (
            1 / expected_rank
            if expected_rank is not None
            else 0
        )

        results.append({
            "question_id": question_id,
            "question": question,
            "expected_document": expected_document,
            "expected_rank": expected_rank,
            "hit": hit,
            "reciprocal_rank": reciprocal_rank,
            "top_distance": distances[0] if distances else None,
            "retrieved_documents": [
                metadata["document_id"]
                for metadata in metadatas
            ]
        })

    return results


def calculate_metrics(results):
    total = len(results)

    hit_rate = sum(
        result["hit"]
        for result in results
    ) / total

    mrr = sum(
        result["reciprocal_rank"]
        for result in results
    ) / total

    return hit_rate, mrr


def display_results(label, results):
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    for result in results:
        print(
            f"\n{result['question_id']} | "
            f"Expected: {result['expected_document']} | "
            f"Rank: {result['expected_rank']} | "
            f"Hit: {result['hit']} | "
            f"Top distance: {result['top_distance']}"
        )

        print(
            f"Retrieved: "
            f"{result['retrieved_documents']}"
        )


def main():

    chroma_client = create_chroma_client(
        CHROMA_DIR
    )

    collection = get_collection(
        chroma_client,
        COLLECTION_NAME
    )

    print("DAY 10 - FULL TOP-K VALIDATION")

    print("\nBaseline configuration:")
    print("top_k = 3")

    baseline_results = evaluate_retrieval(
        collection,
        top_k=3
    )

    print("\nExperiment configuration:")
    print("top_k = 2")

    experiment_results = evaluate_retrieval(
        collection,
        top_k=2
    )

    display_results(
        "BASELINE RESULTS",
        baseline_results
    )

    display_results(
        "TOP-K = 2 RESULTS",
        experiment_results
    )

    baseline_hit_rate, baseline_mrr = calculate_metrics(
        baseline_results
    )

    experiment_hit_rate, experiment_mrr = calculate_metrics(
        experiment_results
    )

    print("\n" + "=" * 80)
    print("AGGREGATE METRICS")
    print("=" * 80)

    print(
        f"\nBaseline Hit Rate: "
        f"{baseline_hit_rate:.3f}"
    )

    print(
        f"Experiment Hit Rate: "
        f"{experiment_hit_rate:.3f}"
    )

    print(
        f"\nBaseline MRR: "
        f"{baseline_mrr:.3f}"
    )

    print(
        f"Experiment MRR: "
        f"{experiment_mrr:.3f}"
    )

    print("\n" + "=" * 80)
    print("REGRESSION CHECK")
    print("=" * 80)

    regressions = []

    for baseline, experiment in zip(
        baseline_results,
        experiment_results
    ):
        if (
            baseline["hit"]
            and not experiment["hit"]
        ):
            regressions.append(
                baseline["question_id"]
            )

    if regressions:
        print(
            "\nREGRESSIONS FOUND:"
        )
        print(regressions)
    else:
        print(
            "\nNo retrieval regressions found."
        )


if __name__ == "__main__":
    main()