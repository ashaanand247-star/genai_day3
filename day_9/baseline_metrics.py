import json


# Load retrieval results

def load_results():

    with open(
        "day_6_Vector_search/retrieval_result_report.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# Find the rank of the expected document

def find_expected_rank(test):

    expected_document_id = test["expected_document_id"]

    results = test["results"]

    for rank, result in enumerate(
        results,
        start=1
    ):

        if result["document_id"] == expected_document_id:
            return rank

    return None


# Calculate Hit@3

def calculate_hit(expected_rank):

    if expected_rank is not None and expected_rank <= 3:
        return 1

    return 0


# Calculate Reciprocal Rank

def calculate_reciprocal_rank(expected_rank):

    if expected_rank is not None:
        return 1 / expected_rank

    return 0


# Calculate per-question metrics

def calculate_metrics(results):

    metrics = []

    for test in results:

        expected_rank = find_expected_rank(
            test
        )

        hit = calculate_hit(
            expected_rank
        )

        reciprocal_rank = calculate_reciprocal_rank(
            expected_rank
        )

        metrics.append({
            "test_number": test["test_number"],
            "question": test["question"],
            "expected_document_id": test["expected_document_id"],
            "expected_rank": expected_rank,
            "hit_at_3": hit,
            "reciprocal_rank": reciprocal_rank
        })

    return metrics


# Calculate overall metrics

def calculate_overall_metrics(metrics):

    total_questions = len(metrics)

    total_hits = sum(
        item["hit_at_3"]
        for item in metrics
    )

    hit_rate_at_3 = (
        total_hits / total_questions
    )

    total_reciprocal_rank = sum(
        item["reciprocal_rank"]
        for item in metrics
    )

    mrr = (
        total_reciprocal_rank / total_questions
    )

    return {
        "hit_rate_at_3": hit_rate_at_3,
        "mrr": mrr
    }


# Save baseline metric report

def save_report(metrics, overall_metrics):

    report = {
        "per_question": metrics,
        "overall": overall_metrics
    }

    with open(
        "day_9/baseline_metrics_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )


# Main flow

def main():

    results = load_results()

    metrics = calculate_metrics(
        results
    )

    overall_metrics = calculate_overall_metrics(
        metrics
    )

    save_report(
        metrics,
        overall_metrics
    )

    print("Per-question metrics:\n")

    for item in metrics:
        print(item)

    print("\nOverall metrics:")

    print(
        overall_metrics
    )

    print(
        "\nBaseline metric report saved to:"
    )

    print(
        "day_9/baseline_metrics_report.json"
    )


# Program entry point

if __name__ == "__main__":

    main()