import json
from pathlib import Path
from collections import Counter


RETRIEVAL_FILE = Path(
    "day_14_evaluation/retrieval_results.json"
)

ANSWER_FILE = Path(
    "day_14_evaluation/answer_results.json"
)

REVIEW_FILE = Path(
    "day_14_evaluation/review_report.json"
)

OUTPUT_FILE = Path(
    "day_14_evaluation/scorecard.json"
)


def main():
    print("Loading evaluation results...")

    with RETRIEVAL_FILE.open("r", encoding="utf-8") as file:
        retrieval_data = json.load(file)

    with ANSWER_FILE.open("r", encoding="utf-8") as file:
        answer_data = json.load(file)

    with REVIEW_FILE.open("r", encoding="utf-8") as file:
        review_data = json.load(file)

    retrieval_metrics = retrieval_data["metrics"]
    answer_metrics = answer_data["metrics"]

    cases = review_data["cases"]

    # ---------------------------------------------------------
    # Failure categories
    # ---------------------------------------------------------

    failure_categories = Counter(
        case["failure_category"]
        for case in cases
        if case["failure_category"] != "pass"
    )

    top_failure_categories = [
        {
            "category": category,
            "count": count,
        }
        for category, count in failure_categories.most_common(3)
    ]

    # ---------------------------------------------------------
    # Latency summary
    # ---------------------------------------------------------

    latencies = [
        case["latency_ms"]
        for case in cases
        if case.get("latency_ms") is not None
    ]

    latency_summary = {}

    if latencies:
        latency_summary = {
            "average_ms": round(
                sum(latencies) / len(latencies),
                2,
            ),
            "minimum_ms": round(
                min(latencies),
                2,
            ),
            "maximum_ms": round(
                max(latencies),
                2,
            ),
            "measured_cases": len(latencies),
        }

    # ---------------------------------------------------------
    # Cost proxy
    # ---------------------------------------------------------

    cost_proxy = {
        "metric": "evaluation_case_count",
        "value": len(cases),
        "description": (
            "Simple proxy for evaluation workload; "
            "not an actual monetary API cost."
        ),
    }

    # ---------------------------------------------------------
    # Scorecard
    # ---------------------------------------------------------

    scorecard = {
        "evaluation": "Day 14 RAG Evaluation Scorecard",

        "case_summary": {
            "total_cases": answer_metrics["total_cases"],
            "passed_cases": answer_metrics["pass_cases"],
            "failed_cases": answer_metrics["fail_cases"],
            "error_cases": answer_metrics["error_cases"],
        },

        "retrieval_metrics": {
            "hit_rate": retrieval_metrics["hit_rate"],
            "recall_at_k": retrieval_metrics["recall_at_k"],
            "mrr": retrieval_metrics["mrr"],
        },

        "answer_metrics": {
            "answerability_accuracy": answer_metrics[
                "answerability_accuracy"
            ],
            "expected_facts_score": answer_metrics[
                "average_expected_facts_score"
            ],
            "citation_validity": answer_metrics[
                "citation_validity"
            ],
            "answer_pass_rate": answer_metrics[
                "answer_pass_rate"
            ],
        },

        "failure_analysis": {
            "top_failure_categories": top_failure_categories,
        },

        "latency": latency_summary,

        "cost_proxy": cost_proxy,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(scorecard, file, indent=2)

    print()
    print("Day 14 scorecard generated successfully.")
    print()

    print("Retrieval:")
    print(
        f"  Hit Rate:   "
        f"{retrieval_metrics['hit_rate']:.3f}"
    )
    print(
        f"  Recall@3:   "
        f"{retrieval_metrics['recall_at_k']:.3f}"
    )
    print(
        f"  MRR:        "
        f"{retrieval_metrics['mrr']:.3f}"
    )

    print()
    print("Answer:")
    print(
        f"  Answerability: "
        f"{answer_metrics['answerability_accuracy']:.3f}"
    )
    print(
        f"  Facts Score:   "
        f"{answer_metrics['average_expected_facts_score']:.3f}"
    )
    print(
        f"  Citations:     "
        f"{answer_metrics['citation_validity']:.3f}"
    )
    print(
        f"  Pass Rate:     "
        f"{answer_metrics['answer_pass_rate']:.3f}"
    )

    print()
    print("Top Failure Categories:")

    if top_failure_categories:
        for item in top_failure_categories:
            print(
                f"  {item['category']}: "
                f"{item['count']}"
            )
    else:
        print("  None")

    print()
    print("Latency:")
    if latency_summary:
        print(
            f"  Average: "
            f"{latency_summary['average_ms']:.2f} ms"
        )
        print(
            f"  Minimum: "
            f"{latency_summary['minimum_ms']:.2f} ms"
        )
        print(
            f"  Maximum: "
            f"{latency_summary['maximum_ms']:.2f} ms"
        )

    print()
    print("Cost Proxy:")
    print(
        f"  Evaluation cases: "
        f"{cost_proxy['value']}"
    )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()