import json
import sys
from pathlib import Path


SCORECARD_PATH = Path(
    sys.argv[1] if len(sys.argv) > 1
    else "day_14_evaluation/scorecard.json"
)

THRESHOLDS = {
    "hit_rate": 0.90,
    "recall_at_k": 0.85,
    "mrr": 0.90,
    "answerability_accuracy": 0.65,
    "expected_facts_score": 0.70,
    "citation_validity": 0.95,
    "answer_pass_rate": 0.60,
}


def load_scorecard():
    if not SCORECARD_PATH.exists():
        print(f"ERROR: Scorecard not found: {SCORECARD_PATH}")
        sys.exit(1)

    with SCORECARD_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def check_metric(name, actual, minimum):
    passed = actual >= minimum

    status = "PASS" if passed else "FAIL"

    print(
        f"{name:<25} "
        f"Actual: {actual:.3f}  "
        f"Minimum: {minimum:.3f}  "
        f"{status}"
    )

    return passed


def main():
    print("Loading Day 14 scorecard...")
    scorecard = load_scorecard()

    retrieval = scorecard.get("retrieval_metrics", {})
    answer = scorecard.get("answer_metrics", {})

    checks = []

    print("\nRegression checks:")
    print("-" * 70)

    checks.append(
        check_metric(
            "Hit Rate",
            retrieval.get("hit_rate", 0),
            THRESHOLDS["hit_rate"],
        )
    )

    checks.append(
        check_metric(
            "Recall@3",
            retrieval.get("recall_at_k", 0),
            THRESHOLDS["recall_at_k"],
        )
    )

    checks.append(
        check_metric(
            "MRR",
            retrieval.get("mrr", 0),
            THRESHOLDS["mrr"],
        )
    )

    checks.append(
        check_metric(
            "Answerability",
            answer.get("answerability_accuracy", 0),
            THRESHOLDS["answerability_accuracy"],
        )
    )

    checks.append(
        check_metric(
            "Expected Facts",
            answer.get("expected_facts_score", 0),
            THRESHOLDS["expected_facts_score"],
        )
    )

    checks.append(
        check_metric(
            "Citation Validity",
            answer.get("citation_validity", 0),
            THRESHOLDS["citation_validity"],
        )
    )

    checks.append(
        check_metric(
            "Answer Pass Rate",
            answer.get("answer_pass_rate", 0),
            THRESHOLDS["answer_pass_rate"],
        )
    )

    print("-" * 70)

    if all(checks):
        print("\nREGRESSION CHECK: PASS")
        print("All metrics meet the minimum acceptable thresholds.")
        return 0

    print("\nREGRESSION CHECK: FAIL")
    print("One or more metrics are below the minimum acceptable thresholds.")
    return 1


if __name__ == "__main__":
    sys.exit(main())