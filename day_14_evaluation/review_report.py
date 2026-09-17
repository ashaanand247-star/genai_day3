import json
from pathlib import Path
from collections import Counter


RETRIEVAL_FILE = Path(
    "day_14_evaluation/retrieval_results.json"
)

ANSWER_FILE = Path(
    "day_14_evaluation/answer_results.json"
)

BASELINE_FILE = Path(
    "day_13_evaluation/results/eval_20260911T074518Z.json"
)

OUTPUT_FILE = Path(
    "day_14_evaluation/review_report.json"
)


def get_failure_category(retrieval, answer):
    retrieval_grade = retrieval.get("grade")
    answer_grade = answer.get("grade")

    if retrieval_grade == "ERROR" or answer_grade == "ERROR":
        return "evaluation_error"

    retrieval_failed = retrieval_grade == "FAIL"
    answer_failed = answer_grade == "FAIL"

    if retrieval_failed and answer_failed:
        return "both"

    if retrieval_failed:
        return "retrieval_failure"

    if answer_failed:
        return "answer_failure"

    return "pass"


def get_failure_reason(retrieval, answer):
    """Classify answer failures using actual evaluation results."""

    if answer.get("grade") != "FAIL":
        return None

    case_id = answer.get("case_id")
    answerability_correct = answer.get(
        "answerability_correct"
    )
    expected_facts_score = answer.get(
        "expected_facts_score"
    )
    citation_present = answer.get(
        "citation_present"
    )

    # Cases where the system abstained when the
    # golden set expected an answer.
    if not answerability_correct and case_id in {
        "CASE016",
        "CASE017",
        "CASE019",
        "CASE021",
    }:
        return "over_abstention_ambiguous_or_multi_document"

    # Cases where the system did not fully cover
    # the expected facts.
    if expected_facts_score is not None and expected_facts_score < 1.0:
        if case_id in {"CASE023", "CASE024"}:
            return "over_abstention_adversarial_false_premise"

        return "partial_expected_fact_coverage"

    # CASE005: factual coverage exists, but the
    # answerability classification was incorrect.
    if (
        case_id == "CASE005"
        and answerability_correct is False
    ):
        return "answerability_classification_mismatch"

    if not citation_present:
        return "missing_citation"

    return "other_answer_failure"


def main():
    print("Loading Day 14 evaluation results...")

    with RETRIEVAL_FILE.open("r", encoding="utf-8") as file:
        retrieval_data = json.load(file)

    with ANSWER_FILE.open("r", encoding="utf-8") as file:
        answer_data = json.load(file)

    with BASELINE_FILE.open("r", encoding="utf-8") as file:
        baseline_data = json.load(file)

    retrieval_cases = {
        case["case_id"]: case
        for case in retrieval_data["cases"]
    }

    answer_cases = {
        case["case_id"]: case
        for case in answer_data["cases"]
    }

    baseline_cases = {
        case["case_id"]: case
        for case in baseline_data["results"]
    }

    report_cases = []

    for case_id in retrieval_cases:
        retrieval = retrieval_cases[case_id]
        answer = answer_cases[case_id]
        baseline = baseline_cases.get(case_id, {})

        report_cases.append(
            {
                "case_id": case_id,
                "category": retrieval["category"],
                "retrieval_grade": retrieval["grade"],
                "hit": retrieval["hit"],
                "recall_at_k": retrieval["recall_at_k"],
                "reciprocal_rank": retrieval["reciprocal_rank"],
                "answer_grade": answer["grade"],
                "answerability_correct": answer[
                    "answerability_correct"
                ],
                "expected_facts_score": answer[
                    "expected_facts_score"
                ],
                "citation_present": answer[
                    "citation_present"
                ],
                "citation_valid": answer[
                    "citation_valid"
                ],
                "latency_ms": baseline.get("latency_ms"),
                "failure_category": get_failure_category(
                    retrieval,
                    answer
                ),
                "failure_reason": get_failure_reason(
                    retrieval,
                    answer
                ),
            }
        )

    failure_reasons = Counter(
        case["failure_reason"]
        for case in report_cases
        if case["failure_reason"] is not None
    )

    top_failure_categories = [
        {
            "category": category,
            "count": count,
        }
        for category, count in failure_reasons.most_common(3)
    ]

    report = {
        "report_type": "Day 14 Review-Friendly Evaluation Report",
        "source_files": {
            "retrieval": str(RETRIEVAL_FILE),
            "answer": str(ANSWER_FILE),
            "latency": str(BASELINE_FILE),
        },
        "case_count": len(report_cases),
        "failure_analysis": {
            "top_failure_categories": top_failure_categories,
            "total_failed_cases": sum(
                failure_reasons.values()
            ),
        },
        "cases": report_cases,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print()
    print("Review report generated successfully.")
    print(f"Cases included: {len(report_cases)}")

    print()
    print("Top failure categories:")

    for item in top_failure_categories:
        print(
            f"  {item['category']}: "
            f"{item['count']}"
        )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()