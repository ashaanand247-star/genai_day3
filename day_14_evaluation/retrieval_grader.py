import json
from pathlib import Path
import sys


# Day 13 valid baseline evaluation
INPUT_FILE = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "day_13_evaluation/results/eval_20260911T074518Z.json"
)

OUTPUT_FILE = Path(
    "day_14_evaluation/retrieval_results.json"
)


def calculate_recall(expected_sources, retrieved_sources):
    """Calculate Recall@K for the retrieved documents."""

    if not expected_sources:
        return None

    expected = set(expected_sources)

    retrieved = {
        source.split(":")[0]
        for source in retrieved_sources
    }

    found = expected.intersection(retrieved)

    return len(found) / len(expected)

def calculate_reciprocal_rank(expected_sources, retrieved_sources):
    """Calculate Reciprocal Rank of the first relevant document."""

    if not expected_sources:
        return None

    expected = set(expected_sources)

    for rank, source in enumerate(retrieved_sources, start=1):
        document_id = source.split(":")[0]

        if document_id in expected:
            return 1 / rank

    return 0.0


def grade_case(case):
    """Grade retrieval performance for one evaluation case."""

    expected_sources = case.get("expected_source_ids", [])
    retrieved_sources = case.get("retrieved_sources", [])
    status = case.get("status")
    error = case.get("error")

    # Provider/API failure â€” do not treat as retrieval failure.
    if error:
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "hit": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
            "grade": "ERROR",
        }

    # Cases without expected documents are not retrieval-hit cases.
    if not expected_sources:
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "hit": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
            "grade": "NOT_APPLICABLE",
        }

    expected_document_ids = set(expected_sources)

    retrieved_document_ids = {
        source.split(":")[0]
        for source in retrieved_sources
    }

    hit = bool(
        expected_document_ids.intersection(retrieved_document_ids)
    )

    recall = calculate_recall(
        expected_sources,
        retrieved_sources
    )

    reciprocal_rank = calculate_reciprocal_rank(
        expected_sources,
        retrieved_sources
    )

    grade = "PASS" if hit else "FAIL"

    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "expected_sources": expected_sources,
        "retrieved_sources": retrieved_sources,
        "hit": hit,
        "recall_at_k": recall,
        "reciprocal_rank": reciprocal_rank,
        "grade": grade,
    }


def main():
    print("Loading Day 13 baseline evaluation...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        evaluation = json.load(file)

    results = []

    for case in evaluation["results"]:
        results.append(grade_case(case))

    applicable_results = [
        result
        for result in results
        if result["grade"] in {"PASS", "FAIL"}
    ]

    error_count = sum(
        result["grade"] == "ERROR"
        for result in results
    )

    not_applicable_count = sum(
        result["grade"] == "NOT_APPLICABLE"
        for result in results
    )

    hit_rate = (
        sum(result["hit"] for result in applicable_results)
        / len(applicable_results)
        if applicable_results
        else None
    )

    recall_values = [
        result["recall_at_k"]
        for result in applicable_results
        if result["recall_at_k"] is not None
    ]

    mrr_values = [
        result["reciprocal_rank"]
        for result in applicable_results
        if result["reciprocal_rank"] is not None
    ]

    summary = {
        "total_cases": len(results),
        "applicable_cases": len(applicable_results),
        "error_cases": error_count,
        "not_applicable_cases": not_applicable_count,
        "hit_rate": hit_rate,
        "recall_at_k": (
            sum(recall_values) / len(recall_values)
            if recall_values
            else None
        ),
        "mrr": (
            sum(mrr_values) / len(mrr_values)
            if mrr_values
            else None
        ),
    }

    output = {
        "input_file": str(INPUT_FILE),
        "metrics": summary,
        "cases": results,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print()
    print("Retrieval grading completed.")
    print()
    print(f"Total cases:       {summary['total_cases']}")
    print(f"Applicable cases:  {summary['applicable_cases']}")
    print(f"Error cases:       {summary['error_cases']}")
    print(f"Not applicable:    {summary['not_applicable_cases']}")
    print(f"Hit Rate:          {summary['hit_rate']:.3f}")
    print(f"Recall@3:          {summary['recall_at_k']:.3f}")
    print(f"MRR:               {summary['mrr']:.3f}")
    print()
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()