import json
import re
from pathlib import Path


INPUT_FILE = Path(
    "day_13_evaluation/results/eval_20260911T074518Z.json"
)

GOLDEN_SET_FILE = Path(
    "day_13_evaluation/golden_set.jsonl"
)

OUTPUT_FILE = Path(
    "day_14_evaluation/answer_results.json"
)


def extract_document_ids(text):
    """Extract document IDs such as DOC001 from an answer or citation."""
    if not text:
        return set()

    return set(re.findall(r"\bDOC\d+\b", text))


def check_answerability(case):
    """Check whether the system answered or correctly abstained."""

    expected = case["expected_answerability"]
    status = case["status"]

    if expected == "answerable":
        return status == "answered"

    if expected == "unanswerable":
        return status == "insufficient_evidence"

    return None


def check_expected_facts(case, golden_case):
    """Check expected facts when the golden set defines them."""

    expected_facts = golden_case.get("expected_facts", [])
    answer = case.get("answer") or ""

    if not expected_facts:
        return None

    answer_lower = answer.lower()

    matched = 0

    for fact in expected_facts:
        # Normalize whitespace and punctuation
        normalized_fact = re.sub(r"\s+", " ", fact.lower()).strip()
        normalized_answer = re.sub(r"\s+", " ", answer_lower).strip()
    
    # Check whether the key content of the fact is present.
        fact_words = [
            word
            for word in re.findall(r"\b[a-z]{4,}\b", normalized_fact)
            if word not in {
                "should",
                "would",
                "could",
                "their",
                "there",
                "about",
                "according",
                "applicable",
            }
        ]

        if fact_words:
            matched_words = sum(
                word in normalized_answer
                for word in fact_words
            )

            if matched_words / len(fact_words) >= 0.70:
                matched += 1

    return matched / len(expected_facts)


def check_citations(case):
    """Check whether citations in the answer point to retrieved documents."""

    answer = case.get("answer") or ""
    retrieved_sources = case.get("retrieved_sources", [])

    if case["status"] == "insufficient_evidence":
        return {
            "citation_present": False,
            "citation_valid": True,
        }

    citation_ids = extract_document_ids(answer)
    retrieved_ids = {
        source.split(":")[0]
        for source in retrieved_sources
    }

    citation_present = bool(citation_ids)

    if not citation_present:
        return {
            "citation_present": False,
            "citation_valid": False,
        }

    valid_ids = citation_ids.intersection(retrieved_ids)

    return {
        "citation_present": True,
        "citation_valid": bool(valid_ids),
    }


def grade_case(case, golden_case):


    """Grade one answer evaluation case."""

    answerability_correct = check_answerability(case)

    fact_score = check_expected_facts(case, golden_case)

    citation_result = check_citations(case)

    if case["status"] == "error":
        grade = "ERROR"

    elif answerability_correct is False:
        grade = "FAIL"

    elif (
        citation_result["citation_valid"] is False
        and case["status"] == "answered"
    ):
        grade = "FAIL"

    elif fact_score is not None and fact_score < 1.0:
        grade = "FAIL"

    else:
        grade = "PASS"

    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "expected_answerability": case["expected_answerability"],
        "actual_status": case["status"],
        "answerability_correct": answerability_correct,
        "expected_facts_score": fact_score,
        "citation_present": citation_result["citation_present"],
        "citation_valid": citation_result["citation_valid"],
        "grade": grade,
    }


def main():
    print("Loading Day 13 baseline evaluation...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        evaluation = json.load(file)

    golden_cases = {}

    with GOLDEN_SET_FILE.open("r", encoding="utf-8") as file:
       for line in file:
        golden_case = json.loads(line)
        golden_cases[golden_case["case_id"]] = golden_case
    

    results = []

    for case in evaluation["results"]:
        golden_case = golden_cases[case["case_id"]]
        results.append(grade_case(case, golden_case))

    scored_results = [
        result
        for result in results
        if result["grade"] in {"PASS", "FAIL"}
    ]

    pass_count = sum(
        result["grade"] == "PASS"
        for result in results
    )

    fail_count = sum(
        result["grade"] == "FAIL"
        for result in results
    )

    error_count = sum(
        result["grade"] == "ERROR"
        for result in results
    )

    answerability_correct_count = sum(
        result["answerability_correct"] is True
        for result in results
    )

    citation_results = [
        result
        for result in results
        if result["actual_status"] == "answered"
    ]

    citation_valid_count = sum(
        result["citation_valid"]
        for result in citation_results
    )

    fact_scores = [
        result["expected_facts_score"]
        for result in results
        if result["expected_facts_score"] is not None
    ]

    summary = {
        "total_cases": len(results),
        "scored_cases": len(scored_results),
        "pass_cases": pass_count,
        "fail_cases": fail_count,
        "error_cases": error_count,
        "answerability_accuracy": (
            answerability_correct_count / len(results)
            if results
            else None
        ),
        "citation_validity": (
            citation_valid_count / len(citation_results)
            if citation_results
            else None
        ),
        "average_expected_facts_score": (
            sum(fact_scores) / len(fact_scores)
            if fact_scores
            else None
        ),
        "answer_pass_rate": (
            pass_count / len(scored_results)
            if scored_results
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
    print("Answer grading completed.")
    print()
    print(f"Total cases:              {summary['total_cases']}")
    print(f"Scored cases:             {summary['scored_cases']}")
    print(f"PASS cases:               {summary['pass_cases']}")
    print(f"FAIL cases:               {summary['fail_cases']}")
    print(f"ERROR cases:              {summary['error_cases']}")
    print(
        f"Answerability accuracy:   "
        f"{summary['answerability_accuracy']:.3f}"
    )
    print(
        f"Citation validity:        "
        f"{summary['citation_validity']:.3f}"
    )
    facts_score = summary["average_expected_facts_score"]

    print(
    "Expected facts score:     "
    + (f"{facts_score:.3f}" if facts_score is not None else "N/A")
)
    print(
        f"Answer pass rate:         "
        f"{summary['answer_pass_rate']:.3f}"
    )
    print()
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()