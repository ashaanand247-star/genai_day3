import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


# -----------------------------
# Configuration
# -----------------------------

CONFIG_PATH = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).parent / "evaluation_config.json"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = json.load(file)

BASE_URL = CONFIG["base_url"]
ASK_ENDPOINT = f"{BASE_URL}/ask"

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

TOP_K = CONFIG["top_k"]
TIMEOUT_SECONDS = CONFIG["timeout_seconds"]


# -----------------------------
# Load golden dataset
# -----------------------------

def load_golden_set():
    cases = []

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                cases.append(json.loads(line))

    return cases


# -----------------------------
# Run one evaluation case
# -----------------------------

def run_case(case):
    payload = {
        "question": case["question"],
        "top_k": TOP_K,
        "filters": None
    }

    start_time = time.perf_counter()

    try:
        response = requests.post(
            ASK_ENDPOINT,
            json=payload,
            timeout=TIMEOUT_SECONDS
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        if response.ok:
            result = response.json()

            return {
                "case_id": case["case_id"],
                "question": case["question"],
                "category": case["category"],
                "expected_source_ids": case["expected_source_ids"],
                "expected_answerability": case["answerability"],

                "answer": result.get("answer"),
                "sources": result.get("sources", []),
                "retrieved_sources": result.get(
                    "retrieved_sources", []
                ),
                "chunk_previews": result.get(
                    "chunk_previews", []
                ),
                "retrieval_scores": result.get(
                    "retrieval_scores", []
                ),
                "status": result.get("status"),

                "request_id": result.get("request_id"),

                "latency_ms": round(
                    latency_ms,
                    2
                ),

                "error": None
            }

        return {
            "case_id": case["case_id"],
            "question": case["question"],
            "category": case["category"],
            "expected_source_ids": case["expected_source_ids"],
            "expected_answerability": case["answerability"],

            "answer": None,
            "sources": [],
            "retrieved_sources": [],
            "chunk_previews": [],
            "retrieval_scores": [],
            "status": "error",
            "request_id": None,

            "latency_ms": round(
                latency_ms,
                2
            ),

            "error": {
                "status_code": response.status_code,
                "message": response.text
            }
        }

    except Exception as exc:

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "case_id": case["case_id"],
            "question": case["question"],
            "category": case["category"],
            "expected_source_ids": case["expected_source_ids"],
            "expected_answerability": case["answerability"],

            "answer": None,
            "sources": [],
            "retrieved_sources": [],
            "chunk_previews": [],
            "retrieval_scores": [],
            "status": "error",
            "request_id": None,

            "latency_ms": round(
                latency_ms,
                2
            ),

            "error": {
                "type": type(exc).__name__,
                "message": str(exc)
            }
        }


# -----------------------------
# Main evaluation
# -----------------------------

def main():

    cases = load_golden_set()

    print(
        f"Loaded {len(cases)} golden evaluation cases."
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    for index, case in enumerate(cases, start=1):

        print(
            f"Running {index}/{len(cases)}: "
            f"{case['case_id']}"
        )

        result = run_case(case)

        results.append(result)

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    result_path = (
        RESULTS_DIR
        / f"eval_{timestamp}.json"
    )

    evaluation_output = {
        "evaluation_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

     "configuration": {
    **CONFIG,
    "golden_set": str(GOLDEN_SET_PATH),
    "config_file": str(CONFIG_PATH)
},

        "case_count": len(cases),

        "results": results
    }

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("Evaluation completed.")
    print(
        f"Results saved to: {result_path}"
    )


if __name__ == "__main__":
    main()