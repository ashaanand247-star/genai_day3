import json
import statistics
import time

import chromadb


# ============================================================
# Configuration
# ============================================================

BASELINE_CHROMA_DIR = "day_7_rag/chroma_db"
EXPERIMENT_CHROMA_DIR = "day_10_retrieval/chroma_db_chunk70"

COLLECTION_NAME = "employee_documents"

QUERY_EMBEDDINGS_FILE = (
    "day_10_retrieval/query_embeddings.json"
)

RUNS_PER_QUESTION = 5

QUESTIONS = [
    ("Q1", "What is the company's policy on employee expenses?"),
    ("Q2", "What are the rules for working from home?"),
    ("Q3", "What should employees do if they become unavailable while working remotely?"),
    ("Q4", "What should an employee do if remote work is unavailable?"),
    ("Q5", "What should an employee do when they have an unexpected absence?"),
    ("Q6", "What is the company's leave policy?"),
    ("Q7", "What are the company's working hours?"),
    ("Q8", "What training is required during employee onboarding?"),
    ("Q9", "What is the company's conflict of interest policy?"),
    ("Q10", "What should employees do about conflicts of interest?"),
]

TOP_K = 3


# ============================================================
# Load cached query embeddings
# ============================================================

def load_query_embeddings():

    with open(
        QUERY_EMBEDDINGS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    embeddings = data["embeddings"]

    questions = data["questions"]

    print(
        "Loaded cached query embeddings:",
        len(embeddings)
    )

    if len(questions) != len(QUESTIONS):

        raise ValueError(
            "Cached query embedding question count "
            "does not match the Day 10 test set."
        )

    return embeddings


# ============================================================
# Create ChromaDB client
# ============================================================

def get_collection(chroma_dir):

    client = chromadb.PersistentClient(
        path=chroma_dir
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# ============================================================
# Measure retrieval latency
# ============================================================

def measure_latency(
    collection,
    query_embedding
):

    start_time = time.perf_counter()

    collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    end_time = time.perf_counter()

    return end_time - start_time


# ============================================================
# Measure all questions
# ============================================================

def evaluate_collection(
    collection,
    query_embeddings,
    label
):

    print(
        "\n"
        + "=" * 70
    )

    print(label)

    print(
        "=" * 70
    )

    all_latencies = []

    question_results = []

    for index, (
        question_id,
        question
    ) in enumerate(QUESTIONS):

        query_embedding = query_embeddings[index]

        # Warm-up query
        collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K
        )

        timings = []

        for _ in range(RUNS_PER_QUESTION):

            latency = measure_latency(
                collection,
                query_embedding
            )

            timings.append(latency)

        average_latency = statistics.mean(
            timings
        )

        all_latencies.extend(
            timings
        )

        question_results.append(
            {
                "question_id": question_id,
                "average_latency_seconds": average_latency,
                "runs": RUNS_PER_QUESTION
            }
        )

        print(
            f"{question_id}: "
            f"{average_latency:.6f} seconds"
        )

    overall_average = statistics.mean(
        all_latencies
    )

    return overall_average, question_results


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DAY 10 - CHUNK SIZE LATENCY COMPARISON"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load cached embeddings
    # --------------------------------------------------------

    query_embeddings = load_query_embeddings()

    # --------------------------------------------------------
    # Load baseline index
    # --------------------------------------------------------

    print(
        "\nLoading baseline ChromaDB..."
    )

    baseline_collection = get_collection(
        BASELINE_CHROMA_DIR
    )

    print(
        "Baseline vectors:",
        baseline_collection.count()
    )

    # --------------------------------------------------------
    # Load selected Day 10 index
    # --------------------------------------------------------

    print(
        "\nLoading chunk_size=70 ChromaDB..."
    )

    experiment_collection = get_collection(
        EXPERIMENT_CHROMA_DIR
    )

    print(
        "Experiment vectors:",
        experiment_collection.count()
    )

    # --------------------------------------------------------
    # Measure baseline
    # --------------------------------------------------------

    baseline_average, baseline_results = (
        evaluate_collection(
            baseline_collection,
            query_embeddings,
            "BASELINE - chunk_size=100"
        )
    )

    # --------------------------------------------------------
    # Measure experiment
    # --------------------------------------------------------

    experiment_average, experiment_results = (
        evaluate_collection(
            experiment_collection,
            query_embeddings,
            "EXPERIMENT - chunk_size=70"
        )
    )

    # --------------------------------------------------------
    # Calculate latency change
    # --------------------------------------------------------

    latency_change_seconds = (
        experiment_average
        - baseline_average
    )

    latency_change_percent = (
        (
            latency_change_seconds
            / baseline_average
        )
        * 100
    )

    # --------------------------------------------------------
    # Complexity information
    # --------------------------------------------------------

    baseline_count = baseline_collection.count()

    experiment_count = experiment_collection.count()

    vector_count_change_percent = (
        (
            experiment_count
            - baseline_count
        )
        / baseline_count
    ) * 100

    # --------------------------------------------------------
    # Display final results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LATENCY RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBaseline average retrieval latency: "
        f"{baseline_average:.6f} seconds"
    )

    print(
        f"Experiment average retrieval latency: "
        f"{experiment_average:.6f} seconds"
    )

    print(
        f"\nLatency change: "
        f"{latency_change_seconds:+.6f} seconds"
    )

    print(
        f"Latency change percentage: "
        f"{latency_change_percent:+.2f}%"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPLEXITY / INDEX SIZE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBaseline vectors: "
        f"{baseline_count}"
    )

    print(
        f"Experiment vectors: "
        f"{experiment_count}"
    )

    print(
        f"Vector count change: "
        f"{vector_count_change_percent:+.2f}%"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {

        "experiment": "chunk_size_latency",

        "baseline": {
            "chunk_size": 100,
            "overlap": 20,
            "top_k": TOP_K,
            "vector_count": baseline_count,
            "average_retrieval_latency_seconds": baseline_average,
            "question_results": baseline_results
        },

        "experiment_configuration": {
            "chunk_size": 70,
            "overlap": 20,
            "top_k": TOP_K,
            "vector_count": experiment_count,
            "average_retrieval_latency_seconds": experiment_average,
            "question_results": experiment_results
        },

        "comparison": {
            "latency_change_seconds": latency_change_seconds,
            "latency_change_percent": latency_change_percent,
            "vector_count_change_percent": vector_count_change_percent
        },

        "runs_per_question": RUNS_PER_QUESTION,

        "note": (
            "Latency measures ChromaDB vector retrieval only. "
            "Query embedding generation is excluded because "
            "the same cached query embeddings are used for both "
            "configurations."
        )
    }

    output_file = (
        "day_10_retrieval/"
        "chunk_size_latency_report.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        f"\nLatency report saved to: "
        f"{output_file}"
    )


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":

    main()