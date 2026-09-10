import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.append(".")

from dotenv import load_dotenv
from openai import OpenAI

from day_7_rag.retrieve import create_chroma_client


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "day_7_rag/chroma_db"
COLLECTION_NAME = "employee_documents"

TEST_SET_FILE = (
    "day_6_Vector_search/retrieval_test_set.json"
)

QUERY_EMBEDDINGS_CACHE = (
    "day_10_retrieval/query_embeddings.json"
)

EMBEDDING_MODEL = (
    "nvidia/llama-nemotron-embed-vl-1b-v2:free"
)

OUTPUT_FILE = (
    "day_10_retrieval/reranking_result_report.json"
)


# ============================================================
# FROZEN BASELINE
# ============================================================

BASELINE_TOP_K = 3


# ============================================================
# RERANKING EXPERIMENT
# ============================================================

CANDIDATE_TOP_K = 5
FINAL_TOP_K = 3


# ============================================================
# OPENROUTER CLIENT
# ============================================================

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY is not set."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# LOAD TEST SET
# ============================================================

def load_test_set():

    with open(
        TEST_SET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in [
            "questions",
            "test_cases",
            "retrieval_tests"
        ]:

            if key in data:
                return data[key]

    raise ValueError(
        "Could not find the question list in "
        f"{TEST_SET_FILE}"
    )


# ============================================================
# EXTRACT QUESTION
# ============================================================

def get_question(item):

    for key in [
        "question",
        "query",
        "user_question"
    ]:

        if key in item:
            return item[key]

    raise ValueError(
        f"Could not find question field in: {item}"
    )


# ============================================================
# EXTRACT EXPECTED DOCUMENT
# ============================================================

def get_expected_document(item):

    for key in [
        "expected_document",
        "expected_doc",
        "expected_document_id"
    ]:

        if key in item:
            return item[key]

    expected = item.get("expected")

    if isinstance(expected, dict):

        for key in [
            "document_id",
            "document",
            "expected_document"
        ]:

            if key in expected:
                return expected[key]

    raise ValueError(
        f"Could not find expected document in: {item}"
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# ============================================================
# TF-IDF RERANKING
# ============================================================

def calculate_tfidf_scores(
    question,
    documents
):
    """
    Calculate TF-IDF cosine similarity between
    the question and candidate chunks.

    TF-IDF is applied AFTER vector retrieval.
    It is used only to rerank the candidate pool.
    """

    question_tokens = tokenize(question)

    document_tokens = [
        tokenize(document)
        for document in documents
    ]

    all_documents = [
        question_tokens
    ] + document_tokens

    vocabulary = set()

    for tokens in all_documents:
        vocabulary.update(tokens)

    vocabulary = list(vocabulary)

    document_frequency = {}

    for term in vocabulary:

        count = 0

        for tokens in all_documents:

            if term in tokens:
                count += 1

        document_frequency[term] = count

    total_documents = len(all_documents)

    def create_vector(tokens):

        term_counts = {}

        for token in tokens:

            term_counts[token] = (
                term_counts.get(token, 0) + 1
            )

        vector = {}

        for term in vocabulary:

            term_frequency = term_counts.get(
                term,
                0
            )

            if term_frequency == 0:

                vector[term] = 0.0
                continue

            df = document_frequency[term]

            idf = (
                math.log(
                    (total_documents + 1)
                    /
                    (df + 1)
                )
                + 1
            )

            vector[term] = (
                term_frequency * idf
            )

        return vector

    question_vector = create_vector(
        question_tokens
    )

    document_vectors = [
        create_vector(tokens)
        for tokens in document_tokens
    ]

    def cosine_similarity(
        vector_a,
        vector_b
    ):

        dot_product = sum(
            vector_a[term]
            *
            vector_b[term]
            for term in vocabulary
        )

        magnitude_a = math.sqrt(
            sum(
                value * value
                for value in vector_a.values()
            )
        )

        magnitude_b = math.sqrt(
            sum(
                value * value
                for value in vector_b.values()
            )
        )

        if (
            magnitude_a == 0
            or
            magnitude_b == 0
        ):

            return 0.0

        return (
            dot_product
            /
            (magnitude_a * magnitude_b)
        )

    return [
        cosine_similarity(
            question_vector,
            document_vector
        )
        for document_vector in document_vectors
    ]


# ============================================================
# QUERY EMBEDDINGS WITH CACHE
# ============================================================

def create_query_embeddings(questions):
    """
    Load query embeddings from cache when available.

    If the cache does not exist or the questions differ,
    generate embeddings in one batch and save them.

    Returns:
        embeddings
        embedding_time
    """

    if os.path.exists(
        QUERY_EMBEDDINGS_CACHE
    ):

        print(
            "\nLoading cached query embeddings..."
        )

        with open(
            QUERY_EMBEDDINGS_CACHE,
            "r",
            encoding="utf-8"
        ) as file:

            cached_data = json.load(file)

        cached_questions = (
            cached_data.get("questions", [])
        )

        cached_model = (
            cached_data.get("model")
        )

        if (
            cached_questions == questions
            and
            cached_model == EMBEDDING_MODEL
        ):

            embeddings = (
                cached_data["embeddings"]
            )

            print(
                f"Loaded {len(embeddings)} "
                "cached query embeddings."
            )

            return embeddings, 0.0

        print(
            "Cached embeddings do not match "
            "the current test set or model."
        )

    print(
        f"\nGenerating embeddings for "
        f"{len(questions)} questions in one batch..."
    )

    start = time.perf_counter()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=questions,
        encoding_format="float"
    )

    embedding_time = (
        time.perf_counter() - start
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    cache_data = {
        "model": EMBEDDING_MODEL,
        "questions": questions,
        "embeddings": embeddings
    }

    cache_path = Path(
        QUERY_EMBEDDINGS_CACHE
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        cache_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cache_data,
            file,
            indent=4
        )

    print(
        f"Saved {len(embeddings)} query embeddings "
        f"to {QUERY_EMBEDDINGS_CACHE}"
    )

    print(
        f"Embedding generation time: "
        f"{embedding_time:.6f}s"
    )

    return embeddings, embedding_time


# ============================================================
# CHROMA QUERY
# ============================================================

def query_collection(
    collection,
    query_embedding,
    top_k
):

    return collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k
    )


# ============================================================
# BUILD CANDIDATE POOL
# ============================================================

def build_candidates(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    seen_chunks = set()

    for index, document in enumerate(
        documents
    ):

        metadata = metadatas[index]

        document_id = metadata.get(
            "document_id"
        )

        chunk_index = metadata.get(
            "chunk_index"
        )

        chunk_key = (
            document_id,
            chunk_index
        )

        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(chunk_key)

        candidates.append(
            {
                "document": document,
                "metadata": metadata,

                # Original vector-search information
                "original_rank": index + 1,
                "original_distance": distances[index],

                # Will be populated by reranking
                "reranking_score": None,
                "final_rank": None
            }
        )

    return candidates


# ============================================================
# RERANK CANDIDATES
# ============================================================

def rerank_candidates(
    question,
    candidates
):

    documents = [
        candidate["document"]
        for candidate in candidates
    ]

    scores = calculate_tfidf_scores(
        question,
        documents
    )

    for index, candidate in enumerate(
        candidates
    ):

        candidate["reranking_score"] = (
            scores[index]
        )

    reranked = sorted(
        candidates,
        key=lambda item: (
            item["reranking_score"],
            -item["original_distance"]
        ),
        reverse=True
    )

    final_results = reranked[
        :FINAL_TOP_K
    ]

    for final_rank, item in enumerate(
        final_results,
        start=1
    ):

        item["final_rank"] = final_rank

    return reranked


# ============================================================
# FIND DOCUMENT RANK
# ============================================================

def find_document_rank(
    results,
    expected_document
):

    for rank, metadata in enumerate(
        results["metadatas"][0],
        start=1
    ):

        if (
            metadata.get("document_id")
            ==
            expected_document
        ):

            return rank

    return None


def find_reranked_document_rank(
    reranked_results,
    expected_document
):

    for item in reranked_results:

        if (
            item["metadata"].get(
                "document_id"
            )
            ==
            expected_document
            and
            item["final_rank"] is not None
        ):

            return item["final_rank"]

    return None


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    question_results,
    rank_field
):

    hits = 0
    reciprocal_rank_sum = 0.0

    for result in question_results:

        rank = result[rank_field]

        if rank is not None:

            if rank <= FINAL_TOP_K:

                hits += 1

                reciprocal_rank_sum += (
                    1.0 / rank
                )

    total = len(question_results)

    if total == 0:

        return {
            "hit_rate": 0.0,
            "mrr": 0.0
        }

    return {
        "hit_rate": hits / total,
        "mrr": reciprocal_rank_sum / total
    }


# ============================================================
# REGRESSION CHECK
# ============================================================

def check_regressions(
    question_results
):

    regressions = []

    for result in question_results:

        baseline_rank = result[
            "baseline_rank"
        ]

        experiment_rank = result[
            "experiment_rank"
        ]

        baseline_hit = (
            baseline_rank is not None
            and
            baseline_rank <= BASELINE_TOP_K
        )

        experiment_hit = (
            experiment_rank is not None
            and
            experiment_rank <= FINAL_TOP_K
        )

        if (
            baseline_hit
            and
            not experiment_hit
        ):

            regressions.append(
                {
                    "question_id": result[
                        "question_id"
                    ],
                    "question": result[
                        "question"
                    ],
                    "baseline_rank": baseline_rank,
                    "experiment_rank": experiment_rank
                }
            )

    return regressions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DAY 10 - FULL RERANKING VALIDATION")
    print("=" * 70)

    print("\nBaseline configuration:")
    print(
        f"Vector top_k = {BASELINE_TOP_K}"
    )

    print("\nReranking experiment:")
    print(
        f"Candidate pool = {CANDIDATE_TOP_K}"
    )

    print(
        f"Final top_k = {FINAL_TOP_K}"
    )

    print(
        "Reranking method = TF-IDF cosine similarity"
    )

    # --------------------------------------------------------
    # Load test set
    # --------------------------------------------------------

    test_set = load_test_set()

    questions = [
        get_question(item)
        for item in test_set
    ]

    expected_documents = [
        get_expected_document(item)
        for item in test_set
    ]

    print(
        f"\nQuestions loaded: {len(questions)}"
    )

    # --------------------------------------------------------
    # Create Chroma client
    # --------------------------------------------------------

    chroma_client = create_chroma_client(
        CHROMA_DIR
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    # --------------------------------------------------------
    # Query embeddings
    # --------------------------------------------------------

    (
        query_embeddings,
        embedding_time
    ) = create_query_embeddings(
        questions
    )

    # --------------------------------------------------------
    # Evaluate every question
    # --------------------------------------------------------

    question_results = []

    total_retrieval_time = 0.0
    total_reranking_time = 0.0

    for index, question in enumerate(
        questions
    ):

        question_id = (
            test_set[index].get(
                "question_id",
                test_set[index].get(
                    "id",
                    f"Q{index + 1}"
                )
            )
        )

        expected_document = (
            expected_documents[index]
        )

        print("\n" + "-" * 70)

        print(
            f"{question_id}: {question}"
        )

        # ----------------------------------------------------
        # ONE vector retrieval
        # ----------------------------------------------------

        start = time.perf_counter()

        candidate_results = query_collection(
            collection,
            query_embeddings[index],
            CANDIDATE_TOP_K
        )

        retrieval_latency = (
            time.perf_counter() - start
        )

        total_retrieval_time += (
            retrieval_latency
        )

        # ----------------------------------------------------
        # Build candidate pool
        # ----------------------------------------------------

        candidates = build_candidates(
            candidate_results
        )

        # ----------------------------------------------------
        # Baseline ranking
        #
        # The frozen baseline is simply the first
        # BASELINE_TOP_K results from the original
        # vector ranking.
        # ----------------------------------------------------

        baseline_rank = (
            find_document_rank(
                {
                    "metadatas": [
                        [
                            candidate["metadata"]
                            for candidate in candidates[
                                :BASELINE_TOP_K
                            ]
                        ]
                    ]
                },
                expected_document
            )
        )

        # ----------------------------------------------------
        # Reranking
        # ----------------------------------------------------

        start = time.perf_counter()

        reranked_candidates = (
            rerank_candidates(
                question,
                candidates
            )
        )

        reranking_latency = (
            time.perf_counter() - start
        )

        total_reranking_time += (
            reranking_latency
        )

        # ----------------------------------------------------
        # Experiment rank
        # ----------------------------------------------------

        experiment_rank = (
            find_reranked_document_rank(
                reranked_candidates,
                expected_document
            )
        )

        # ----------------------------------------------------
        # Hit status
        # ----------------------------------------------------

        baseline_hit = (
            baseline_rank is not None
            and
            baseline_rank <= BASELINE_TOP_K
        )

        experiment_hit = (
            experiment_rank is not None
            and
            experiment_rank <= FINAL_TOP_K
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print(
            f"Expected document: "
            f"{expected_document}"
        )

        print(
            f"Baseline rank: "
            f"{baseline_rank}"
        )

        print(
            f"Reranked rank: "
            f"{experiment_rank}"
        )

        print(
            f"Baseline hit: "
            f"{baseline_hit}"
        )

        print(
            f"Reranked hit: "
            f"{experiment_hit}"
        )

        print(
            f"Vector retrieval latency: "
            f"{retrieval_latency:.6f}s"
        )

        print(
            f"TF-IDF reranking latency: "
            f"{reranking_latency:.6f}s"
        )

        # ----------------------------------------------------
        # Display complete candidate audit trail
        # ----------------------------------------------------

        print(
            "\nCandidate pool and final ordering:"
        )

        for candidate in candidates:

            metadata = candidate[
                "metadata"
            ]

            print(
                f"  Original Rank "
                f"{candidate['original_rank']}"
                f" -> Final Rank "
                f"{candidate['final_rank']}"
                f" | "
                f"{metadata.get('document_id')}"
                f" | Chunk "
                f"{metadata.get('chunk_index')}"
                f" | Distance "
                f"{candidate['original_distance']:.6f}"
                f" | TF-IDF "
                f"{candidate['reranking_score']:.6f}"
            )

        # ----------------------------------------------------
        # Save per-question result
        # ----------------------------------------------------

        question_results.append(
            {
                "question_id": question_id,

                "question": question,

                "expected_document":
                    expected_document,

                "baseline_rank":
                    baseline_rank,

                "baseline_hit":
                    baseline_hit,

                "experiment_rank":
                    experiment_rank,

                "experiment_hit":
                    experiment_hit,

                "baseline_top_k":
                    BASELINE_TOP_K,

                "candidate_top_k":
                    CANDIDATE_TOP_K,

                "final_top_k":
                    FINAL_TOP_K,

                "vector_retrieval_latency_seconds":
                    retrieval_latency,

                "reranking_latency_seconds":
                    reranking_latency,

                # Preserve complete candidate
                # information for auditability.
                "candidate_pool": [
                    {
                        "document_id":
                            candidate[
                                "metadata"
                            ].get(
                                "document_id"
                            ),

                        "chunk_index":
                            candidate[
                                "metadata"
                            ].get(
                                "chunk_index"
                            ),

                        "original_rank":
                            candidate[
                                "original_rank"
                            ],

                        "original_distance":
                            candidate[
                                "original_distance"
                            ],

                        "reranking_score":
                            candidate[
                                "reranking_score"
                            ],

                        "final_rank":
                            candidate[
                                "final_rank"
                            ]
                    }
                    for candidate in candidates
                ]
            }
        )

    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    baseline_hits = sum(
        1
        for result in question_results
        if result["baseline_hit"]
    )

    baseline_rr_sum = 0.0

    for result in question_results:

        rank = result["baseline_rank"]

        if (
            rank is not None
            and
            rank <= BASELINE_TOP_K
        ):

            baseline_rr_sum += (
                1.0 / rank
            )

    total_questions = len(
        question_results
    )

    baseline_metrics = {
        "hit_rate": (
            baseline_hits / total_questions
            if total_questions
            else 0.0
        ),

        "mrr": (
            baseline_rr_sum / total_questions
            if total_questions
            else 0.0
        )
    }

    experiment_metrics = calculate_metrics(
        question_results,
        "experiment_rank"
    )

    # ========================================================
    # METRIC CHANGES
    # ========================================================

    hit_rate_change = (
        experiment_metrics["hit_rate"]
        -
        baseline_metrics["hit_rate"]
    )

    mrr_change = (
        experiment_metrics["mrr"]
        -
        baseline_metrics["mrr"]
    )

    # ========================================================
    # REGRESSION CHECK
    # ========================================================

    regressions = check_regressions(
        question_results
    )

    # ========================================================
    # LATENCY
    # ========================================================

    average_vector_retrieval_latency = (
        total_retrieval_time / total_questions
        if total_questions
        else 0.0
    )

    average_reranking_latency = (
        total_reranking_time / total_questions
        if total_questions
        else 0.0
    )

    average_experiment_latency = (
        average_vector_retrieval_latency
        +
        average_reranking_latency
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    report = {

        "experiment":
            "Day 10 - Full Reranking Validation",

        "objective":
            "Evaluate TF-IDF reranking after vector "
            "retrieval using a larger candidate pool.",

        "baseline_configuration": {

            "embedding_model":
                EMBEDDING_MODEL,

            "top_k":
                BASELINE_TOP_K,

            "retrieval_method":
                "Vector search"
        },

        "experiment_configuration": {

            "embedding_model":
                EMBEDDING_MODEL,

            "candidate_top_k":
                CANDIDATE_TOP_K,

            "final_top_k":
                FINAL_TOP_K,

            "reranking_method":
                "TF-IDF cosine similarity",

            "retrieval_method":
                "Vector search followed by TF-IDF reranking"
        },

        "metrics": {

            "baseline":
                baseline_metrics,

            "reranking":
                experiment_metrics,

            "hit_rate_change":
                hit_rate_change,

            "mrr_change":
                mrr_change
        },

        "regression_check": {

            "regression_count":
                len(regressions),

            "regressions":
                regressions,

            "no_regressions":
                len(regressions) == 0
        },

        "latency": {

            "batch_query_embedding_seconds":
                embedding_time,

            "total_vector_retrieval_seconds":
                total_retrieval_time,

            "total_reranking_seconds":
                total_reranking_time,

            "average_vector_retrieval_seconds":
                average_vector_retrieval_latency,

            "average_reranking_seconds":
                average_reranking_latency,

            "average_experiment_latency_seconds":
                average_experiment_latency
        },

        "question_results":
            question_results
    }

    # ========================================================
    # SAVE REPORT
    # ========================================================

    output_path = Path(
        OUTPUT_FILE
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("RERANKING AGGREGATE RESULTS")
    print("=" * 70)

    print(
        f"\nBaseline Hit Rate: "
        f"{baseline_metrics['hit_rate']:.3f}"
    )

    print(
        f"Reranking Hit Rate: "
        f"{experiment_metrics['hit_rate']:.3f}"
    )

    print(
        f"\nBaseline MRR: "
        f"{baseline_metrics['mrr']:.3f}"
    )

    print(
        f"Reranking MRR: "
        f"{experiment_metrics['mrr']:.3f}"
    )

    print(
        f"\nHit Rate change: "
        f"{hit_rate_change:+.3f}"
    )

    print(
        f"MRR change: "
        f"{mrr_change:+.3f}"
    )

    print(
        f"\nRegressions: "
        f"{len(regressions)}"
    )

    print(
        f"\nAverage vector retrieval latency: "
        f"{average_vector_retrieval_latency:.6f}s"
    )

    print(
        f"Average reranking latency: "
        f"{average_reranking_latency:.6f}s"
    )

    print(
        f"Average total experiment latency: "
        f"{average_experiment_latency:.6f}s"
    )

    print(
        "\nResult file saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()