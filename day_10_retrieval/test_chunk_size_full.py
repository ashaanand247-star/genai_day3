import os
import sys
import json
import time
import uuid

sys.path.append(".")

import chromadb

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError

from day_7_rag.retrieve import (
    create_chroma_client,
    get_collection,
    retrieve_chunks
)

load_dotenv()

EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

SOURCE_DIR = "day5_preprocessing/cleaned_documents"

BASELINE_CHUNK_SIZE = 100
EXPERIMENT_CHUNK_SIZE = 70
OVERLAP = 20
TOP_K = 3

BATCH_SIZE = 10

CACHE_FILE = (
    "day_10_retrieval/"
    "chunk70_embeddings.jsonl"
)

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

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


def chunk_text(text, chunk_size, overlap):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        start = end - overlap

    return chunks


def load_cached_embeddings():

    cache = {}

    if not os.path.exists(CACHE_FILE):
        return cache

    with open(
        CACHE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            record = json.loads(line)

            cache[record["id"]] = record["embedding"]

    return cache


def save_embedding(
    chunk_id,
    document,
    chunk_index,
    embedding
):

    with open(
        CACHE_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        record = {
            "id": chunk_id,
            "document_id": document,
            "chunk_index": chunk_index,
            "embedding": embedding
        }

        file.write(
            json.dumps(record)
            + "\n"
        )


def create_embeddings_for_chunks(
    chunks_to_embed,
    cached_embeddings
):

    embeddings = {}

    for item in chunks_to_embed:

        chunk_id = item["id"]

        if chunk_id in cached_embeddings:

            embeddings[chunk_id] = (
                cached_embeddings[chunk_id]
            )

    pending = [
        item
        for item in chunks_to_embed
        if item["id"] not in embeddings
    ]

    print(
        f"\nCached embeddings: "
        f"{len(embeddings)}"
    )

    print(
        f"New embeddings required: "
        f"{len(pending)}"
    )

    for start in range(
        0,
        len(pending),
        BATCH_SIZE
    ):

        batch = pending[
            start:start + BATCH_SIZE
        ]

        texts = [
            item["text"]
            for item in batch
        ]

        print(
            f"\nEmbedding batch "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"of {len(pending)}"
        )

        while True:

            try:

                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                    encoding_format="float"
                )

                break

            except RateLimitError:

                print(
                    "\nOpenRouter rate limit reached."
                )

                print(
                    "Waiting 30 seconds before retry..."
                )

                time.sleep(30)

        for item, embedding_data in zip(
            batch,
            response.data
        ):

            embedding = embedding_data.embedding

            embeddings[item["id"]] = embedding

            save_embedding(
                item["id"],
                item["document_id"],
                item["chunk_index"],
                embedding
            )

        # Small pause between batches
        time.sleep(3)

    return embeddings


def prepare_experiment_chunks():

    all_chunks = []

    files = [
        filename
        for filename in os.listdir(SOURCE_DIR)
        if filename.endswith(".md")
    ]

    files.sort()

    print(
        f"\nDocuments found: "
        f"{len(files)}"
    )

    for filename in files:

        source_path = os.path.join(
            SOURCE_DIR,
            filename
        )

        with open(
            source_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        document_id = filename.split("_")[0]

        chunks = chunk_text(
            text,
            EXPERIMENT_CHUNK_SIZE,
            OVERLAP
        )

        print(
            f"{document_id}: "
            f"{len(chunks)} chunks"
        )

        for index, chunk in enumerate(
            chunks
        ):

            chunk_id = (
                f"{document_id}_chunk70_{index}"
            )

            all_chunks.append({
                "id": chunk_id,
                "document_id": document_id,
                "chunk_index": index,
                "text": chunk
            })

    return all_chunks


def build_experiment_collection():

    all_chunks = prepare_experiment_chunks()

    cached_embeddings = (
        load_cached_embeddings()
    )

    embeddings = create_embeddings_for_chunks(
        all_chunks,
        cached_embeddings
    )

    temp_client = chromadb.Client()

    collection = temp_client.create_collection(
        name=f"chunk70_{uuid.uuid4().hex}"
    )

    collection.add(
        ids=[
            item["id"]
            for item in all_chunks
        ],
        documents=[
            item["text"]
            for item in all_chunks
        ],
        embeddings=[
            embeddings[item["id"]]
            for item in all_chunks
        ],
        metadatas=[
            {
                "document_id": item["document_id"],
                "chunk_index": item["chunk_index"]
            }
            for item in all_chunks
        ]
    )

    print(
        f"\nTotal experiment chunks: "
        f"{len(all_chunks)}"
    )

    return collection


def evaluate(collection):

    results = []

    for (
        question_id,
        question,
        expected_document
    ) in QUESTIONS:

        retrieval = retrieve_chunks(
            collection,
            question,
            top_k=TOP_K
        )

        metadatas = retrieval["metadatas"][0]

        distances = retrieval["distances"][0]

        expected_rank = None

        for rank, metadata in enumerate(
            metadatas,
            start=1
        ):

            if (
                metadata["document_id"]
                == expected_document
            ):

                expected_rank = rank

                break

        hit = expected_rank is not None

        reciprocal_rank = (
            1 / expected_rank
            if expected_rank
            else 0
        )

        results.append({
            "question_id": question_id,
            "expected_document": expected_document,
            "expected_rank": expected_rank,
            "hit": hit,
            "reciprocal_rank": reciprocal_rank,
            "top_distance": (
                distances[0]
                if distances
                else None
            ),
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


def display_results(
    label,
    results
):

    print(
        "\n" + "=" * 80
    )

    print(label)

    print(
        "=" * 80
    )

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

    print(
        "DAY 10 - FULL CHUNK SIZE VALIDATION"
    )

    print(
        "\nBaseline:"
    )

    print(
        "Existing ChromaDB "
        "(chunk_size=100, overlap=20)"
    )

    baseline_client = create_chroma_client(
        "day_7_rag/chroma_db"
    )

    baseline_collection = get_collection(
        baseline_client,
        "employee_documents"
    )

    print(
        "\nEvaluating baseline..."
    )

    baseline_results = evaluate(
        baseline_collection
    )

    print(
        "\nBuilding chunk_size=70 "
        "experiment index..."
    )

    experiment_collection = (
        build_experiment_collection()
    )

    print(
        "\nEvaluating chunk_size=70..."
    )

    experiment_results = evaluate(
        experiment_collection
    )

    display_results(
        "BASELINE - chunk_size=100",
        baseline_results
    )

    display_results(
        "EXPERIMENT - chunk_size=70",
        experiment_results
    )

    baseline_hit_rate, baseline_mrr = (
        calculate_metrics(
            baseline_results
        )
    )

    experiment_hit_rate, experiment_mrr = (
        calculate_metrics(
            experiment_results
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "AGGREGATE METRICS"
    )

    print(
        "=" * 80
    )

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

    print(
        "\n" + "=" * 80
    )

    print(
        "REGRESSION CHECK"
    )

    print(
        "=" * 80
    )

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

        print(
            regressions
        )

    else:

        print(
            "\nNo retrieval regressions found."
        )


if __name__ == "__main__":
    main()