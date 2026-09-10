import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# Allow imports from the project root
sys.path.append(".")


# Load environment variables
load_dotenv()


# Configuration

SOURCE_FILE = Path(
    "day5_preprocessing/cleaned_documents/DOC004_employee_onboarding_guide.md"
)

CHROMA_DIR = "day_7_rag/chroma_db"

COLLECTION_NAME = "employee_documents"

EMBEDDING_MODEL = (
    "nvidia/llama-nemotron-embed-vl-1b-v2:free"
)

QUESTION = (
    "What training is required during employee onboarding?"
)

EXPECTED_DOCUMENT = "DOC004"

BASELINE_CHUNK_SIZE = 100
EXPERIMENT_CHUNK_SIZE = 150
OVERLAP = 20

TOP_K = 3


# OpenRouter client

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# --------------------------------------------------
# Function 1: Load document
# --------------------------------------------------

def load_document(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# --------------------------------------------------
# Function 2: Chunk text
# --------------------------------------------------

def chunk_text(
    text,
    chunk_size=100,
    overlap=20
):

    if overlap >= chunk_size:

        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    lines = text.splitlines()

    line_data = []

    for line in lines:

        words = line.split()

        if words:

            line_data.append(words)

    chunks = []

    start_line = 0

    while start_line < len(line_data):

        chunk_words = []

        end_line = start_line

        while end_line < len(line_data):

            next_line = line_data[end_line]

            if (
                chunk_words
                and len(chunk_words)
                + len(next_line)
                > chunk_size
            ):

                break

            chunk_words.extend(next_line)

            end_line += 1

        if not chunk_words:

            break

        chunks.append(
            " ".join(chunk_words)
        )

        if end_line >= len(line_data):

            break

        overlap_words = 0

        next_start_line = end_line

        while next_start_line > start_line:

            next_start_line -= 1

            overlap_words += len(
                line_data[next_start_line]
            )

            if overlap_words >= overlap:

                break

        if next_start_line == start_line:

            start_line = end_line

        else:

            start_line = next_start_line

    return chunks


# --------------------------------------------------
# Function 3: Create embeddings
# --------------------------------------------------

def create_embeddings(texts):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float"
    )

    return [
        item.embedding
        for item in response.data
    ]


# --------------------------------------------------
# Function 4: Retrieve from temporary collection
# --------------------------------------------------

def retrieve_from_chunks(
    chunks,
    question,
    top_k=3
):

    embeddings = create_embeddings(
        chunks
    )

    query_embedding = create_embeddings(
        [question]
    )[0]

    temp_client = chromadb.Client()

    collection = temp_client.get_or_create_collection(
    name="chunk_experiment"
)

    ids = [
        f"DOC004_chunk_{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "document_id": EXPECTED_DOCUMENT,
            "chunk_index": index
        }
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(
            top_k,
            len(chunks)
        )
    )

    return results


# --------------------------------------------------
# Function 5: Display retrieval results
# --------------------------------------------------

def display_results(
    results
):

    for index, document in enumerate(
        results["documents"][0]
    ):

        metadata = (
            results["metadatas"][0][index]
        )

        distance = (
            results["distances"][0][index]
        )

        print(
            f"\nRank {index + 1}"
        )

        print(
            "Document:",
            metadata["document_id"]
        )

        print(
            "Chunk:",
            metadata["chunk_index"]
        )

        print(
            "Distance:",
            distance
        )

        print(
            "Text:",
            document
        )

        print(
            "-" * 60
        )


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    print("Experiment: Test 8")

    print(
        "Variable changed: chunk_size"
    )

    print(
        f"Baseline chunk_size: "
        f"{BASELINE_CHUNK_SIZE}"
    )

    print(
        f"Experiment chunk_size: "
        f"{EXPERIMENT_CHUNK_SIZE}"
    )

    print(
        f"Overlap kept constant: "
        f"{OVERLAP}"
    )

    print(
        "\nQuestion:"
    )

    print(QUESTION)

    print(
        "\nExpected document:",
        EXPECTED_DOCUMENT
    )


    # ----------------------------------------------
    # Load source document
    # ----------------------------------------------

    document_text = load_document(
        SOURCE_FILE
    )


    # ----------------------------------------------
    # Create baseline chunks
    # ----------------------------------------------

    baseline_chunks = chunk_text(
        document_text,
        chunk_size=BASELINE_CHUNK_SIZE,
        overlap=OVERLAP
    )


    # ----------------------------------------------
    # Create experiment chunks
    # ----------------------------------------------

    experiment_chunks = chunk_text(
        document_text,
        chunk_size=EXPERIMENT_CHUNK_SIZE,
        overlap=OVERLAP
    )


    print(
        "\nChunking comparison:"
    )

    print(
        f"Baseline: "
        f"{len(baseline_chunks)} chunks"
    )

    print(
        f"Experiment: "
        f"{len(experiment_chunks)} chunks"
    )


    # ----------------------------------------------
    # Baseline retrieval
    # ----------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "BASELINE RETRIEVAL"
    )

    print(
        f"chunk_size={BASELINE_CHUNK_SIZE}, "
        f"overlap={OVERLAP}"
    )

    print(
        "=" * 60
    )

    baseline_results = retrieve_from_chunks(
        baseline_chunks,
        QUESTION,
        TOP_K
    )

    display_results(
        baseline_results
    )


    # ----------------------------------------------
    # Experiment retrieval
    # ----------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "EXPERIMENT RETRIEVAL"
    )

    print(
        f"chunk_size={EXPERIMENT_CHUNK_SIZE}, "
        f"overlap={OVERLAP}"
    )

    print(
        "=" * 60
    )

    experiment_results = retrieve_from_chunks(
        experiment_chunks,
        QUESTION,
        TOP_K
    )

    display_results(
        experiment_results
    )


    # ----------------------------------------------
    # Compare top result
    # ----------------------------------------------

    baseline_top_document = (
        baseline_results["metadatas"][0][0]
        ["document_id"]
    )

    experiment_top_document = (
        experiment_results["metadatas"][0][0]
        ["document_id"]
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "RESULT COMPARISON"
    )

    print(
        "=" * 60
    )

    print(
        "Baseline top document:",
        baseline_top_document
    )

    print(
        "Experiment top document:",
        experiment_top_document
    )


    if (
        baseline_top_document
        == EXPECTED_DOCUMENT
    ):

        print(
            "Baseline result: PASS"
        )

    else:

        print(
            "Baseline result: FAIL"
        )


    if (
        experiment_top_document
        == EXPECTED_DOCUMENT
    ):

        print(
            "Experiment result: PASS"
        )

    else:

        print(
            "Experiment result: FAIL"
        )


    print(
        "\nTest 8 completed."
    )


# Program entry point

if __name__ == "__main__":

    main()