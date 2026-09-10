import os
import sys
import uuid

sys.path.append(".")

import chromadb

from dotenv import load_dotenv
from openai import OpenAI

from day_7_rag.retrieve import (
    create_chroma_client,
    get_collection,
    retrieve_chunks
)


load_dotenv()


EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

QUESTION = (
    "What training is required during employee onboarding?"
)

SOURCE_FILE = (
    "day5_preprocessing/cleaned_documents/"
    "DOC004_employee_onboarding_guide.md"
)

BASELINE_CHUNK_SIZE = 100
EXPERIMENT_CHUNK_SIZE = 70
OVERLAP = 20
TOP_K = 3


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

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start = end - overlap

    return chunks


def create_embeddings(chunks):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks,
        encoding_format="float"
    )

    return [
        item.embedding
        for item in response.data
    ]


def retrieve_from_chunks(chunks, embeddings):

    temp_client = chromadb.Client()

    collection = temp_client.create_collection(
    name=f"experiment_collection_{uuid.uuid4().hex}"
)

    ids = [
        f"chunk_{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "document_id": "DOC004",
            "chunk_index": index
        }
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return retrieve_chunks(
        collection,
        QUESTION,
        top_k=TOP_K
    )


def display_results(label, results):

    print(f"\n{'=' * 70}")
    print(label)
    print(f"{'=' * 70}")

    for rank, document in enumerate(
        results["documents"][0],
        start=1
    ):

        metadata = results["metadatas"][0][rank - 1]

        distance = results["distances"][0][rank - 1]

        print(f"\nRank: {rank}")
        print(f"Document: {metadata['document_id']}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(f"Distance: {distance}")
        print(f"Text: {document}")


def main():

    with open(
        SOURCE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()


    baseline_chunks = chunk_text(
        text,
        BASELINE_CHUNK_SIZE,
        OVERLAP
    )


    experiment_chunks = chunk_text(
        text,
        EXPERIMENT_CHUNK_SIZE,
        OVERLAP
    )


    print("DAY 10 - CHUNK SIZE EXPERIMENT")

    print("\nQuestion:")
    print(QUESTION)

    print("\nBaseline configuration:")
    print("chunk_size = 100")
    print("overlap = 20")
    print("top_k = 3")

    print("\nExperiment configuration:")
    print("chunk_size = 70")
    print("overlap = 20")
    print("top_k = 3")

    print("\nBaseline chunk count:")
    print(len(baseline_chunks))

    print("\nExperiment chunk count:")
    print(len(experiment_chunks))


    baseline_embeddings = create_embeddings(
        baseline_chunks
    )


    experiment_embeddings = create_embeddings(
        experiment_chunks
    )


    baseline_results = retrieve_from_chunks(
        baseline_chunks,
        baseline_embeddings
    )


    experiment_results = retrieve_from_chunks(
        experiment_chunks,
        experiment_embeddings
    )


    display_results(
        "BASELINE RESULTS",
        baseline_results
    )


    display_results(
        "CHUNK SIZE EXPERIMENT RESULTS",
        experiment_results
    )


if __name__ == "__main__":
    main()