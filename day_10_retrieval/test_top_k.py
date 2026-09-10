import sys

sys.path.append(".")

from day_7_rag.retrieve import (
    create_chroma_client,
    get_collection,
    retrieve_chunks
)


CHROMA_DIR = "day_7_rag/chroma_db"

COLLECTION_NAME = "employee_documents"

QUESTION = (
    "What should employees do if they become unavailable "
    "while working remotely?"
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

    chroma_client = create_chroma_client(
        CHROMA_DIR
    )

    collection = get_collection(
        chroma_client,
        COLLECTION_NAME
    )

    baseline_results = retrieve_chunks(
        collection,
        QUESTION,
        top_k=3
    )

    experiment_results = retrieve_chunks(
        collection,
        QUESTION,
        top_k=2
    )

    print("DAY 10 - TOP-K EXPERIMENT")

    print("\nQuestion:")
    print(QUESTION)

    print("\nBaseline configuration:")
    print("top_k = 3")

    print("\nExperiment configuration:")
    print("top_k = 2")

    display_results(
        "BASELINE RESULTS",
        baseline_results
    )

    display_results(
        "EXPERIMENT RESULTS",
        experiment_results
    )


if __name__ == "__main__":
    main()