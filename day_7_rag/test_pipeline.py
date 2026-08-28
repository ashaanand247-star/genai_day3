from ingest import (
    create_chroma_client
)

from retrieve import (
    get_collection,
    retrieve_chunks,
    prepare_context
)


CHROMA_DIR = "day_7_rag/chroma_db"

COLLECTION_NAME = "employee_documents"

TEST_QUESTION = (
    "How does an employee request leave?"
)


def test_ingest_then_retrieve():

    chroma_client = create_chroma_client(
        CHROMA_DIR
    )

    collection = get_collection(
        chroma_client,
        COLLECTION_NAME
    )

    results = retrieve_chunks(
        collection,
        TEST_QUESTION,
        top_k=3,
        distance_threshold=1.2
    )

    context = prepare_context(
        results,
        max_chunks=3
    )

    assert context != ""

    assert "Employee Leave Policy" in context

    print(
        "Integration test passed."
    )


if __name__ == "__main__":
    test_ingest_then_retrieve()