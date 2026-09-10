import sys

sys.path.append(".")


from day_7_rag.retrieve import (
    create_chroma_client,
    get_collection,
    retrieve_chunks
)


# Experiment question
# Query rewritten using terminology from the policy

question = (
    "How should employees report an unplanned absence "
    "by notifying their manager or designated team contact?"
)


# Create ChromaDB client

chroma_client = create_chroma_client(
    "day_7_rag/chroma_db"
)


# Get collection

collection = get_collection(
    chroma_client,
    "employee_documents"
)


# Baseline:
# Original question
# top_k = 3
#
# Experiment:
# Rewritten question
# top_k = 3
#
# Only the query wording is changed.

results = retrieve_chunks(
    collection,
    question,
    top_k=3
)


# Display experiment information

print("Experiment: Test 5")
print("Variable changed: query_rewrite")
print("Baseline top_k: 3")
print("Experiment top_k: 3")

print("\nRewritten question:")
print(question)

print("\nExpected document: DOC003")

print("\nRetrieved chunks:\n")


for rank, document in enumerate(
    results["documents"][0],
    start=1
):

    metadata = results["metadatas"][0][rank - 1]
    distance = results["distances"][0][rank - 1]

    print(f"Rank {rank}")

    print(
        f"Document: {metadata['document_id']}"
    )

    print(
        f"Chunk: {metadata['chunk_index']}"
    )

    print(
        f"Distance: {distance}"
    )

    print(
        f"Text: {document}"
    )

    print("\n" + "-" * 60)
    