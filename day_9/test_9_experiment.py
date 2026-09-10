import sys

sys.path.append(".")


from day_7_rag.retrieve import (
    create_chroma_client,
    get_collection,
    retrieve_chunks
)


# Experiment question

question = (
    "What should employees do if they have a potential "
    "conflict of interest?"
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
# top_k = 3
# no metadata filter
#
# Experiment:
# top_k = 3
# filter to DOC005 only


results = retrieve_chunks(
    collection,
    question,
    top_k=3,
    where={
        "document_id": "DOC005"
    }
)


# Display experiment information

print("Experiment: Test 9")
print("Variable changed: metadata_filter")
print("Baseline filter: none")
print("Experiment filter: DOC005")
print("top_k: 3")

print("\nQuestion:")
print(question)

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