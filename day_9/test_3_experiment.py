import sys

sys.path.append(".")


from day_7_rag.retrieve import (
    create_chroma_client,
    get_collection,
    retrieve_chunks
)


# Experiment question

question = "What are the requirements for working from home?"


# Create ChromaDB client

chroma_client = create_chroma_client(
    "day_7_rag/chroma_db"
)


# Get collection

collection = get_collection(
    chroma_client,
    "employee_documents"
)


# Baseline: top_k = 3
# Experiment: top_k = 5

results = retrieve_chunks(
    collection,
    question,
    top_k=5
)


# Display experiment information

print("Experiment: Test 3")
print("Variable changed: top_k")
print("Baseline top_k: 3")
print("Experiment top_k: 5")

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