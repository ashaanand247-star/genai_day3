from day_8.citations import (
    extract_citations,
    validate_citations,
    build_allowed_citations
)

from day_7_rag.retrieve import (
    create_chroma_client,
    get_collection,
    retrieve_chunks
)

# Question

question = "When should employees submit their leave requests?"


# Create ChromaDB client

chroma_client = create_chroma_client(
    "day_7_rag/chroma_db"
)


# Get collection

collection = get_collection(
    chroma_client,
    "employee_documents"
)


# Retrieve chunks

results = retrieve_chunks(
    collection,
    question,
    top_k=3
)


# Build allowed citations from retrieved chunks

allowed_citations = build_allowed_citations(
    results
)


# Simulated LLM answer

answer = """
Employees should submit leave requests through the approved
internal leave system.
[Source: Employee Leave Policy | Document: DOC001 | Chunk: 1]

Employees must wait for manager approval.
[Source: Employee Leave Policy | Document: DOC001 | Chunk: 2]

The company provides 30 days of annual leave.
[Source: Employee Leave Policy | Document: DOC001 | Chunk: 5]
"""


# Extract citations from answer

citations = extract_citations(
    answer
)


# Validate citations

valid_citations = validate_citations(
    citations,
    allowed_citations
)


# Display results

print("Allowed citations:")
print(allowed_citations)

print("\nExtracted citations:")
print(citations)

print("\nValid citations:")
print(valid_citations)