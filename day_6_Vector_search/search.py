import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# Configuration

EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
DEFAULT_TOP_K = 3

CHROMA_DB_PATH = "day_6_Vector_search/chroma_db"
COLLECTION_NAME = "employee_documents"

# Function 1: Create OpenRouter client


def create_embedding_client():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    return client

# Function 2: Connect to ChromaDB


def get_collection(db_path, collection_name):
    chroma_client = chromadb.PersistentClient(
        path=db_path
    )

    collection = chroma_client.get_collection(
        name=collection_name
    )

    return collection

# Function 3: Get search inputs


def get_search_inputs():
    question = input("Enter your question: ")

    top_k_input = input(
        f"Enter top-k (press Enter for {DEFAULT_TOP_K}): "
    )

    if top_k_input.strip():
        top_k = int(top_k_input)
    else:
        top_k = DEFAULT_TOP_K

    document_id = input(
        "Enter document ID filter "
        "(press Enter for all documents): "
    ).strip()

    threshold_input = input(
        "Enter maximum distance threshold "
        "(press Enter for no threshold): "
    )

    if threshold_input.strip():
        max_distance = float(threshold_input)
    else:
        max_distance = None

    return question, top_k, document_id, max_distance

# Function 4: Create question embedding


def create_question_embedding(
    client,
    question,
    embedding_model
):
    response = client.embeddings.create(
        model=embedding_model,
        input=question,
        encoding_format="float"
    )

    question_embedding = response.data[0].embedding

    return question_embedding

# Function 5: Build metadata filter


def build_metadata_filter(document_id):

    where_filter = None

    if document_id:
        where_filter = {
            "document_id": document_id
        }

    return where_filter

# Function 6: Search ChromaDB


def search_documents(
    collection,
    question_embedding,
    top_k,
    where_filter
):
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where=where_filter
    )

    return results

# Function 7: Display search results


def display_results(results, max_distance):

    print("\nSearch results:\n")

    result_count = len(
        results["documents"][0]
    )

    displayed_count = 0

    for i in range(result_count):

        distance = results["distances"][0][i]

        # Apply optional maximum distance threshold
        if (
            max_distance is not None
            and distance > max_distance
        ):
            continue

        metadata = results["metadatas"][0][i]
        document = results["documents"][0][i]

        displayed_count += 1

        print(
            f"Result {displayed_count}"
        )

        print(
            "Distance:",
            distance
        )

        print(
            "Document ID:",
            metadata["document_id"]
        )

        print(
            "Title:",
            metadata["title"]
        )

        print(
            "Source:",
            metadata["source_path"]
        )

        print(
            "Text:",
            document
        )

        print("-" * 60)

    if displayed_count == 0:
        print(
            "No results matched "
            "the specified threshold."
        )

# Function 8: Main program


def main():

    # Create OpenRouter client
    client = create_embedding_client()

    # Connect to ChromaDB
    collection = get_collection(
        CHROMA_DB_PATH,
        COLLECTION_NAME
    )

    # Get user inputs
    (
        question,
        top_k,
        document_id,
        max_distance
    ) = get_search_inputs()

    # Create question embedding
    question_embedding = create_question_embedding(
        client,
        question,
        EMBEDDING_MODEL
    )

    print(
        "\nQuestion embedding dimension:",
        len(question_embedding)
    )

    # Build metadata filter
    where_filter = build_metadata_filter(
        document_id
    )

    # Search ChromaDB
    results = search_documents(
        collection,
        question_embedding,
        top_k,
        where_filter
    )

    # Display results
    display_results(
        results,
        max_distance
    )

# Program entry point


if __name__ == "__main__":
    main()