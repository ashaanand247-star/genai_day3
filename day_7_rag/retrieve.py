import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables

load_dotenv()


# Configuration

CHROMA_DIR = "day_7_rag/chroma_db"

COLLECTION_NAME = "employee_documents"

EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

DISTANCE_THRESHOLD = 1.2


# Create OpenRouter client

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# Function 1: Create ChromaDB client

def create_chroma_client(chroma_dir):

    return chromadb.PersistentClient(
        path=chroma_dir
    )


# Function 2: Get ChromaDB collection

def get_collection(client, collection_name):

    return client.get_collection(
        name=collection_name
    )


# Function 3: Generate question embedding

def create_query_embedding(question):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
        encoding_format="float"
    )

    return response.data[0].embedding


# Function 4: Retrieve relevant chunks

def retrieve_chunks(
    collection,
    question,
    top_k=3,
    where=None,
    distance_threshold=None
):

    query_embedding = create_query_embedding(
        question
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where
    )

    if distance_threshold is not None:

        filtered_results = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        for index, distance in enumerate(
            results["distances"][0]
        ):

            if distance <= distance_threshold:

                filtered_results["documents"][0].append(
                    results["documents"][0][index]
                )

                filtered_results["metadatas"][0].append(
                    results["metadatas"][0][index]
                )

                filtered_results["distances"][0].append(
                    distance
                )

        results = filtered_results

    return results


# Function 5: Prepare context for generation

def prepare_context(results, max_chunks=3):

    context_parts = []
    seen_chunks = set()

    for index, document in enumerate(
        results["documents"][0][:max_chunks]
    ):

        metadata = results["metadatas"][0][index]

        chunk_key = (
            metadata["document_id"],
            metadata["chunk_index"]
        )

        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(chunk_key)

        source_label = (
            f"[Source: {metadata['title']} "
            f"| Document: {metadata['document_id']} "
            f"| Chunk: {metadata['chunk_index']}]"
        )

        context_parts.append(
            source_label
            + "\n"
            + document
        )

    return "\n\n".join(context_parts)


# Main retrieval flow

def main():

    question = input(
        "Enter your question: "
    )

    chroma_client = create_chroma_client(
        CHROMA_DIR
    )

    collection = get_collection(
        chroma_client,
        COLLECTION_NAME
    )

    results = retrieve_chunks(
        collection,
        question,
        top_k=3,
        distance_threshold=DISTANCE_THRESHOLD
    )

    context = prepare_context(
        results,
        max_chunks=3
    )

    print("\nPrepared context:\n")

    print(context)


# Program entry point

if __name__ == "__main__":
    main()