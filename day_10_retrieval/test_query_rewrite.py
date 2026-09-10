import os
import sys

sys.path.append(".")

from dotenv import load_dotenv
from openai import OpenAI

from day_7_rag.retrieve import (
    create_chroma_client,
    get_collection,
    retrieve_chunks
)


# Load environment variables

load_dotenv()


# Configuration

GENERATION_MODEL = "openrouter/free"

CHROMA_DIR = "day_7_rag/chroma_db"

COLLECTION_NAME = "employee_documents"

QUESTION = (
    "What should employees do if they become unavailable "
    "while working remotely?"
)


# Create OpenRouter client

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# Function 1: Generate rewritten query

def rewrite_query(question):

    prompt = f"""
Rewrite the user's question into a clearer,
more specific question for retrieving information
from an employee policy document.

Keep the original meaning.

Use terminology that is likely to appear
in the policy.

Return only the rewritten question.
Do not provide an answer.

User question:
{question}

Rewritten question:
"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()


# Function 2: Display retrieval results

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


# Main experiment

def main():

    # Create ChromaDB client

    chroma_client = create_chroma_client(
        CHROMA_DIR
    )


    # Get ChromaDB collection

    collection = get_collection(
        chroma_client,
        COLLECTION_NAME
    )


    # Generate rewritten question automatically

    rewritten_question = rewrite_query(
        QUESTION
    )


    # Baseline retrieval using original question

    baseline_results = retrieve_chunks(
        collection,
        QUESTION,
        top_k=3
    )


    # Experiment retrieval using rewritten question

    experiment_results = retrieve_chunks(
        collection,
        rewritten_question,
        top_k=3
    )


    # Display experiment information

    print("DAY 10 - AUTOMATIC QUERY REWRITING EXPERIMENT")

    print("\nOriginal question:")
    print(QUESTION)

    print("\nAutomatically generated rewritten question:")
    print(rewritten_question)

    print("\nBaseline configuration:")
    print("Original query + top_k=3")

    print("\nExperiment configuration:")
    print("LLM rewritten query + top_k=3")


    # Display results

    display_results(
        "BASELINE RESULTS",
        baseline_results
    )

    display_results(
        "QUERY REWRITE RESULTS",
        experiment_results
    )


if __name__ == "__main__":
    main()