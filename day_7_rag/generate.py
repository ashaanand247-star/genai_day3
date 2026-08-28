import os
import sys

sys.path.append(".")

from dotenv import load_dotenv
from openai import OpenAI

from retrieve import (
    create_chroma_client,
    get_collection,
    retrieve_chunks,
    prepare_context
)

from day_8.citations import (
    extract_citations,
    validate_citations,
    build_allowed_citations
)

from day_8.models import AnswerResponse


# Load environment variables

load_dotenv()


# Configuration

GENERATION_MODEL = "openrouter/free"

CHROMA_DIR = "day_7_rag/chroma_db"

COLLECTION_NAME = "employee_documents"

EVIDENCE_THRESHOLD = 1.0


# Create OpenRouter client

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# Function 1: Build the prompt

def build_prompt(
    question,
    context
):

    prompt = f"""
You are an employee policy assistant.

Answer the user's question using only the
provided context.

Do not use outside knowledge.
Do not make unsupported assumptions.

For every factual statement, cite the source using
this exact format:

[Source: <title> | Document: <document_id> | Chunk: <chunk_index>]

Do not use Markdown such as **Source:**.
Do not change the citation format.

If the provided context does not contain enough
information to answer the question, say:

"The provided context does not contain enough
information to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


# Function 2: Generate answer

def generate_answer(
    question,
    context
):

    prompt = build_prompt(
        question,
        context
    )

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# Main RAG flow

def main():

    # Get user's question

    question = input(
        "Enter your question: "
    )


    # Create ChromaDB client

    chroma_client = create_chroma_client(
        CHROMA_DIR
    )


    # Get ChromaDB collection

    collection = get_collection(
        chroma_client,
        COLLECTION_NAME
    )


    # Retrieve relevant chunks

    results = retrieve_chunks(
        collection,
        question,
        top_k=3
    )


    # Build allowed citations from retrieved chunks

    allowed_citations = build_allowed_citations(
        results
    )


    # Prepare context

    context = prepare_context(
        results,
        max_chunks=3
    )


    # Display prepared context

    print("\nPrepared context:\n")

    print(
        context
    )


    # Generate final answer

    answer = generate_answer(
        question,
        context
    )


    # Extract citations from the answer

    citations = extract_citations(
        answer
    )


    # Validate citations

    valid_citations = validate_citations(
        citations,
        allowed_citations
    )


    # Get retrieved chunk previews

    chunk_previews = results["documents"][0]


    # Get retrieval scores

    retrieval_scores = results["distances"][0]


    # Check whether any retrieved chunk
    # passes the evidence threshold

    has_evidence = any(
        score <= EVIDENCE_THRESHOLD
        for score in retrieval_scores
    )


    # Determine answer status

    abstention_message = (
        "The provided context does not contain enough"
        " information to answer this question."
    )


    if not has_evidence:

        answer = abstention_message

        valid_citations = []

        status = "insufficient_evidence"


    elif abstention_message in answer:

        valid_citations = []

        status = "insufficient_evidence"


    elif len(citations) == 0:

        status = "insufficient_evidence"


    elif len(valid_citations) == 0:

        status = "insufficient_evidence"


    else:

        status = "answered"


    # Create structured response

    response = AnswerResponse(
        answer=answer,
        sources=valid_citations,
        chunk_previews=chunk_previews,
        retrieval_scores=retrieval_scores,
        status=status
    )


    # Display structured response

    print("\nStructured response:\n")

    print(
        response
    )


    # Display extracted citations

    print("\nExtracted citations:")

    print(
        citations
    )


    # Display valid citations

    print("\nValid citations:")

    print(
        valid_citations
    )


    # Display answer

    print("\nGenerated answer:\n")

    print(
        answer
    )


# Program entry point

if __name__ == "__main__":

    main()