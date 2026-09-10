import sys
import re
import math

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

# Baseline configuration
BASELINE_TOP_K = 3

# Reranking configuration
CANDIDATE_TOP_K = 5
FINAL_TOP_K = 3


def tokenize(text):
    """
    Convert text into simple lowercase word tokens.
    """
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def calculate_tfidf_scores(question, documents):
    """
    Calculate a simple TF-IDF cosine similarity between
    the question and each candidate document.

    This is used only for reranking the candidates
    already retrieved by vector search.
    """

    question_tokens = tokenize(question)

    document_tokens = [
        tokenize(document)
        for document in documents
    ]

    all_documents = [question_tokens] + document_tokens

    vocabulary = set()

    for tokens in all_documents:
        vocabulary.update(tokens)

    vocabulary = list(vocabulary)

    # Document frequency
    document_frequency = {}

    for term in vocabulary:
        count = 0

        for tokens in all_documents:
            if term in tokens:
                count += 1

        document_frequency[term] = count

    total_documents = len(all_documents)

    def tfidf_vector(tokens):
        term_counts = {}

        for token in tokens:
            term_counts[token] = (
                term_counts.get(token, 0) + 1
            )

        vector = {}

        for term in vocabulary:
            tf = term_counts.get(term, 0)

            if tf == 0:
                vector[term] = 0.0
                continue

            df = document_frequency[term]

            idf = math.log(
                (total_documents + 1) / (df + 1)
            ) + 1

            vector[term] = tf * idf

        return vector

    question_vector = tfidf_vector(question_tokens)

    document_vectors = [
        tfidf_vector(tokens)
        for tokens in document_tokens
    ]

    def cosine_similarity(vector_a, vector_b):
        dot_product = sum(
            vector_a[term] * vector_b[term]
            for term in vocabulary
        )

        magnitude_a = math.sqrt(
            sum(
                value * value
                for value in vector_a.values()
            )
        )

        magnitude_b = math.sqrt(
            sum(
                value * value
                for value in vector_b.values()
            )
        )

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (
            magnitude_a * magnitude_b
        )

    scores = [
        cosine_similarity(
            question_vector,
            document_vector
        )
        for document_vector in document_vectors
    ]

    return scores


def display_baseline(results):
    print("\n" + "=" * 70)
    print("BASELINE VECTOR RETRIEVAL")
    print("=" * 70)

    for rank, document in enumerate(
        results["documents"][0],
        start=1
    ):
        metadata = results["metadatas"][0][rank - 1]
        distance = results["distances"][0][rank - 1]

        print(f"\nOriginal Rank: {rank}")
        print(f"Document: {metadata['document_id']}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(f"Original Distance: {distance}")
        print(f"Text: {document}")


def display_reranked(results, reranked_results):
    print("\n" + "=" * 70)
    print("RERANKED RESULTS")
    print("=" * 70)

    for final_rank, item in enumerate(
        reranked_results,
        start=1
    ):
        metadata = item["metadata"]

        print(f"\nFinal Rank: {final_rank}")
        print(f"Original Rank: {item['original_rank']}")
        print(f"Document: {metadata['document_id']}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(
            f"Original Vector Distance: "
            f"{item['original_distance']}"
        )
        print(
            f"Reranking Score: "
            f"{item['reranking_score']:.6f}"
        )
        print(f"Text: {item['document']}")


def main():
    print("DAY 10 - RERANKING EXPERIMENT")

    print("\nQuestion:")
    print(QUESTION)

    chroma_client = create_chroma_client(
        CHROMA_DIR
    )

    collection = get_collection(
        chroma_client,
        COLLECTION_NAME
    )

    # -------------------------------------------------
    # 1. Baseline retrieval
    # -------------------------------------------------

    baseline_results = retrieve_chunks(
        collection,
        QUESTION,
        top_k=BASELINE_TOP_K
    )

    # -------------------------------------------------
    # 2. Retrieve a larger candidate set
    # -------------------------------------------------

    candidate_results = retrieve_chunks(
        collection,
        QUESTION,
        top_k=CANDIDATE_TOP_K
    )

    documents = candidate_results["documents"][0]
    distances = candidate_results["distances"][0]
    metadatas = candidate_results["metadatas"][0]

    # -------------------------------------------------
    # 3. Calculate reranking scores
    # -------------------------------------------------

    reranking_scores = calculate_tfidf_scores(
        QUESTION,
        documents
    )

    # -------------------------------------------------
    # 4. Preserve original retrieval information
    # -------------------------------------------------

    candidates = []

    for index, document in enumerate(documents):
        candidates.append(
            {
                "document": document,
                "metadata": metadatas[index],
                "original_rank": index + 1,
                "original_distance": distances[index],
                "reranking_score": reranking_scores[index]
            }
        )

    # -------------------------------------------------
    # 5. Rerank candidates
    # -------------------------------------------------

    reranked_results = sorted(
        candidates,
        key=lambda item: item["reranking_score"],
        reverse=True
    )

    # -------------------------------------------------
    # 6. Select final top K
    # -------------------------------------------------

    reranked_results = reranked_results[
        :FINAL_TOP_K
    ]

    # -------------------------------------------------
    # 7. Display experiment configuration
    # -------------------------------------------------

    print("\nBaseline configuration:")
    print(
        f"Vector retrieval top_k = "
        f"{BASELINE_TOP_K}"
    )

    print("\nReranking configuration:")
    print(
        f"Vector candidate pool = "
        f"{CANDIDATE_TOP_K}"
    )

    print(
        f"Final selected chunks = "
        f"{FINAL_TOP_K}"
    )

    print(
        "\nReranking method: "
        "TF-IDF cosine similarity"
    )

    # -------------------------------------------------
    # 8. Display results
    # -------------------------------------------------

    display_baseline(
        baseline_results
    )

    display_reranked(
        candidate_results,
        reranked_results
    )


if __name__ == "__main__":
    main()