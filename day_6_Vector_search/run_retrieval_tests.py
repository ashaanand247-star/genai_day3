import os
import json

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDING_MODEL = (
    "nvidia/llama-nemotron-embed-vl-1b-v2:free"
)

TOP_K = 3

TEST_FILE = (
    "day_6_Vector_search/retrieval_test_set.json"
)

REPORT_FILE = (
    "day_6_Vector_search/retrieval_result_report.json"
)

CHROMA_DB_PATH = (
    "day_6_Vector_search/chroma_db"
)

COLLECTION_NAME = "employee_documents"


# --------------------------------------------------
# Function 1: Create OpenRouter client
# --------------------------------------------------

def create_embedding_client():

    load_dotenv()

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    return client


# --------------------------------------------------
# Function 2: Connect to ChromaDB
# --------------------------------------------------

def get_collection():

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# --------------------------------------------------
# Function 3: Load retrieval test cases
# --------------------------------------------------

def load_test_cases(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        test_cases = json.load(file)

    return test_cases


# --------------------------------------------------
# Function 4: Create question embedding
# --------------------------------------------------

def create_question_embedding(
    client,
    question
):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
        encoding_format="float"
    )

    question_embedding = (
        response.data[0].embedding
    )

    return question_embedding


# --------------------------------------------------
# Function 5: Search ChromaDB
# --------------------------------------------------

def search_documents(
    collection,
    question_embedding
):

    search_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K
    )

    return search_results


# --------------------------------------------------
# Function 6: Get returned document IDs
# --------------------------------------------------

def get_returned_document_ids(
    search_results
):

    returned_document_ids = [
        metadata["document_id"]
        for metadata in search_results["metadatas"][0]
    ]

    return returned_document_ids


# --------------------------------------------------
# Function 7: Check whether test passed
# --------------------------------------------------

def check_test_result(
    expected_document_id,
    returned_document_ids
):

    passed = (
        expected_document_id
        in returned_document_ids
    )

    return passed


# --------------------------------------------------
# Function 8: Build detailed retrieval results
# --------------------------------------------------

def build_retrieval_results(
    search_results
):

    retrieval_results = []

    ids = search_results["ids"][0]
    distances = search_results["distances"][0]
    documents = search_results["documents"][0]
    metadatas = search_results["metadatas"][0]

    for index in range(len(ids)):

        retrieval_results.append({
            "chunk_id": ids[index],
            "distance": distances[index],
            "document_id": metadatas[index]["document_id"],
            "title": metadatas[index]["title"],
            "source_path": metadatas[index]["source_path"],
            "updated_at": metadatas[index]["updated_at"],
            "chunk_index": metadatas[index]["chunk_index"],
            "embedding_model": metadatas[index]["embedding_model"],
            "text": documents[index]
        })

    return retrieval_results


# --------------------------------------------------
# Function 9: Run all retrieval tests
# --------------------------------------------------

def run_tests(
    client,
    collection,
    test_cases
):

    results = []

    for number, test_case in enumerate(
        test_cases,
        start=1
    ):

        question = test_case["question"]

        expected_document_id = (
            test_case["expected_document_id"]
        )

        # Create question embedding

        question_embedding = (
            create_question_embedding(
                client,
                question
            )
        )

        # Search ChromaDB

        search_results = search_documents(
            collection,
            question_embedding
        )

        # Get returned document IDs

        returned_document_ids = (
            get_returned_document_ids(
                search_results
            )
        )

        # Check expected document

        passed = check_test_result(
            expected_document_id,
            returned_document_ids
        )

        # Build detailed retrieval information

        detailed_results = (
            build_retrieval_results(
                search_results
            )
        )

        # Store complete test result

        results.append({
            "test_number": number,
            "question": question,
            "expected_document_id":
                expected_document_id,
            "top_k": TOP_K,
            "results": detailed_results,
            "passed": passed
        })

        # Display result

        print(f"\nTest {number}")
        print("Question:", question)
        print(
            "Expected:",
            expected_document_id
        )
        print(
            "Top 3:",
            returned_document_ids
        )
        print(
            "Result:",
            "PASS" if passed else "FAIL"
        )

    return results


# --------------------------------------------------
# Function 10: Save retrieval result report
# --------------------------------------------------

def save_report(
    results,
    file_path
):

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        "\nRetrieval report saved to:",
        file_path
    )


# --------------------------------------------------
# Function 11: Display test summary
# --------------------------------------------------

def display_summary(results):

    passed_count = sum(
        result["passed"]
        for result in results
    )

    failed_count = (
        len(results) - passed_count
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST SUMMARY")
    print("=" * 60)

    print(
        "Total tests:",
        len(results)
    )

    print(
        "Passed:",
        passed_count
    )

    print(
        "Failed:",
        failed_count
    )


# --------------------------------------------------
# Function 12: Main program
# --------------------------------------------------

def main():

    # Create OpenRouter client

    client = create_embedding_client()

    # Connect to ChromaDB

    collection = get_collection()

    # Load test cases

    test_cases = load_test_cases(
        TEST_FILE
    )

    # Run retrieval tests

    results = run_tests(
        client,
        collection,
        test_cases
    )

    # Save detailed retrieval report

    save_report(
        results,
        REPORT_FILE
    )

    # Display summary

    display_summary(results)


# --------------------------------------------------
# Program entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()