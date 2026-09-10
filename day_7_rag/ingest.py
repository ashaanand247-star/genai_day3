import json
import logging
from pathlib import Path

import chromadb


# Configuration

EMBEDDINGS_FILE = Path(
    "day_6_Vector_search/embeddings.jsonl"
)

CHROMA_DIR = "day_7_rag/chroma_db"

COLLECTION_NAME = "employee_documents"

APPROVED_DOCUMENTS = {
    f"DOC{i:03d}"
    for i in range(1, 31)
}


# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# Function 1: Load existing embeddings

def load_embeddings(file_path):

    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            records.append(
                json.loads(line)
            )

    return records


# Function 2: Select approved documents

def select_approved_documents(
    records,
    approved_documents
):

    selected_records = []

    for record in records:

        if record["document_id"] in approved_documents:

            selected_records.append(
                record
            )

    return selected_records


# Function 3: Create ChromaDB client

def create_chroma_client(chroma_dir):

    client = chromadb.PersistentClient(
        path=chroma_dir
    )

    return client


# Function 4: Create ChromaDB collection

def create_chroma_collection(
    client,
    collection_name
):

    collection = client.get_or_create_collection(
        name=collection_name
    )

    return collection


# Function 5: Index embeddings

def index_embeddings(
    collection,
    records
):

    indexed_count = 0
    failed_documents = []

    documents_by_id = {}

    for record in records:

        document_id = record["document_id"]

        if document_id not in documents_by_id:

            documents_by_id[document_id] = []

        documents_by_id[document_id].append(
            record
        )

    for document_id, document_records in documents_by_id.items():

        try:

            ids = [
                record["chunk_id"]
                for record in document_records
            ]

            embeddings = [
                record["embedding"]
                for record in document_records
            ]

            documents = [
                record["text"]
                for record in document_records
            ]

            metadatas = [
                {
                    "document_id": record["document_id"],
                    "title": record["title"],
                    "source_path": record["source_path"],
                    "updated_at": record["updated_at"],
                    "chunk_index": record["chunk_index"],
                    "embedding_model": record["embedding_model"]
                }
                for record in document_records
            ]

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            indexed_count += len(ids)

            logging.info(
                f"Indexed {document_id}: "
                f"{len(ids)} chunks"
            )

        except Exception as error:

            failed_documents.append(
                document_id
            )

            logging.error(
                f"Failed to index {document_id}: "
                f"{error}"
            )

            continue

    return indexed_count, failed_documents


# Reusable ingestion flow for FastAPI

def ingest_file(file_reference):

    embedding_records = load_embeddings(
        EMBEDDINGS_FILE
    )

    selected_records = select_approved_documents(
        embedding_records,
        APPROVED_DOCUMENTS
    )

    selected_documents = {
        record["document_id"]
        for record in selected_records
    }

    chroma_client = create_chroma_client(
        CHROMA_DIR
    )

    collection = create_chroma_collection(
        chroma_client,
        COLLECTION_NAME
    )

    indexed_count, failed_documents = index_embeddings(
        collection,
        selected_records
    )

    return {
        "document_id": list(selected_documents),
        "chunk_count": indexed_count,
        "status": (
            "processed"
            if not failed_documents
            else "partially_processed"
        )
    }


# Main ingestion flow

def main():

    embedding_records = load_embeddings(
        EMBEDDINGS_FILE
    )

    print(
        "Loaded embeddings:",
        len(embedding_records)
    )

    selected_records = select_approved_documents(
        embedding_records,
        APPROVED_DOCUMENTS
    )

    selected_documents = {
        record["document_id"]
        for record in selected_records
    }

    print(
        "Selected chunks:",
        len(selected_records)
    )

    print(
        "Selected documents:",
        len(selected_documents)
    )

    chroma_client = create_chroma_client(
        CHROMA_DIR
    )

    print(
        "ChromaDB client created successfully"
    )

    collection = create_chroma_collection(
        chroma_client,
        COLLECTION_NAME
    )

    print(
        "ChromaDB collection:",
        collection.name
    )

    indexed_count, failed_documents = index_embeddings(
        collection,
        selected_records
    )

    print(
        "Indexed vectors:",
        indexed_count
    )

    print(
        "Total vectors in collection:",
        collection.count()
    )

    if failed_documents:

        print(
            "Failed documents:",
            failed_documents
        )

    else:

        print(
            "Failed documents: None"
        )


# Program entry point

if __name__ == "__main__":

    main()