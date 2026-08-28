import json
from pathlib import Path

import chromadb

# Configuration

EMBEDDINGS_FILE = Path(
    "day_6_Vector_search/embeddings.jsonl"
)

SELECTED_DOCUMENTS = {
    "DOC001",
    "DOC002",
    "DOC003",
    "DOC004",
    "DOC005",
}

CHROMA_DB_PATH = "day_6_Vector_search/chroma_db"
COLLECTION_NAME = "employee_documents"

# Function 1: Load embedding records


def load_embedding_records(file_path):
    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        records = [
            json.loads(line)
            for line in file
        ]

    return records

# Function 2: Select POC documents


def select_documents(records, selected_documents):
    selected_records = [
        record
        for record in records
        if record["document_id"] in selected_documents
    ]

    return selected_records

# Function 3: Create / access ChromaDB collection


def create_collection(db_path, collection_name):
    client = chromadb.PersistentClient(
        path=db_path
    )

    collection = client.get_or_create_collection(
        name=collection_name
    )

    return collection

# Function 4: Prepare data for ChromaDB


def prepare_chroma_data(selected_records):

    ids = [
        record["chunk_id"]
        for record in selected_records
    ]

    embeddings = [
        record["embedding"]
        for record in selected_records
    ]

    documents = [
        record["text"]
        for record in selected_records
    ]

    metadatas = [
        {
            "document_id": record["document_id"],
            "title": record["title"],
            "source_path": record["source_path"],
            "updated_at": record["updated_at"],
            "chunk_index": record["chunk_index"],
            "embedding_model": record["embedding_model"],
        }
        for record in selected_records
    ]

    return ids, embeddings, documents, metadatas

# Function 5: Insert vectors into ChromaDB


def upsert_vectors(
    collection,
    ids,
    embeddings,
    documents,
    metadatas
):

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

# Function 6: Main program


def main():

    # Load existing embeddings
    records = load_embedding_records(
        EMBEDDINGS_FILE
    )

    print(
        "Total embedding records:",
        len(records)
    )

    # Select only the five POC documents
    selected_records = select_documents(
        records,
        SELECTED_DOCUMENTS
    )

    print(
        "Selected chunks:",
        len(selected_records)
    )

    # Create / access ChromaDB
    collection = create_collection(
        CHROMA_DB_PATH,
        COLLECTION_NAME
    )

    print(
        "ChromaDB collection:",
        collection.name
    )

    # Prepare data
    (
        ids,
        embeddings,
        documents,
        metadatas
    ) = prepare_chroma_data(
        selected_records
    )

    # Insert vectors + metadata
    upsert_vectors(
        collection,
        ids,
        embeddings,
        documents,
        metadatas
    )

    print(
        "Injected vectors:",
        len(ids)
    )

    print(
        "Total vectors in collection:",
        collection.count()
    )

# Program entry point


if __name__ == "__main__":
    main()