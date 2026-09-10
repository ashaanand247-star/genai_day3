import json
import logging
from pathlib import Path

import chromadb


# ============================================================
# Configuration
# ============================================================

EMBEDDINGS_FILE = Path(
    "day_10_retrieval/chunk70_embeddings.jsonl"
)

SOURCE_DIR = Path(
    "day5_preprocessing/cleaned_documents"
)

CHROMA_DIR = "day_10_retrieval/chroma_db_chunk70"

COLLECTION_NAME = "employee_documents"

CHUNK_SIZE = 70
OVERLAP = 20

APPROVED_DOCUMENTS = {
    f"DOC{i:03d}"
    for i in range(1, 31)
}


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# ============================================================
# Function 1: Load cached embeddings
# ============================================================

def load_embeddings(file_path):

    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if line.strip():

                records.append(
                    json.loads(line)
                )

    return records


# ============================================================
# Function 2: Chunk text
# Same logic used in test_chunk_size_full.py
# ============================================================

def chunk_text(
    text,
    chunk_size,
    overlap
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        start = end - overlap

    return chunks


# ============================================================
# Function 3: Reconstruct chunk text
# ============================================================

def prepare_chunks():

    all_chunks = []

    files = [
        filename
        for filename in SOURCE_DIR.iterdir()
        if filename.suffix == ".md"
    ]

    files.sort()

    print(
        "Documents found:",
        len(files)
    )

    for source_path in files:

        filename = source_path.name

        document_id = filename.split("_")[0]

        if document_id not in APPROVED_DOCUMENTS:

            continue

        with open(
            source_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        chunks = chunk_text(
            text,
            CHUNK_SIZE,
            OVERLAP
        )

        print(
            f"{document_id}: "
            f"{len(chunks)} chunks"
        )

        title = filename

        for index, chunk in enumerate(chunks):

            chunk_id = (
                f"{document_id}_chunk70_{index}"
            )

            all_chunks.append(
                {
                    "id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": index,
                    "text": chunk,
                    "title": title,
                    "source_path": str(source_path)
                }
            )

    return all_chunks


# ============================================================
# Function 4: Create ChromaDB client
# ============================================================

def create_chroma_client(chroma_dir):

    return chromadb.PersistentClient(
        path=chroma_dir
    )


# ============================================================
# Function 5: Create clean collection
# ============================================================

def create_chroma_collection(
    client,
    collection_name
):

    # Remove an existing Day 10 collection
    # so the index is rebuilt cleanly.
    try:

        client.delete_collection(
            name=collection_name
        )

        print(
            "Existing Day 10 collection removed."
        )

    except Exception:

        pass

    collection = client.create_collection(
        name=collection_name
    )

    return collection


# ============================================================
# Function 6: Build final index
# ============================================================

def index_embeddings(
    collection,
    chunks,
    embedding_records
):

    embeddings_by_id = {
        record["id"]: record["embedding"]
        for record in embedding_records
    }

    missing_embeddings = []

    for chunk in chunks:

        if chunk["id"] not in embeddings_by_id:

            missing_embeddings.append(
                chunk["id"]
            )

    if missing_embeddings:

        raise ValueError(
            "Missing embeddings for "
            f"{len(missing_embeddings)} chunks."
        )

    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = [
        embeddings_by_id[chunk["id"]]
        for chunk in chunks
    ]

    metadatas = [
        {
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
            "title": chunk["title"],
            "source_path": chunk["source_path"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DAY 10 - FINAL CHUNK SIZE 70 INDEX"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load existing embeddings
    # --------------------------------------------------------

    embedding_records = load_embeddings(
        EMBEDDINGS_FILE
    )

    print(
        "\nLoaded cached embeddings:",
        len(embedding_records)
    )

    # --------------------------------------------------------
    # Reconstruct exact chunk-70 text
    # --------------------------------------------------------

    chunks = prepare_chunks()

    print(
        "\nReconstructed chunks:",
        len(chunks)
    )

    # --------------------------------------------------------
    # Verify counts match
    # --------------------------------------------------------

    if len(chunks) != len(embedding_records):

        raise ValueError(
            "\nChunk count does not match "
            "embedding count.\n"
            f"Chunks: {len(chunks)}\n"
            f"Embeddings: {len(embedding_records)}"
        )

    print(
        "Chunk count matches embedding count."
    )

    # --------------------------------------------------------
    # Verify every chunk ID has an embedding
    # --------------------------------------------------------

    embedding_ids = {
        record["id"]
        for record in embedding_records
    }

    chunk_ids = {
        chunk["id"]
        for chunk in chunks
    }

    missing_embeddings = chunk_ids - embedding_ids

    extra_embeddings = embedding_ids - chunk_ids

    if missing_embeddings:

        raise ValueError(
            "Missing embedding IDs: "
            f"{list(missing_embeddings)[:5]}"
        )

    if extra_embeddings:

        raise ValueError(
            "Unexpected embedding IDs: "
            f"{list(extra_embeddings)[:5]}"
        )

    print(
        "All chunk IDs match cached embeddings."
    )

    # --------------------------------------------------------
    # Create separate Day 10 ChromaDB
    # --------------------------------------------------------

    chroma_client = create_chroma_client(
        CHROMA_DIR
    )

    print(
        "\nChromaDB client created successfully."
    )

    collection = create_chroma_collection(
        chroma_client,
        COLLECTION_NAME
    )

    print(
        "ChromaDB collection:",
        collection.name
    )

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    indexed_count = index_embeddings(
        collection,
        chunks,
        embedding_records
    )

    print(
        "\nIndexed vectors:",
        indexed_count
    )

    print(
        "Total vectors in collection:",
        collection.count()
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if collection.count() == len(chunks):

        print(
            "\nSUCCESS: Day 10 chunk_size=70 "
            "index created successfully."
        )

        print(
            "Baseline index was NOT modified."
        )

    else:

        raise RuntimeError(
            "\nIndex validation failed."
        )


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":

    main()