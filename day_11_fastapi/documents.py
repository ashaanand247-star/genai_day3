import chromadb


# Configuration

CHROMA_DIR = "day_10_retrieval/chroma_db_chunk70"

COLLECTION_NAME = "employee_documents"


# Create ChromaDB client

def create_chroma_client():

    return chromadb.PersistentClient(
        path=CHROMA_DIR
    )


# Get ChromaDB collection

def get_collection():

    client = create_chroma_client()

    return client.get_collection(
        name=COLLECTION_NAME
    )


# Get document metadata

def get_document(document_id):

    collection = get_collection()

    results = collection.get(
        where={
            "document_id": document_id
        },
        include=[
            "metadatas"
        ]
    )

    metadatas = results["metadatas"]

    if not metadatas:

        return None

    first_metadata = metadatas[0]

    return {
        "document_id": document_id,
        "title": first_metadata.get(
            "title",
            "Unknown"
        ),
        "source_path": first_metadata.get(
            "source_path",
            "Unknown"
        ),
        "updated_at": first_metadata.get(
            "updated_at",
            "Unknown"
        ),
        "chunk_count": len(metadatas),
        "status": "processed"
    }