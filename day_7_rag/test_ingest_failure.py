from ingest import index_embeddings


class FakeCollection:

    def upsert(
        self,
        ids,
        embeddings,
        documents,
        metadatas
    ):

        document_id = metadatas[0]["document_id"]

        if document_id == "DOC002":

            raise Exception(
                "Simulated indexing failure"
            )

        print(
            f"Fake collection indexed {document_id}"
        )


def test_failed_document_does_not_stop_processing():

    records = [
        {
            "chunk_id": "DOC001_CHUNK001",
            "document_id": "DOC001",
            "title": "Document 1",
            "source_path": "doc1.md",
            "updated_at": 1,
            "chunk_index": 0,
            "embedding_model": "test-model",
            "embedding": [0.1, 0.2],
            "text": "Document 1 text"
        },
        {
            "chunk_id": "DOC002_CHUNK001",
            "document_id": "DOC002",
            "title": "Document 2",
            "source_path": "doc2.md",
            "updated_at": 1,
            "chunk_index": 0,
            "embedding_model": "test-model",
            "embedding": [0.3, 0.4],
            "text": "Document 2 text"
        },
        {
            "chunk_id": "DOC003_CHUNK001",
            "document_id": "DOC003",
            "title": "Document 3",
            "source_path": "doc3.md",
            "updated_at": 1,
            "chunk_index": 0,
            "embedding_model": "test-model",
            "embedding": [0.5, 0.6],
            "text": "Document 3 text"
        }
    ]

    collection = FakeCollection()

    indexed_count, failed_documents = index_embeddings(
        collection,
        records
    )

    assert indexed_count == 2

    assert failed_documents == [
        "DOC002"
    ]

    print(
        "Failure-handling test passed."
    )


if __name__ == "__main__":
    test_failed_document_does_not_stop_processing()