from fastapi.testclient import TestClient

from day_11_fastapi.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invalid_question():
    response = client.post(
        "/ask",
        json={
            "question": "",
            "top_k": 3
        }
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"]["category"] == "validation_error"
    assert data["request_id"] != ""


def test_insufficient_evidence():
    response = client.post(
        "/ask",
        json={
            "question": "What is the employee policy for underwater moon travel?",
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "insufficient_evidence"
    assert data["request_id"] != ""


def test_provider_failure(monkeypatch):
    from openai import APIError

    def fake_answer_question(
        question,
        top_k=3,
        filters=None
    ):
        raise APIError(
            "Simulated provider failure",
            request=None,
            body=None
        )

    monkeypatch.setattr(
        "day_11_fastapi.routes.answer_question",
        fake_answer_question
    )

    response = client.post(
        "/ask",
        json={
            "question": "What is the leave policy?",
            "top_k": 3
        }
    )

    assert response.status_code == 502

    data = response.json()

    assert data["error"]["category"] == "provider_failure"
    assert data["error"]["message"] == "Upstream AI provider failed"
    assert data["request_id"] != ""

def test_successful_asking(monkeypatch):
    from day_8.models import AnswerResponse

    def fake_answer_question(
        question,
        top_k=3,
        filters=None
    ):
        return AnswerResponse(
            answer="Employees can request annual leave.",
            sources=[
                "[Source: Employee Leave Policy | Document: DOC001 | Chunk: 0]"
            ],
            retrieved_sources=[
                "DOC001:chunk_0"
            ],
            chunk_previews=[
                "Employee Leave Policy"
            ],
            retrieval_scores=[
                0.5
            ],
            status="answered"
        )

    monkeypatch.setattr(
        "day_11_fastapi.routes.answer_question",
        fake_answer_question
    )

    response = client.post(
        "/ask",
        json={
            "question": "What is the leave policy?",
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "answered"
    assert data["answer"] == "Employees can request annual leave."
    assert data["request_id"] != ""
    assert len(data["retrieved_sources"]) == 1

def test_successful_ingestion(monkeypatch):

    def fake_ingest_file(file_reference):
        return {
            "document_id": ["DOC_TEST"],
            "chunk_count": 5,
            "status": "ingested"
        }

    monkeypatch.setattr(
        "day_11_fastapi.routes.ingest_file",
        fake_ingest_file
    )

    response = client.post(
        "/ingest",
        json={
            "file_reference": "test_document.md"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_id"] == ["DOC_TEST"]
    assert data["chunk_count"] == 5
    assert data["status"] == "ingested"
    assert data["request_id"] != ""

def test_unknown_document():
    response = client.get(
        "/documents/UNKNOWN_DOC"
    )

    assert response.status_code == 404

    data = response.json()

    assert data["error"]["category"] == "document_not_found"
    assert data["error"]["message"] == "Document not found"
    assert data["request_id"] != ""