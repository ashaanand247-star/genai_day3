# GenAI Day 4 - Prompt Testing and Validation

## Project Overview

This project extends the Day 3 Prompt Engineering project into a testable and measurable GenAI component.

The application uses the OpenRouter API with Python to generate AI responses and validates the generated output using Pydantic models.

The project was progressively extended from prompt engineering into document preprocessing, embeddings, vector search, Retrieval-Augmented Generation (RAG), grounded generation, citations, abstention, and retrieval optimization.

Day 4 focuses on:

* Structured outputs
* Response validation
* Fixed prompt test cases
* Prompt test automation
* Failure categorization
* Prompt version comparison
* Latency and token measurement

Day 5 focuses on:

* Document sanitization
* Document chunking
* Preparing documents for embeddings and retrieval

Day 6 focuses on:

* Embedding generation
* Converting document chunks into vector representations

Day 7 focuses on:

* Shared vector-store ingestion
* Semantic retrieval
* Context preparation
* RAG generation
* Integration and failure-handling tests

Day 8 focuses on:

* Grounded answer generation
* Structured responses
* Citation validation
* Evidence thresholds
* Abstention when evidence is insufficient

Day 9 focuses on:

* Retrieval evaluation
* Root-cause analysis of weak retrieval cases
* Controlled retrieval experiments
* `top_k` optimization
* Metadata filtering
* Query rewriting
* Chunk-size experimentation
* Measuring retrieval behavior

---

## Features

### Day 4 - Prompt Testing and Validation

* Prompt-based AI interaction
* Prompt templates stored separately from Python code
* Multiple AI task prompt templates:

  * Text Summarization
  * Text Classification
  * Information Extraction
* Secure API key management using `.env`
* Pydantic output models
* AI response validation
* Fixed 10-case prompt test dataset
* Automated prompt test runner
* Failure categorization
* Prompt version logging
* Model version logging
* API latency measurement
* Token usage measurement
* V1 and V2 prompt comparison
* Machine-readable JSON test results

### Day 5 - Document Preprocessing

* Processing of 30 Markdown employee/company policy documents
* Sensitive-data detection using regular expressions
* Sanitization of sensitive information
* Separate sanitized document collection
* Document chunking
* Structured chunk metadata
* Machine-readable `chunks.jsonl` dataset

Each chunk contains metadata including:

* `chunk_id`
* `document_id`
* `title`
* `source_path`
* `updated_at`
* `chunk_index`
* `text`

### Day 6 - Embeddings

* Embedding generation for document chunks
* Vector representation of chunk text
* Preservation of document and chunk metadata
* Preparation of embeddings for vector-store ingestion

### Day 7 - Baseline RAG

* Ingestion of the complete 30-document collection
* Shared ChromaDB vector store
* Semantic retrieval
* Configurable `top_k`
* Metadata filtering
* Distance threshold support
* Retrieved source metadata
* Stable source labels
* Context preparation
* LLM generation using retrieved evidence
* Integration testing
* Document-level failure handling

The baseline ingestion successfully loaded and indexed:

* 30 documents
* 134 chunks
* 134 embeddings
* 134 vectors in the shared `employee_documents` ChromaDB collection

No documents failed during the successful ingestion run.

### Day 8 - Grounded Generation

* Grounded answer prompt
* Structured answer response model
* Evidence-based response generation
* Citation mapping
* Citation validation
* Validation that citations refer only to supplied source chunks
* Evidence threshold
* Abstention when evidence is insufficient
* Fallback response instead of guessing
* Testing of answerable and unsupported questions

The system is designed so that when the retrieved evidence does not support an answer, it returns a clear fallback instead of inventing information.

### Day 9 - Retrieval Optimization

* Fixed 10-question retrieval evaluation set
* Hit Rate@3 measurement
* Mean Reciprocal Rank (MRR)
* Selection of five weak/problematic retrieval cases
* Root-cause hypotheses
* Controlled retrieval experiments
* `top_k` comparison
* Metadata filtering
* Query rewriting
* Chunk-size comparison
* Before/after retrieval analysis
* Retrieval-behavior documentation

---

## Day 5 - Document Preprocessing

Day 5 prepares the source documents for the RAG pipeline.

The project uses a collection of 30 Markdown employee/company policy documents.

### Document Sanitization

Sensitive information is detected using regular expressions and sanitized before the documents are used for retrieval.

The preprocessing stage handles sensitive values such as:

* Employee IDs
* Email addresses
* Percentages
* Monetary amounts

The sanitized documents are stored separately from the original documents.

### Document Chunking

The sanitized policy documents are divided into smaller chunks suitable for embedding and retrieval.

Each chunk contains document identity, chunk identity, source information, and the chunk text.

The resulting chunk dataset is stored as:

```text
day5_preprocessing/chunks.jsonl
```

This dataset becomes the input for the embedding stage.

---

## Day 6 - Embedding Generation

Day 6 converts the document chunks into vector representations.

The chunks generated during Day 5 are used as the source dataset for embedding generation.

The embedding process:

1. Reads the chunk dataset.
2. Extracts the chunk text.
3. Generates an embedding vector.
4. Preserves the original chunk metadata.
5. Produces data suitable for vector-store ingestion.

Embeddings allow the system to perform semantic retrieval rather than depending only on exact keyword matching.

---

## Day 7 - Baseline RAG

Day 7 builds the end-to-end Retrieval-Augmented Generation pipeline.

The complete collection of 30 policy documents is connected through a shared ChromaDB vector store.

This means a question can retrieve information from any document in the collection rather than being restricted to a single document.

### RAG Flow

```text
30 Policy Documents
        |
        v
Document Sanitization
        |
        v
Chunking
        |
        v
Embedding Generation
        |
        v
Shared ChromaDB Vector Store
        |
        v
User Question
        |
        v
Question Embedding
        |
        v
Semantic Retrieval
        |
        v
Relevant Chunks
        |
        v
Prepared Context
        |
        v
LLM Generation
        |
        v
Answer
```

### Retrieval

The retrieval module:

* Accepts a natural-language question.
* Creates an embedding using the same embedding model.
* Queries the shared `employee_documents` ChromaDB collection.
* Supports configurable `top_k`.
* Supports metadata filtering.
* Can apply a distance threshold.
* Returns retrieved documents, metadata, and distances.

The baseline retrieval configuration uses `top_k=3`.

### Testing

Day 7 includes:

* An integration test covering ingestion, retrieval, and context preparation.
* A failure-handling test verifying that one failed document does not stop the remaining documents from being processed.

---

## Day 8 - Grounded Generation, Citations and Abstention

Day 8 adds reliability controls to the generation stage.

### Grounded Generation

The generation prompt instructs the model to answer using only the retrieved context.

The model should not use unsupported information from its general knowledge.

### Structured Response

The generated answer is validated using a structured response model.

This provides a predictable response format and allows invalid model outputs to be detected.

### Citation Validation

The system validates citations against the source chunks supplied to the model.

Citation validation ensures that:

* A citation refers to a chunk actually supplied to the model.
* Invalid citation references are rejected or removed.
* The final answer does not claim support from unavailable source chunks.

### Abstention

The system abstains when the available evidence is insufficient.

Abstention occurs when:

* No retrieved chunk passes the evidence threshold, or
* The retrieved context does not support the requested answer.

Instead of guessing, the system returns a clear fallback response.

This establishes an important reliability principle:

```text
Evidence supports answer
        |
        v
     Generate

Evidence does not support answer
        |
        v
      Abstain
```

---

## Day 9 - Retrieval Evaluation and Optimization

Day 9 focuses on understanding and improving retrieval behavior.

Rather than assuming that a retrieval result is poor, the system uses measurable experiments to investigate possible causes.

### Baseline Retrieval Metrics

The fixed 10-question retrieval set produced:

* Hit Rate@3: **1.0**
* MRR: **1.0**

Therefore, the five selected questions were treated as weak/problematic cases for controlled experimentation rather than proven baseline failures.

### Five Controlled Experiments

#### 1. `top_k` - Working From Home

Question:

```text
What are the requirements for working from home?
```

Expected document:

```text
DOC002
```

The experiment compared `top_k=3` with `top_k=5`.

Increasing `top_k` from 3 to 5 retrieved additional chunks, including semantically related content from another document.

Conclusion:

* A larger `top_k` can increase context.
* Additional context can introduce cross-document content.
* More retrieved chunks do not automatically mean better evidence.

#### 2. Metadata Filtering - Remote Unavailability

Question:

```text
What should employees do if they become unavailable while working remotely?
```

Expected document:

```text
DOC002
```

The experiment compared unrestricted retrieval with retrieval filtered to `DOC002`.

Applying the metadata filter restricted the results to `DOC002`.

Conclusion:

* Metadata filtering can remove cross-document competition.
* Filtering can focus retrieval on the expected document.

#### 3. Query Rewriting - Unexpected Absence

Question:

```text
How should employees report an unexpected absence?
```

Expected document:

```text
DOC003
```

The policy used terminology such as "unplanned absence", while the question used "unexpected absence".

The rewritten query used terminology closer to the policy:

```text
How should employees report an unplanned absence by notifying their
manager or designated team contact?
```

After rewriting, `DOC003` was retrieved at rank 1.

Conclusion:

* Vocabulary mismatch can affect retrieval relevance.
* Query rewriting can improve retrieval when user terminology differs from source terminology.

#### 4. Chunk Size - Employee Onboarding Training

Question:

```text
What training is required during employee onboarding?
```

Expected document:

```text
DOC004
```

The experiment compared:

```text
chunk_size = 100
```

with:

```text
chunk_size = 150
overlap = 20
```

Results:

* Baseline: 5 chunks
* Experiment: 3 chunks
* Baseline top document: DOC004
* Experiment top document: DOC004

Both configurations retrieved `DOC004` at rank 1.

Conclusion:

* Increasing chunk size reduced the number of chunks.
* It did not improve the top-document ranking.
* Therefore, the experiment did not confirm an improvement from the larger chunk size.

#### 5. Metadata Filtering - Conflict of Interest

Question:

```text
What should employees do if they have a potential conflict of interest?
```

Expected document:

```text
DOC005
```

The experiment compared unrestricted retrieval with retrieval filtered to `DOC005`.

The metadata filter returned `DOC005` chunks, with the relevant "Conflicts of Interest" section appearing in the highest-ranked result.

Conclusion:

* Metadata filtering can focus retrieval on the expected document.
* Filtering can remove cross-document competition.

### Day 9 Experimental Variables

The five experiments changed one primary retrieval variable at a time:

1. `top_k`
2. Metadata filtering
3. Query rewriting
4. `chunk_size`
5. Metadata filtering

This controlled approach makes it possible to identify the effect of individual retrieval changes instead of changing several variables simultaneously.

### Day 9 Key Findings

The experiments demonstrated that:

* Increasing `top_k` can introduce additional context and cross-document content.
* Metadata filtering can remove cross-document competition.
* Query rewriting can improve retrieval when vocabulary differs between the question and policy.
* Increasing chunk size can reduce the number of chunks without improving retrieval ranking.
* Retrieval changes should be evaluated using measurements rather than assumptions.

These results are treated as retrieval-behavior observations rather than claims that every optimization improves the baseline system.

---

## Overall RAG Architecture

```text
                    30 Policy Documents
                            |
                            v
                    Document Sanitization
                            |
                            v
                         Chunking
                            |
                            v
                      Embeddings
                            |
                            v
                Shared ChromaDB Collection
                            |
                            v
                      User Question
                            |
                            v
                    Query Embedding
                            |
                            v
                  Retrieval / Filtering
                            |
                            v
                    Relevant Chunks
                            |
                            v
                  Grounded Generation
                            |
                            v
                  Structured Response
                            |
                    +-------+-------+
                    |               |
                    v               v
                Citations       Abstention
                    |               |
                    +-------+-------+
                            |
                            v
                     Final Answer
```

---

## Project Progress

| Day   | Topic                                         | Status    |
| ----- | --------------------------------------------- | --------- |
| Day 1 | Engineering foundations                       | Completed |
| Day 2 | LLM and Prompt Engineering fundamentals       | Completed |
| Day 3 | Prompt Engineering project                    | Completed |
| Day 4 | Prompt Testing and Validation                 | Completed |
| Day 5 | Document Sanitization and Chunking            | Completed |
| Day 6 | Embedding Generation                          | Completed |
| Day 7 | Baseline RAG                                  | Completed |
| Day 8 | Grounded Generation, Citations and Abstention | Completed |
| Day 9 | Retrieval Evaluation and Optimization         | Completed |

---

## Key Concepts Learned

The project now covers:

* Prompt Engineering
* Prompt Templates
* Structured Outputs
* Pydantic Validation
* Prompt Testing
* Prompt Versioning
* Failure Categorization
* Latency Measurement
* Token Usage Measurement
* Document Sanitization
* Regular Expressions
* Document Chunking
* Chunk Metadata
* Embeddings
* Vector Search
* ChromaDB
* Top-k Retrieval
* Metadata Filtering
* Distance Thresholds
* Retrieval-Augmented Generation
* Grounded Generation
* Citation Validation
* Evidence Thresholds
* Abstention
* Retrieval Evaluation
* Hit Rate@k
* Mean Reciprocal Rank (MRR)
* Root-Cause Analysis
* Query Rewriting
* Chunk-Size Optimization
* Controlled Retrieval Experiments

---

## Overall Project Goal

The project progressively evolves from a simple prompt-based GenAI application into a more reliable Retrieval-Augmented Generation system.

```text
Prompt Engineering
        |
        v
Structured and Validated Outputs
        |
        v
Document Preprocessing
        |
        v
Chunking
        |
        v
Embeddings
        |
        v
Vector Search
        |
        v
Baseline RAG
        |
        v
Grounded Generation
        |
        v
Citation Validation
        |
        v
Abstention
        |
        v
Retrieval Evaluation
        |
        v
Retrieval Optimization
```

The goal is not only to generate answers, but to build a GenAI system that can:

* Retrieve relevant evidence
* Generate answers grounded in that evidence
* Cite the correct source chunks
* Abstain when evidence is insufficient
* Measure retrieval behavior
* Diagnose retrieval problems
* Experiment with retrieval strategies
* Improve the system using controlled, measurable changes
# GenAI Day 11 - FastAPI Layer for RAG

## Project Overview

Day 11 adds a FastAPI API layer on top of the existing RAG pipeline.

The goal is to expose the existing RAG functionality through simple REST API endpoints without duplicating the retrieval, generation, citation, or validation logic.

The FastAPI routes remain thin and delegate the actual RAG processing to the existing modules.

## Day 11 Focus

Day 11 focuses on:

- FastAPI application setup
- REST API endpoint creation
- Request validation using Pydantic
- Document ingestion through an approved file reference
- Question answering through the existing RAG pipeline
- Document metadata lookup
- HTTP error handling
- Swagger API documentation
- Keeping route logic thin

## Project Structure

```text
day_11_fastapi/
│
├── main.py
├── routes.py
├── models.py
└── documents.py

# Day 12 — API Observability, Error Handling and Testing

## Objective

Make the FastAPI RAG service diagnosable by recording request IDs, sources, timing, model/prompt versions, outcomes and errors.

---

## Work Completed

### 1. SQL Request Logging

Created SQLite observability tables in:

`day_11_fastapi/observability.db`

#### requests table

Stores:

- request_id
- endpoint
- start_time
- total_latency_ms
- model_version
- prompt_version
- outcome
- error_category

#### retrieved_sources table

Stores:

- request_id
- source_id
- retrieval score

This connects each RAG request with the chunks retrieved for that request.

---

### 2. Structured Request Logging

Every API request receives a unique request ID.

The middleware:

- Generates a UUID request ID
- Records request start and end
- Measures request latency
- Adds `X-Request-ID` to the response
- Handles unexpected internal errors

Example:

`X-Request-ID: 93ccab96-dea3-4250-814e-77cb9f1a3efc`

---

### 3. RAG Request Logging

The `/ask` endpoint records:

- Request ID
- Endpoint
- Latency
- Generation model
- Prompt version
- Outcome
- Error category
- Retrieved source IDs
- Retrieval scores

Possible outcomes include:

- `answered`
- `insufficient_evidence`
- `error`

---

### 4. Ingestion Logging

The `/ingest` endpoint now records successful and failed ingestion requests in SQLite.

Example outcome:

`ingested`

LLM model and prompt version are left as `NULL` because ingestion does not perform generation.

---

### 5. Consistent Error Responses

Standardized error responses were implemented for:

- Invalid request input
- Unknown documents
- Provider failures
- Internal errors

Example:

```json
{
  "request_id": "example-request-id",
  "error": {
    "category": "document_not_found",
    "message": "Document not found"
  }
}

---

# Day 13 — Golden Evaluation Dataset and Runner

## Objective

Create a representative golden evaluation dataset and a reproducible evaluation runner to measure the quality of the RAG system consistently.

## What Was Completed

### 1. Evaluation Case Format

Defined a Pydantic `EvaluationCase` model in:

`day_13_evaluation/models.py`

Each evaluation case contains:

- `case_id`
- `question`
- `category`
- `expected_source_ids`
- `answerability`
- `expected_facts`
- `answer_notes`

### 2. Golden Evaluation Dataset

Created:

`day_13_evaluation/golden_set.jsonl`

The dataset contains 25 cases using the approved document corpus.

| Category | Cases |
|---|---:|
| Answerable | 10 |
| Unanswerable | 5 |
| Ambiguous | 4 |
| Multi-document | 3 |
| Adversarial | 3 |
| **Total** | **25** |

The cases are designed to test normal answering, insufficient evidence, ambiguity, multi-document reasoning, and false-premise/adversarial questions.

### 3. Dataset Review

Created:

`day_13_evaluation/review_notes.md`

The dataset was manually reviewed for:

- Question clarity
- Correct category
- Expected source IDs
- Answerability
- Expected facts
- Scoreability

A correction was documented for `CASE025`, an adversarial leave-policy question containing a false premise.

### 4. Evaluation Runner

Created:

`day_13_evaluation/run_evals.py`

The runner reads the golden dataset and sends every question to the existing FastAPI `/ask` endpoint.

It reuses the existing RAG pipeline rather than implementing a second RAG flow.

Evaluation flow:

```text
golden_set.jsonl
       ↓
run_evals.py
       ↓
FastAPI /ask
       ↓
Existing RAG pipeline
       ↓
Evaluation result
```

---

# Day 14 — RAG Evaluation, Scorecard, and Regression Checks

## Objective

Build an automated evaluation and regression framework around the Day 13 golden evaluation dataset to measure retrieval quality, answer quality, failures, latency, and configuration regressions.

## What Was Completed

### 1. Retrieval Grader

Created:

`day_14_evaluation/retrieval_grader.py`

The retrieval grader evaluates:

- Hit Rate
- Recall@3
- Mean Reciprocal Rank (MRR)
- Retrieval errors
- Not-applicable cases

Baseline results:

| Metric | Result |
|---|---:|
| Hit Rate | 1.000 |
| Recall@3 | 0.904 |
| MRR | 0.947 |

### 2. Answer Grader

Created:

`day_14_evaluation/answer_grader.py`

The answer grader evaluates:

- Answerability accuracy
- Expected fact coverage
- Citation presence
- Citation validity
- Abstention behavior
- Answer pass rate

Baseline results:

| Metric | Result |
|---|---:|
| Answerability | 0.720 |
| Expected Facts | 0.781 |
| Citation Validity | 1.000 |
| Answer Pass Rate | 0.640 |

### 3. Review-Friendly Evaluation Report

Created:

`day_14_evaluation/review_report.py`

The report combines retrieval and answer evaluation results for each case.

It also records:

- Latency
- Failure category
- Failure reason

The report automatically identifies the top failure categories from the evaluation results.

### 4. Failure Analysis

The evaluation identified the following top failure categories:

| Failure Category | Cases |
|---|---:|
| Over-abstention on ambiguous or multi-document cases | 4 |
| Partial expected-fact coverage | 2 |
| Over-abstention on adversarial false-premise cases | 2 |

### 5. Evaluation Scorecard

Created:

`day_14_evaluation/scorecard.py`

The scorecard consolidates retrieval and answer metrics into a single evaluation summary.

It also includes:

- Failure analysis
- Average latency
- Minimum latency
- Maximum latency
- Evaluation workload cost proxy

### 6. Regression Checks

Created:

`day_14_evaluation/regression_check.py`

Minimum acceptable thresholds were defined for the main retrieval and answer metrics.

The normal baseline configuration passed all regression checks.

A deliberately weakened configuration using:

`top_k = 1`

was also tested.

The weakened configuration reduced retrieval performance:

| Metric | Baseline | Weakened |
|---|---:|---:|
| Hit Rate | 1.000 | 0.895 |
| Recall@3 | 0.904 | 0.772 |
| MRR | 0.947 | 0.895 |

The regression checker correctly detected the degradation and returned:

`REGRESSION CHECK: FAIL`

The original baseline configuration was then restored and verified with:

`REGRESSION CHECK: PASS`

## Day 14 Evaluation Flow

```text
Day 13 Golden Set
       ↓
RAG Evaluation Results
       ↓
┌───────────────────────┐
│ Retrieval Grader      │
│ Answer Grader         │
└───────────────────────┘
       ↓
Review Report
       ↓
Scorecard
       ↓
Regression Checks
       ↓
PASS / FAIL
```
