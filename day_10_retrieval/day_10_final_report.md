# Day 10 — Advanced Retrieval and Improvement: Final Report

## 1. Objective

The goal of Day 10 was to improve weak retrieval cases using measured retrieval experiments and select a configuration that improves retrieval quality without introducing unacceptable regressions.

The experiments evaluated:

- Top-k changes
- Automatic query rewriting
- Chunk-size changes
- TF-IDF reranking

All experiments were compared using the complete 10-question retrieval test set where applicable.

---

## 2. Baseline Note

The Day 9 baseline report records:

- Hit Rate@3 = 1.000
- MRR = 1.000

That report was calculated from the saved Day 6 retrieval-result artifact.

For Day 10, the same 10-question test set was evaluated against the current Day 7 ChromaDB using the current retrieval logic.

Therefore, the reproducible Day 10 baseline was:

- Hit Rate@3 = 0.700
- MRR = 0.700

The Day 9 artifacts were preserved and were not overwritten.

---

## 3. Experiment 1 — Top-k 3 → 2

### Configuration

| Parameter | Baseline | Experiment |
|---|---:|---:|
| Top-k | 3 | 2 |
| Chunk size | 100 | 100 |
| Overlap | 20 | 20 |

### Target-case result

Question:

> What should employees do if they become unavailable while working remotely?

Baseline:

- DOC002 rank 1, distance 0.805142
- DOC003 rank 2, distance 0.948272
- DOC019 rank 3, distance 1.056665

Experiment:

- DOC002 rank 1, distance 0.805142
- DOC003 rank 2, distance 0.948272

The relevant DOC002 remained rank 1 and the unnecessary third result was removed.

### Full validation

| Metric | Baseline | Top-k 2 |
|---|---:|---:|
| Hit Rate@3 | 0.700 | 0.700 |
| MRR | 0.700 | 0.700 |
| Regressions | — | 0 |

### Decision

**Rejected as final configuration.**

Top-k reduction was safe and reduced the amount of retrieved context, but it produced no aggregate retrieval gain.

---

## 4. Experiment 2 — Automatic Query Rewriting

### Approach

The original user question was automatically rewritten internally before embedding.

Original:

> What should employees do if they become unavailable while working remotely?

Generated rewrite:

> What is the prescribed procedure for employees to follow when they become unavailable while working remotely?

### Result

Baseline:

- DOC002 rank 1, distance 0.805142
- DOC003 rank 2, distance 0.948272
- DOC019 rank 3, distance 1.056665

Rewritten query:

- DOC002 rank 1, distance 0.919793
- DOC003 rank 2, distance 1.021239
- DOC019 rank 3, distance 1.157526

The relevant document remained rank 1, but its distance became worse.

### Decision

**Rejected.**

The rewrite preserved the original meaning, but the measured retrieval result did not improve. This experiment therefore did not provide evidence for selecting query rewriting.

---

## 5. Experiment 3 — Chunk Size 100 → 70

### Configuration

| Parameter | Baseline | Experiment |
|---|---:|---:|
| Chunk size | 100 | 70 |
| Overlap | 20 | 20 |
| Top-k | 3 | 3 |

The chunk-size=70 index was built separately at:

`day_10_retrieval/chroma_db_chunk70`

It contains 192 vectors from 30 documents.

The original baseline index was not modified.

### Full Q1–Q10 validation

| Question | Expected Doc | Baseline Rank | Chunk 70 Rank |
|---|---|---:|---:|
| Q1 | DOC001 | Not found | Not found |
| Q2 | DOC002 | 1 | 1 |
| Q3 | DOC002 | 1 | 1 |
| Q4 | DOC002 | 1 | 1 |
| Q5 | DOC003 | 1 | 1 |
| Q6 | DOC003 | Not found | Not found |
| Q7 | DOC003 | Not found | 3 |
| Q8 | DOC004 | 1 | 1 |
| Q9 | DOC005 | 1 | 1 |
| Q10 | DOC005 | 1 | 1 |

### Aggregate retrieval metrics

| Metric | Baseline | Chunk 70 | Change |
|---|---:|---:|---:|
| Hit Rate@3 | 0.700 | 0.800 | +0.100 |
| MRR | 0.700 | 0.733 | +0.033 |
| Regressions | — | 0 | None |

### Important improvement

Q7 improved from:

`DOC003: Not found`

to:

`DOC003: Rank 3`

This demonstrates a real retrieval improvement on a previously weak question.

### Decision

**Selected.**

Chunk size 70 produced a measurable retrieval gain and caused no retrieval regressions in the complete Q1–Q10 validation.

---

## 6. Experiment 4 — TF-IDF Reranking

### Approach

The reranking pipeline was:

```text
Question
   ↓
Vector retrieval: top 5 candidate pool
   ↓
TF-IDF reranking
   ↓
Final top 3