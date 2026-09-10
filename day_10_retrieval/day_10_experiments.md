# Day 10 — Retrieval Experiments

## Objective

Improve weak retrieval cases using measured retrieval experiments while
avoiding regressions on previously passing questions.

The experiments compare retrieval configurations using the fixed 10-question
retrieval test set.

---

## Baseline Note

Day 9's frozen baseline report recorded:

- Hit Rate@3 = 1.00
- MRR = 1.00

Those metrics were calculated from the saved Day 6 retrieval result artifact.

For Day 10 experiments, the same 10-question test set was evaluated against
the current Day 7 ChromaDB using the same retrieval evaluation logic.

Therefore, the reproducible Day 10 baseline is:

- Chunk size = 100
- Chunk overlap = 20
- Top-k = 3
- Metadata filter = None
- Embedding model = `nvidia/llama-nemotron-embed-vl-1b-v2:free`
- Hit Rate@3 = 0.700
- MRR = 0.700

Day 9 artifacts were preserved and were not modified.

---

## Experiment 1 — Top-k = 3 → 2

### Configuration

**Baseline:** `top_k = 3`

**Experiment:** `top_k = 2`

### Target Question

What should employees do if they become unavailable while working remotely?

### Result

Baseline:

- DOC002 rank 1
- DOC003 rank 2
- DOC019 rank 3

Experiment:

- DOC002 rank 1
- DOC003 rank 2

DOC019 at rank 3 was removed.

### Full Validation

| Metric | Baseline | Experiment |
|---|---:|---:|
| Hit Rate@3 | 0.700 | 0.700 |
| MRR | 0.700 | 0.700 |

### Regression

No retrieval regressions were found across the complete 10-question
retrieval test set.

### Outcome

**Rejected as the final improvement.**

Reducing top-k removed an unnecessary third result in the tested case,
but it did not improve aggregate retrieval metrics.

The configuration was therefore not selected.

---

## Experiment 2 — Automatic Query Rewriting

### Configuration

**Baseline:** Original query + `top_k=3`

**Experiment:** LLM-generated rewritten query + `top_k=3`

### Original Query

What should employees do if they become unavailable while working remotely?

### Generated Rewrite

What is the prescribed procedure for employees to follow when they become
unavailable while working remotely?

### Result

Baseline:

- DOC002 rank 1
- DOC003 rank 2
- DOC019 rank 3
- DOC002 distance = 0.8051

Rewritten query:

- DOC002 rank 1
- DOC003 rank 2
- DOC019 rank 3
- DOC002 distance = 0.9198

Lower distance represents better similarity.

### Outcome

**Rejected.**

The rewritten query preserved the ranking but produced a worse distance
for the relevant document.

This experiment therefore did not provide a measured retrieval improvement
for the tested case.

The rejection is based on measured evidence rather than assuming that
query rewriting is generally ineffective.

---

## Experiment 3 — Chunk Size 100 → 70

### Configuration

**Baseline:**

- `chunk_size = 100`
- `chunk_overlap = 20`
- `top_k = 3`

**Experiment:**

- `chunk_size = 70`
- `chunk_overlap = 20`
- `top_k = 3`

Only the chunk size was changed.

### Target Question

What training is required during employee onboarding?

### Target-Case Result

| Configuration | Rank-1 Distance |
|---|---:|
| chunk_size=100 | 0.9686 |
| chunk_size=70 | 0.8997 |

The baseline rank-1 chunk contained only the beginning of the Training
section.

The smaller chunk produced a more focused chunk containing the relevant
training requirements.

### Full 10-Question Validation

| Metric | Baseline | chunk_size=70 | Change |
|---|---:|---:|---:|
| Hit Rate@3 | 0.700 | 0.800 | +0.100 |
| MRR | 0.700 | 0.733 | +0.033 |

### Regression

No retrieval regressions were found across the complete 10-question
retrieval test set.

Q7 showed a measurable improvement:

- Baseline: expected DOC003 not retrieved
- chunk_size=70: expected DOC003 retrieved at rank 3

### Outcome

**Selected as the final retrieval improvement.**

The smaller chunk size improved both Hit Rate@3 and MRR without causing
retrieval regressions.

---

## Experiment 4 — TF-IDF Reranking

### Configuration

**Baseline:**

- Vector retrieval
- Candidate pool = top 3

**Experiment:**

- Vector retrieval candidate pool = top 5
- Apply TF-IDF reranking
- Select final top 3
- Preserve original vector distance and original rank

The experiment preserved:

- Original rank
- Original vector distance
- TF-IDF reranking score
- Final rank/order
- Document and chunk metadata

### Full 10-Question Validation

| Metric | Baseline | Reranking |
|---|---:|---:|
| Hit Rate@3 | 1.000 | 1.000 |
| MRR | 1.000 | 0.883 |

### Regression

No Hit Rate regressions were observed, but ranking quality decreased.

Examples:

- Q5 relevant document moved from rank 1 → rank 2
- Q9 relevant document moved from rank 1 → rank 3

### Latency

Average vector retrieval latency:

`0.033168 seconds`

Average reranking latency:

`0.002975 seconds`

Average total experiment latency:

`0.036143 seconds`

### Outcome

**Rejected.**

TF-IDF reranking did not improve Hit Rate and reduced MRR from 1.000
to 0.883 in the reranking validation.

The approach was rejected based on measured results.

This does not imply that reranking is generally ineffective; this specific
TF-IDF reranking approach did not improve retrieval quality on this dataset.

---

## Latency and Complexity Analysis — Selected Configuration

The selected configuration changes the chunk size from 100 to 70 while
keeping overlap and top-k unchanged.

### Retrieval Latency

| Metric | Baseline | chunk_size=70 |
|---|---:|---:|
| Average retrieval latency | 0.002501 s | 0.003028 s |

Latency change:

`+0.000527 seconds`

Percentage change:

`+21.06%`

### Vector Count

| Metric | Baseline | chunk_size=70 |
|---|---:|---:|
| Vector count | 134 | 192 |

Vector count change:

`+43.28%`

### Trade-off

The selected configuration provides:

- Hit Rate@3 improvement: +0.100
- MRR improvement: +0.033
- No retrieval regressions
- Approximately 21.06% higher retrieval latency
- Approximately 43.28% more vectors

The measured quality improvement was considered worthwhile for the Day 10
retrieval objective.

---

## Selected Final Configuration

The selected Day 10 retrieval configuration is:

```text
Embedding model: nvidia/llama-nemotron-embed-vl-1b-v2:free
Chunk size: 70
Chunk overlap: 20
Top-k: 3
Metadata filter: None
Score threshold: 1.0
Prompt version: day_8_current