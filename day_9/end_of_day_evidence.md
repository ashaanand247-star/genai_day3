# Day 9 — End-of-Day Evidence

## Purpose

This document records the root-cause hypotheses for the five selected weak retrieval questions and the measurements used to confirm or reject those hypotheses.

The Day 9 baseline metrics were calculated automatically from the 10-question retrieval set:

- Hit Rate@3: **1.0**
- MRR: **1.0**

Therefore, these five questions are treated as selected weak/problematic retrieval cases for controlled experimentation, rather than as proven baseline failures.

---

## Test 3 — Working From Home

**Question:**  
What are the requirements for working from home?

**Expected document:** DOC002

**Failure classification:** Excessive context

### Root-cause hypothesis

Increasing `top_k` may introduce additional semantically related content from other documents, creating excessive context without improving the evidence for DOC002.

### Measurement

Compare retrieval with `top_k=3` and `top_k=5`.

Check:

- which documents are retrieved,
- their distances,
- whether additional useful DOC002 evidence appears,
- whether unrelated documents are introduced.

### Experiment result

Increasing `top_k` from 3 to 5 retrieved additional chunks, including semantically related content from another document.

### Conclusion

The experiment supports the hypothesis that a larger `top_k` can increase retrieved context and introduce cross-document content.

---

## Test 4 — Remote Unavailability

**Question:**  
What should employees do if they become unavailable while working remotely?

**Expected document:** DOC002

**Failure classification:** Missing metadata

### Root-cause hypothesis

Semantically related content from other documents may compete with DOC002 during retrieval even though the expected source is known.

### Measurement

Compare unrestricted retrieval with retrieval filtered to DOC002.

Check whether the relevant DOC002 evidence remains at or moves to the top ranking and whether cross-document results are removed.

### Experiment result

Applying the DOC002 metadata filter restricted the retrieved results to DOC002.

### Conclusion

The experiment demonstrates that metadata filtering can remove cross-document competition and focus retrieval on the selected document.

---

## Test 5 — Unexpected Absence

**Question:**  
How should employees report an unexpected absence?

**Expected document:** DOC003

**Failure classification:** Vocabulary mismatch

### Root-cause hypothesis

The question uses the phrase "unexpected absence", while the policy uses terminology such as "unplanned absence" and explicitly mentions notifying the manager or designated team contact.

### Measurement

Compare the ranking of DOC003 before and after rewriting the query using terminology closer to the policy.

### Experiment result

The rewritten query was:

> How should employees report an unplanned absence by notifying their manager or designated team contact?

DOC003 was retrieved at rank 1.

### Conclusion

The experiment supports the vocabulary-mismatch hypothesis: using terminology closer to the source policy can improve retrieval relevance.

---

## Test 8 — Employee Onboarding Training

**Question:**  
What training is required during employee onboarding?

**Expected document:** DOC004

**Failure classification:** Poor chunk boundary

### Root-cause hypothesis

A baseline `chunk_size` of 100 may divide related onboarding information across multiple chunks, affecting how much training-related evidence is contained in each retrieved chunk.

### Measurement

Compare `chunk_size=100` with `chunk_size=150`, keeping overlap fixed at 20.

Compare:

- the number of generated chunks,
- the retrieved chunks,
- the ranking of the expected document.

### Experiment result

- Baseline: **5 chunks**
- Experiment: **3 chunks**
- Baseline top document: **DOC004**
- Experiment top document: **DOC004**

Both configurations retrieved DOC004 at rank 1.

### Conclusion

Increasing chunk size changed the number of chunks but did not improve the top-document ranking for this question. Therefore, the experiment did **not confirm an improvement** from the larger chunk size.

---

## Test 9 — Conflict of Interest

**Question:**  
What should employees do if they have a potential conflict of interest?

**Expected document:** DOC005

**Failure classification:** Missing metadata

### Root-cause hypothesis

Semantically related content from other policy documents may compete with DOC005 even though the relevant conflict-of-interest information is contained in DOC005.

### Measurement

Compare unrestricted retrieval with retrieval filtered to DOC005.

Check whether the relevant "Conflicts of Interest" evidence is retrieved and whether cross-document competition is removed.

### Experiment result

The DOC005 metadata filter returned DOC005 chunks, with the relevant "Conflicts of Interest" section appearing in the highest-ranked result.

### Conclusion

The experiment supports the hypothesis that metadata filtering can focus retrieval on the expected document and remove cross-document competition.

---

## Overall Evidence

The five experiments tested controlled changes to:

1. `top_k`
2. metadata filtering
3. query rewriting
4. `chunk_size`
5. metadata filtering

Each experiment changed one primary retrieval variable at a time.

The results show that:

- Increasing `top_k` can introduce additional context.
- Metadata filtering can remove cross-document competition.
- Query rewriting can help when user terminology differs from policy terminology.
- Increasing chunk size can reduce chunk count without necessarily improving retrieval ranking.

These experiments should be interpreted as **retrieval-behavior observations**, not as claims that every change improves the baseline system.

## Day 9 Evidence Statement

For each selected weak question, a root-cause hypothesis was defined and a measurable experiment was used to confirm, reject, or qualify that hypothesis.