# GenAI Training – Day 5, Day 6 & Day 7

## Overview

This project covers the document preprocessing, vector-search, and baseline Retrieval-Augmented Generation (RAG) stages of a GenAI pipeline.

The project is built incrementally across Day 5, Day 6, and Day 7.

---

# Day 5 – Document Preprocessing

## Objective

The objective of Day 5 was to prepare raw employee and company policy documents so that they could later be used for embedding and vector search.

The preprocessing pipeline was:

```text
Raw Documents
      ↓
Sanitization
      ↓
Cleaning
      ↓
Chunking
      ↓
chunks.jsonl