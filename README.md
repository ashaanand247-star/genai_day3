# GenAI Day 4 - Prompt Testing and Validation

## Project Overview

This project extends the Day 3 Prompt Engineering project into a testable and measurable GenAI component.

The application uses the OpenRouter API with Python to generate AI responses and validates the generated output using Pydantic models.

Day 4 focuses on:

- Structured outputs
- Response validation
- Fixed prompt test cases
- Prompt test automation
- Failure categorization
- Prompt version comparison
- Latency and token measurement

---

## Features

- Prompt-based AI interaction
- Prompt templates stored separately from Python code
- Multiple AI task prompt templates:
  - Text Summarization
  - Text Classification
  - Information Extraction
- Secure API key management using `.env`
- Pydantic output models
- AI response validation
- Fixed 10-case prompt test dataset
- Automated prompt test runner
- Failure categorization
- Prompt version logging
- Model version logging
- API latency measurement
- Token usage measurement
- V1 and V2 prompt comparison
- Machine-readable JSON test results

---

## Project Structure

```text
Gen_ai_day3prompt/
│
├── src/
│   │
│   ├── datasets/
│   │   └── prompt_test_cases.json
│   │
│   ├── inputs/
│   │   ├── article.txt
│   │   ├── employee.txt
│   │   ├── news.txt
│   │   ├── summary_normal.txt
│   │   ├── summary_long.txt
│   │   ├── summary_ambiguous.txt
│   │   ├── summary_incomplete.txt
│   │   ├── summary_empty.txt
│   │   ├── summary_malformed.txt
│   │   ├── summary_mixed.txt
│   │   ├── summary_numbers.txt
│   │   ├── summary_short.txt
│   │   └── summary_special.txt
│   │
│   ├── outputs/
│   │   ├── prompt_comparison.md
│   │   ├── results_v1.json
│   │   └── results_v2.json
│   │
│   ├── prompts/
│   │   ├── classification.txt
│   │   ├── extraction.txt
│   │   ├── summarization.txt
│   │   └── summarization_v2.txt
│   │
│   ├── config.py
│   ├── logger.py
│   ├── main.py
│   ├── model_client.py
│   ├── models.py
│   ├── prompt_txt.py
│   ├── test_runner.py
│   └── validator.py
│
├── .env
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt