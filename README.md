# GenAI Day 3 - Prompt Engineering

## Project Overview

This project demonstrates Prompt Engineering using the OpenRouter API with Python. The application loads prompts from separate text files, sends them to an AI model, and generates responses for different NLP tasks.

---

## Features

- Prompt-based AI interaction
- Prompt templates stored separately from Python code
- Multiple AI tasks:
  - Text Summarization
  - Text Classification
  - Information Extraction
- Secure API key management using `.env`
- Measures API latency
- Displays token usage
- Organized project structure

---

## Project Structure

```
Gen_ai_day3prompt/
│
├── inputs/
│   ├── article.txt
│   ├── employee.txt
│   ├── news.txt
│   ├── summary_normal.txt
│   ├── summary_long.txt
│   ├── summary_ambiguous.txt
│   └── summary_incomplete.txt
│
├── outputs/
│
├── prompts/
│   ├── summarization.txt
│   ├── classification.txt
│   └── extraction.txt
│
├── model_client.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Technologies Used

- Python 3.13
- OpenRouter API
- OpenAI Python SDK
- python-dotenv

---

## Installation

Clone the repository

```bash
git clone https://github.com/ashaanand247-star/genai_day3.git
```

Move into the project folder

```bash
cd genai_day3
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configure API Key

Create a `.env` file in the project root.

```env
OPENROUTER_API_KEY=your_api_key_here
```

---

## Run the Project

```bash
python main.py
```

---

## Prompt Engineering

Prompts are stored separately inside the `prompts` folder.

Benefits:

- Easy to modify prompts
- Version control using Git
- Keeps Python code clean
- Reusable prompt templates

---

## Sample Test Inputs

The project includes different input types for testing prompt quality.

- Normal Input
- Long Input
- Ambiguous Input
- Incomplete Input

These help evaluate how the AI responds under different scenarios.

---

## Example Tasks

### Summarization

Generates concise summaries from long articles.

### Classification

Categorizes text into appropriate labels.

### Information Extraction

Extracts structured information from unstructured text.

---

## Output

The application displays:

- Generated AI Response
- Model Name
- Response Latency
- Prompt Tokens
- Completion Tokens
- Total Tokens

---

## Learning Outcomes

By completing this project, I learned:

- Prompt Engineering fundamentals
- Working with OpenRouter APIs
- Using the OpenAI SDK
- Secure API key management with `.env`
- Separating prompts from application logic
- Testing prompts using different input scenarios
- Git and GitHub version control

---
