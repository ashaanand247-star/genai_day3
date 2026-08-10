import os

# -----------------------------
# API Configuration
# -----------------------------

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

# -----------------------------
# Model Configuration
# -----------------------------

MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b:free"

# -----------------------------
# Prompt Files
# -----------------------------

PROMPT_V1 = "src/prompts/summarization.txt"
PROMPT_V2 = "src/prompts/summarization_v2.txt"

# -----------------------------
# Dataset
# -----------------------------

TEST_CASES_FILE = "src/datasets/prompt_test_cases.json"

# -----------------------------
# Output Files
# -----------------------------

RESULTS_FILE = "src/outputs/results.json"

# -----------------------------
# Logging
# -----------------------------

LOG_FILE = "src/outputs/application.log"

# -----------------------------
# Version
# -----------------------------

PROMPT_VERSION = "v2"