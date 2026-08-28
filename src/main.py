from model_client import ModelClient
from validator import validate_response
from models import SummarizationOutput

# Read prompt template
with open("prompts/summarization.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Read input text
with open("inputs/summary_incomplete.txt", "r", encoding="utf-8") as f:
    article = f.read()

# Replace placeholder with actual text
system_prompt = system_prompt.replace("{{TEXT}}", article)

# Generate AI response
client = ModelClient()

try:
    result = client.generate(system_prompt, "")
except Exception as e:
    print("Model/Network Error:")
    print(e)
    exit()

# Validate response
validated_data, error = validate_response(
    {"summary": result["text"]},
    SummarizationOutput
)

# Check validation result
if error:
    print("Validation Error:")
    print(error)
else:
    print("Validation Successful!")
    print("Generated Text:")
    print(validated_data.summary)

# Model information
print("\nModel:", result["model"])
print("Latency:", result["latency"])
print("Prompt Tokens:", result["prompt_tokens"])
print("Completion Tokens:", result["completion_tokens"])
print("Total Tokens:", result["total_tokens"])