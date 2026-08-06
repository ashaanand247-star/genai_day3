from model_client import ModelClient

# Read prompt template
with open("prompts/classification.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Read input text
with open("inputs/news.txt", "r", encoding="utf-8") as f:
    article = f.read()

# Replace placeholder with actual text
system_prompt = system_prompt.replace("{{TEXT}}", article)

client = ModelClient()

result = client.generate(system_prompt, "")

print("Generated Text:")
print(result["text"])

print("\nModel:", result["model"])
print("Latency:", result["latency"])
print("Prompt Tokens:", result["prompt_tokens"])
print("Completion Tokens:", result["completion_tokens"])
print("Total Tokens:", result["total_tokens"])