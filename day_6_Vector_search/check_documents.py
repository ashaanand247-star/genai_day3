import json

with open("day5_preprocessing/chunks.jsonl", "r", encoding="utf-8") as file:
    chunks = [json.loads(line) for line in file]

documents = {}

for chunk in chunks:
    document_id = chunk["document_id"]

    if document_id not in documents:
        documents[document_id] = chunk["title"]

    if len(documents) == 5:
        break

print("Selected 5 documents:")

for document_id, title in documents.items():
    print(document_id, "->", title)