import json
from pathlib import Path


EMBEDDINGS_FILE = Path(
    "day_6_Vector_search/embeddings.jsonl"
)

SELECTED_DOCUMENTS = {
    "DOC001",
    "DOC002",
    "DOC003",
    "DOC004",
    "DOC005",
}


with EMBEDDINGS_FILE.open("r", encoding="utf-8") as file:
    records = [
        json.loads(line)
        for line in file
    ]


selected_records = [
    record
    for record in records
    if record["document_id"] in SELECTED_DOCUMENTS
]


print("Total selected chunks:", len(selected_records))

print("\n" + "=" * 80)

for record in selected_records:
    print("\nDocument ID:", record["document_id"])
    print("Title:", record["title"])
    print("Chunk Index:", record["chunk_index"])
    print("Text:")
    print(record["text"])
    print("=" * 80)