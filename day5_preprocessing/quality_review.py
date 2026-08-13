from pathlib import Path
import json


chunks_file = Path("day5_preprocessing/chunks.jsonl")
review_file = Path("day5_preprocessing/chunk_quality_review.md")


selected_documents = [
    "DOC001",
    "DOC004",
    "DOC012"
]


with open(chunks_file, "r", encoding="utf-8") as file:
    records = [json.loads(line) for line in file]


with open(review_file, "w", encoding="utf-8") as file:

    file.write("# Chunk Quality Review\n\n")

    file.write("## Purpose\n\n")
    file.write(
        "Selected chunk examples were inspected from short, long, "
        "and structured documents.\n\n"
    )

    file.write("## Selected Documents\n\n")

    for document_id in selected_documents:

        document_chunks = [
            record
            for record in records
            if record["document_id"] == document_id
        ]

        if not document_chunks:
            continue

        file.write(
            f"### {document_id} - "
            f"{document_chunks[0]['title']}\n\n"
        )

        file.write(
            f"Number of chunks: {len(document_chunks)}\n\n"
        )

        for record in document_chunks[:2]:

            text_words = record["text"].split()

            start = " ".join(text_words[:20])
            end = " ".join(text_words[-20:])

            file.write(
                f"**Chunk ID:** `{record['chunk_id']}`  \n"
            )

            file.write(
                f"**Chunk Index:** {record['chunk_index']}  \n"
            )

            file.write(
                f"**Source:** `{record['source_path']}`  \n\n"
            )

            file.write(
                f"**START:** {start}\n\n"
            )

            file.write(
                f"**END:** {end}\n\n"
            )

    file.write("## Chunking Issue Corrected\n\n")

    file.write(
        "The initial chunking implementation could create a final chunk "
        "containing only the overlap from the previous chunk. This caused "
        "unnecessary duplicate text.\n\n"
    )

    file.write(
        "The chunking logic was updated to stop when the remaining chunk "
        "contained only the configured overlap:\n\n"
    )

    file.write(
        "```python\n"
        "if start > 0 and len(chunk_words) <= overlap:\n"
        "    break\n"
        "```\n\n"
    )

    file.write(
        "After the correction, overlap-only final chunks are no longer "
        "generated.\n"
    )


print(f"Quality review saved to: {review_file}")