from pathlib import Path
import json


def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text, chunk_size=100, overlap=20):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    lines = text.splitlines()

    # Store each non-empty line along with its words
    line_data = []

    for line in lines:
        words = line.split()

        if words:
            line_data.append(words)

    chunks = []
    start_line = 0

    while start_line < len(line_data):

        chunk_words = []
        end_line = start_line

        # Build a chunk without splitting a line
        while end_line < len(line_data):
            next_line = line_data[end_line]

            if chunk_words and len(chunk_words) + len(next_line) > chunk_size:
                break

            chunk_words.extend(next_line)
            end_line += 1

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        # We reached the end of the document.
        # Do not create another chunk containing only overlap.
        if end_line >= len(line_data):
            break

        # Find lines to keep as overlap
        overlap_words = 0
        next_start_line = end_line

        while next_start_line > start_line:
            next_start_line -= 1
            overlap_words += len(line_data[next_start_line])

            if overlap_words >= overlap:
                break

        # Prevent the same chunk from being repeated
        if next_start_line == start_line:
            start_line = end_line
        else:
            start_line = next_start_line

    return chunks


def create_metadata(document_path, chunk_index):
    filename = document_path.stem
    parts = filename.split("_", 1)

    document_id = parts[0]
    title = parts[1].replace("_", " ").title()
    updated_at = document_path.stat().st_mtime

    return {
        "chunk_id": f"{document_id}_chunk_{chunk_index}",
        "document_id": document_id,
        "title": title,
        "source_path": str(document_path),
        "updated_at": updated_at,
        "chunk_index": chunk_index
    }


# Input folder containing cleaned Markdown documents
cleaned_folder = Path(
    "day5_preprocessing/cleaned_documents"
)

# Output JSONL file
output_file = Path(
    "day5_preprocessing/chunks.jsonl"
)


with open(output_file, "w", encoding="utf-8") as file:

    for document_path in cleaned_folder.glob("*.md"):

        # Load document
        document_content = load_document(document_path)

        # Create chunks
        chunks = chunk_text(
            document_content,
            chunk_size=100,
            overlap=20
        )

        print(f"\n{document_path.name}")
        print(f"Number of chunks: {len(chunks)}")

        # Process each chunk
        for index, chunk in enumerate(chunks):

            # Create metadata
            metadata = create_metadata(
                document_path,
                index
            )

            # Combine metadata and chunk text
            record = {
                **metadata,
                "text": chunk
            }

            # Write one JSON object per line
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            # Display chunk information
            print(f"\nChunk {index}:")

            print(
                "START:",
                " ".join(chunk.split()[:20])
            )

            print(
                "END:",
                " ".join(chunk.split()[-20:])
            )

            print(
                "METADATA:",
                metadata
            )


print(
    f"\nChunk dataset saved to: {output_file}"
)