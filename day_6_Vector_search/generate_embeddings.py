import os
import json
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables and create client

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# Configuration

INPUT_FILE = Path("day5_preprocessing/chunks.jsonl")
OUTPUT_FILE = Path("day_6_Vector_search/embeddings.jsonl")

EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

BATCH_SIZE = 10

# Function 1: Read chunk dataset

def load_chunks(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file]

    return chunks

# Function 2: Load existing embeddings

def load_existing_embeddings(file_path):
    existing_embeddings = {}

    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                existing_embeddings[record["chunk_id"]] = record

    return existing_embeddings

# Function 3: Check whether an embedding can be reused

def prepare_chunk_for_embedding(
    chunk,
    existing_embeddings,
    embedding_model
):
    text_hash = hashlib.sha256(
        chunk["text"].encode("utf-8")
    ).hexdigest()

    existing = existing_embeddings.get(
        chunk["chunk_id"]
    )

    if existing and existing.get("embedding_model") == embedding_model:

        saved_hash = existing.get("text_hash")

        # Older embeddings may not have text_hash
        if saved_hash is None:
            saved_hash = hashlib.sha256(
                existing["text"].encode("utf-8")
            ).hexdigest()

        # Text has not changed → reuse embedding
        if saved_hash == text_hash:

            record = existing.copy()
            record["text_hash"] = text_hash

            return record, None

    # New chunk or changed chunk
    return None, (chunk, text_hash)

# Function 4: Generate embeddings for new/changed chunks


def generate_new_embeddings(
    chunks_to_embed,
    embedding_model,
    client
):
    embedded_chunks = []

    if not chunks_to_embed:
        return embedded_chunks

    texts = [
        chunk["text"]
        for chunk, _ in chunks_to_embed
    ]

    response = client.embeddings.create(
        model=embedding_model,
        input=texts,
        encoding_format="float"
    )

    for (chunk, text_hash), embedding_data in zip(
        chunks_to_embed,
        response.data
    ):
        record = chunk.copy()

        record["embedding"] = embedding_data.embedding
        record["embedding_model"] = embedding_model
        record["text_hash"] = text_hash

        embedded_chunks.append(record)

    return embedded_chunks

# Function 5: Process one batch


def process_batch(
    batch,
    existing_embeddings,
    embedding_model,
    client
):
    reused_chunks = []
    chunks_to_embed = []

    for chunk in batch:

        reused_record, new_chunk = prepare_chunk_for_embedding(
            chunk,
            existing_embeddings,
            embedding_model
        )

        if reused_record is not None:
            reused_chunks.append(reused_record)

        else:
            chunks_to_embed.append(new_chunk)

    generated_chunks = generate_new_embeddings(
        chunks_to_embed,
        embedding_model,
        client
    )

    embedded_chunks = (
        reused_chunks + generated_chunks
    )

    return (
        embedded_chunks,
        len(reused_chunks),
        len(generated_chunks)
    )


# Function 6: Save embeddings

def save_embeddings(records, file_path):
    with file_path.open("w", encoding="utf-8") as file:

        for record in records:
            file.write(
                json.dumps(record) + "\n"
            )


# Function 7: Main program


def main():

    # Load chunks
    chunks = load_chunks(INPUT_FILE)

    print("Total chunks:", len(chunks))

    # Load existing embeddings
    existing_embeddings = load_existing_embeddings(
        OUTPUT_FILE
    )

    print(
        "Existing embeddings:",
        len(existing_embeddings)
    )

    # Store final results
    embedded_chunks = []

    # Process chunks in batches
    for start in range(
        0,
        len(chunks),
        BATCH_SIZE
    ):

        batch = chunks[
            start:start + BATCH_SIZE
        ]

        (
            batch_results,
            reused_count,
            generated_count
        ) = process_batch(
            batch,
            existing_embeddings,
            EMBEDDING_MODEL,
            client
        )

        embedded_chunks.extend(
            batch_results
        )

        print(
            f"Processing batch: "
            f"{start + 1}-"
            f"{min(start + BATCH_SIZE, len(chunks))} | "
            f"Reused: {reused_count} | "
            f"Generated: {generated_count}"
        )

    # Save final embeddings
    save_embeddings(
        embedded_chunks,
        OUTPUT_FILE
    )

    print(
        "Saved embeddings:",
        len(embedded_chunks)
    )


# Program entry point


if __name__ == "__main__":
    main()