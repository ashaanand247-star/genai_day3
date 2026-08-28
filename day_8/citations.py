import re


def extract_citations(answer):

    pattern = r"\[Source: .*?\| Document: .*?\| Chunk: .*?\]"

    return re.findall(
        pattern,
        answer
    )


def validate_citations(
    citations,
    allowed_citations
):

    return [
        citation
        for citation in citations
        if citation in allowed_citations
    ]


def build_allowed_citations(results):

    allowed_citations = []

    for metadata in results["metadatas"][0]:

        citation = (
            f"[Source: {metadata['title']} "
            f"| Document: {metadata['document_id']} "
            f"| Chunk: {metadata['chunk_index']}]"
        )

        allowed_citations.append(
            citation
        )

    return allowed_citations