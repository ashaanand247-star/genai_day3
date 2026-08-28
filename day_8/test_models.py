from models import AnswerResponse


response = AnswerResponse(
    answer="Employees should submit leave requests through the approved internal leave system.",
    sources=[
        "[Source: Employee Leave Policy | Document: DOC001 | Chunk: 1]"
    ],
    chunk_previews=[
        "Employees should submit leave requests through the approved internal leave system."
    ],
    retrieval_scores=[
        0.45
    ],
    status="unknown"
)


print(response)