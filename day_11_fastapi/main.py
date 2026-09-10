import time
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from day_11_fastapi.routes import router


app = FastAPI(
    title="Employee RAG API",
    version="1.0.0"
)


# Handle validation errors

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    request_id = getattr(
        request.state,
        "request_id",
        str(uuid.uuid4())
    )

    return JSONResponse(
        status_code=422,
        content={
            "request_id": request_id,
            "error": {
                "category": "validation_error",
                "message": "Invalid request input"
            }
        },
        headers={
            "X-Request-ID": request_id
        }
    )


# Handle HTTP errors

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException
):
    request_id = getattr(
        request.state,
        "request_id",
        str(uuid.uuid4())
    )

    if isinstance(exc.detail, dict):

        category = exc.detail.get(
            "category",
            "http_error"
        )

        message = exc.detail.get(
            "message",
            "HTTP request failed"
        )

    elif exc.status_code == 404:

        category = "document_not_found"
        message = "Document not found"

    else:

        category = "http_error"
        message = str(exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": request_id,
            "error": {
                "category": category,
                "message": message
            }
        },
        headers={
            "X-Request-ID": request_id
        }
    )


# Request logging middleware

@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next
):
    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()

    print(
        f"REQUEST_START "
        f"request_id={request_id} "
        f"endpoint={request.url.path}"
    )

    try:

        response = await call_next(request)

    except Exception as exc:

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        print(
            f"REQUEST_ERROR "
            f"request_id={request_id} "
            f"endpoint={request.url.path} "
            f"latency_ms={total_latency_ms:.2f} "
            f"error={type(exc).__name__}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error": {
                    "category": "internal_error",
                    "message": "Internal server error"
                }
            },
            headers={
                "X-Request-ID": request_id
            }
        )

    total_latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    response.headers["X-Request-ID"] = request_id

    print(
        f"REQUEST_END "
        f"request_id={request_id} "
        f"endpoint={request.url.path} "
        f"latency_ms={total_latency_ms:.2f}"
    )

    return response


# Register routes

app.include_router(router)