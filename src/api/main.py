"""
FastAPI application - HTTP endpoints for chat service.

PSEUDOCODE - The actual REST API that receives HTTP requests
"""

# Imports needed:
# - FastAPI, HTTPException
# - ChatRequest, ChatResponse, HealthResponse from models
# - ChatService from chat_service
# - AppConfig from configs

app = FastAPI(title="LocalRagChat API", version="0.1.0")

# Global variable (singleton pattern)
chat_service = None


@app.on_event("startup")
def startup():
    """
    RUN ONCE WHEN SERVER STARTS

    Logic:
    1. Load AppConfig.production()
    2. Create ChatService(config) → this loads the model
    3. Store in global variable
    4. Log "API ready to accept requests"

    This is why startup takes 10+ seconds but requests are fast
    """
    pass


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    POST /chat - Main chat endpoint

    Request body:
    {
        "message": "hello",
        "session_id": "abc-123"  // optional
    }

    Response:
    {
        "response": "Hi there!",
        "session_id": "abc-123"
    }

    Logic:
    1. Validate request (automatic via Pydantic)
    2. Call chat_service.chat(request.message, request.session_id)
    3. Get (response, session_id) tuple back
    4. Return ChatResponse object
    5. If error → raise HTTPException
    """
    pass


@app.get("/health", response_model=HealthResponse)
def health_endpoint():
    """
    GET /health - Health check endpoint

    Response:
    {
        "status": "healthy",
        "model_loaded": true,
        "storage_type": "pgvector"
    }

    Logic:
    1. Check if chat_service exists
    2. Call chat_service.is_ready()
    3. Get config.storage_type
    4. Return HealthResponse
    """
    pass


@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    """
    DELETE /sessions/{session_id} - Clean up a session

    URL: /sessions/abc-123

    Response:
    {
        "message": "Session deleted",
        "session_id": "abc-123"
    }

    Logic:
    1. Call chat_service.cleanup_session(session_id)
    2. If True → return success
    3. If False → raise 404 Not Found
    """
    pass


@app.get("/sessions/count")
def session_count_endpoint():
    """
    GET /sessions/count - Get number of active sessions

    Response:
    {
        "count": 5
    }

    Logic:
    1. Call chat_service.get_session_count()
    2. Return as JSON
    """
    pass


# Entry point for running with: uvicorn src.api.main:app
# Can also add: --reload for development
# Can also add: --host 0.0.0.0 --port 8000
