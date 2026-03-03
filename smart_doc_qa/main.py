from fastapi import FastAPI

app = FastAPI(
    title="Smart Document Q&A Assistant",
    description="API for processing documents and answering questions using RAG.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """
    Root endpoint to verify the API is running.
    """
    return {"message": "Welcome to the Smart Document Q&A Assistant API."}

# TODO: Include routers for agents, processors, etc.
