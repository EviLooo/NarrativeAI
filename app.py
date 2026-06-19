import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from generation_engine import generate_narrative_response

app = FastAPI(title="AI Narrative Continuity Companion")

# Templates Setup
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Serves the main single-page application UI."""
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/media/S01_E01_helllofriend.mp4")
async def get_video():
    return FileResponse("S01_E01_helllofriend.mp4", media_type="video/mp4")

@app.get("/api/ask")
async def ask_question(q: str, timestamp: float):
    """
    RAG API Endpoint:
    1. Enforces Spoiler Shield at database query.
    2. Runs hybrid retrieval.
    3. Runs LLM inference via watsonx.ai.
    """
    try:
        # Generate the response and return the context so the UI can show the active shield
        answer, context = generate_narrative_response(q, timestamp)
        return {"answer": answer, "context": context}
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}", "context": "Error during database query."}

if __name__ == "__main__":
    # Start the server locally on port 8000
    print("=========================================================")
    print("Starting AI Narrative Continuity Companion MVP Web App...")
    print("Open http://localhost:8000 in your browser to test!")
    print("=========================================================")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
