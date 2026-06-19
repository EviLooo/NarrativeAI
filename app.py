import os
import shutil
import threading
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from agent_engine import generate_narrative_response

app = FastAPI(title="AI Narrative Continuity Companion")

# Templates Setup
templates = Jinja2Templates(directory="templates")

# Tracks background ingestion progress for the upload status endpoint
upload_status = {"state": "idle", "message": "", "video_filename": "S01_E01_helllofriend.mp4"}

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Serves the main single-page application UI."""
    return templates.TemplateResponse(request, "index.html", {})

# Use StaticFiles for media to support HTTP Range requests (crucial for video seeking!)
app.mount("/media", StaticFiles(directory="."), name="media")

@app.get("/api/ask")
async def ask_question(q: str, timestamp: float):
    """
    Agentic RAG API Endpoint:
    1. Llama 3.1 autonomously selects which tool to call.
    2. Tool enforces Spoiler Shield at database query.
    3. LLM generates final answer from tool output.
    """
    try:
        answer, context = generate_narrative_response(q, timestamp)
        return {"answer": answer, "context": context}
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}", "context": "Error during agent execution."}

@app.post("/api/upload")
async def upload_files(video: UploadFile = File(...), subtitle: UploadFile = File(...)):
    """
    Accepts a video file and an SRT subtitle file.
    Saves them to disk and triggers background re-ingestion into Cloudant.
    """
    global upload_status

    if upload_status["state"] == "ingesting":
        return {"success": False, "message": "Ingestion already in progress. Please wait."}

    # Save uploaded video
    video_filename = video.filename.replace(" ", "_")
    video_path = os.path.join(".", video_filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Save uploaded subtitle
    subtitle_filename = subtitle.filename.replace(" ", "_")
    subtitle_path = os.path.join(".", subtitle_filename)
    with open(subtitle_path, "wb") as f:
        shutil.copyfileobj(subtitle.file, f)

    upload_status = {"state": "ingesting", "message": "Ingestion started. This may take 2-5 minutes...", "video_filename": video_filename}

    # Run ingestion in a background thread so the API returns immediately
    def run_ingestion():
        global upload_status
        try:
            from data_parser import main as run_parser
            success = run_parser(subtitle_file=subtitle_path, db_name="narrative_ai_db")
            if success:
                upload_status = {"state": "done", "message": "Ingestion complete! Your content is ready.", "video_filename": video_filename}
            else:
                upload_status = {"state": "error", "message": "Ingestion failed. Check your subtitle file format.", "video_filename": video_filename}
        except Exception as e:
            upload_status = {"state": "error", "message": f"Ingestion error: {str(e)}", "video_filename": video_filename}

    thread = threading.Thread(target=run_ingestion, daemon=True)
    thread.start()

    return {"success": True, "message": "Files uploaded. Ingestion running in background.", "video_filename": video_filename}

@app.get("/api/upload-status")
async def get_upload_status():
    """Returns the current state of the background ingestion process."""
    return upload_status

if __name__ == "__main__":
    print("=========================================================")
    print("Starting AI Narrative Continuity Companion...")
    print("Open http://localhost:8000 in your browser to test!")
    print("=========================================================")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
