# AI Narrative Continuity Companion

An AI-powered video companion that prevents spoilers while you watch shows. It uses IBM watsonx.ai, Watson NLU, and Cloudant to provide a "Spoiler Shield" that restricts LLM context strictly to the timeline of the video you have watched so far.

## Features
- **Dynamic Context Window (Spoiler Shield)**: The AI is physically incapable of spoiling the show because its context window is hard-locked to the exact second you have paused the video.
- **True Semantic Search**: Uses `ibm/slate-30m-english-rtrvr-v2` dense vectors to find scenes based on mathematical concept, not just keywords.
- **Intent Router**: Automatically detects if you are asking a specific question (which triggers a strict threshold semantic search) or if you are asking for a full lore recap (which triggers a massive contextual summary).
- **Metadata Entity Boosting**: Leverages Watson NLU to identify named characters, organizations, and locations, granting massive relevance boosts when you ask about them by name.

## IBM Tech Stack
- **watsonx.ai**: Llama 3.1 for text generation & Slate 30m for vector embeddings.
- **Watson NLU**: NLP entity extraction for characters and keywords.
- **IBM Cloudant**: NoSQL document store holding the vector embeddings, text chunks, and extracted entities.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Set up your `.env` file with your IBM credentials (`IBM_CLOUD_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, `CLOUDANT_API_KEY`, `CLOUDANT_URL`, `NLU_API_KEY`, and `NLU_URL`).
5. Run the data ingestion script (only needed once to populate Cloudant): `python data_parser.py`
6. Start the server: `uvicorn app:app --host 0.0.0.0 --port 8000 --reload`
7. Open `http://localhost:8000` in your browser.
