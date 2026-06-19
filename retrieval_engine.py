import os
from dotenv import load_dotenv
from ibmcloudant.cloudant_v1 import CloudantV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watsonx_ai.foundation_models import Embeddings
import numpy as np

# Load environment variables
load_dotenv()

DATABASE_NAME = "mr_robot_s01"

def get_cloudant_client():
    """Returns a connected Cloudant client."""
    cloudant_key = os.getenv("CLOUDANT_API_KEY")
    cloudant_url = os.getenv("CLOUDANT_URL")
    
    if not cloudant_key or not cloudant_url:
        raise ValueError("Cloudant credentials missing in .env")
        
    authenticator = IAMAuthenticator(cloudant_key)
    client = CloudantV1(authenticator=authenticator)
    client.set_service_url(cloudant_url)
    return client

def get_watsonx_embeddings():
    """Initializes and returns the watsonx.ai Embeddings client."""
    api_key = os.getenv("IBM_CLOUD_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL")
    
    if not api_key or not project_id or not url:
        raise ValueError("watsonx.ai credentials missing in .env")
        
    return Embeddings(
        model_id="ibm/slate-30m-english-rtrvr-v2",
        credentials={"url": url, "apikey": api_key},
        project_id=project_id
    )

def parse_time_str_to_seconds(time_str):
    """Parses HH:MM:SS or MM:SS to total seconds."""
    parts = time_str.strip().split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    else:
        return float(parts[0])

def retrieve_safe_chunks(user_pause_time_sec):
    """
    Executes the Spoiler Shield by querying Cloudant.
    Retrieves ONLY chunks that started before or at the user's current pause time.
    """
    client = get_cloudant_client()
    
    # Selector: get all chunks for this episode that have start_time_sec <= user_pause_time_sec
    # This physically excludes any future chunks from entering the pool.
    selector = {
        "episode": {"$eq": 1},
        "start_time_sec": {"$lte": user_pause_time_sec}
    }
    
    response = client.post_find(
        db=DATABASE_NAME,
        selector=selector,
        limit=100  # Episode is max 65 mins, so 100 limit covers everything
    ).get_result()
    
    safe_chunks = response.get('docs', [])
    print(f"[Spoiler Shield] Locked database query. Retained {len(safe_chunks)} safe chunks out of 65.")
    return safe_chunks

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def calculate_relevance_score(query_embedding, chunk_embedding, query_lower, entities):
    """Calculates a true semantic score using cosine similarity, plus a metadata bonus."""
    if not chunk_embedding or not query_embedding:
        return 0.0
        
    # 1. Base semantic score (cosine similarity)
    # Cosine similarity is usually between -1 and 1
    score = cosine_similarity(query_embedding, chunk_embedding)
    
    # Scale it up slightly so it's easier to read (e.g. 0.82 => 82)
    score *= 100.0
    
    # 2. Entity matching bonus (High priority metadata)
    for entity in entities:
        if entity.lower() in query_lower:
            # Significant boost if a specific entity (like "Gideon") is requested
            score += 15.0
            
    return score

def classify_intent(user_query):
    """Classify the user's intent as either 'SUMMARIZE' or 'QUESTION'."""
    from generation_engine import get_watsonx_model
    
    prompt = f"""Classify the following user query as either 'SUMMARIZE' (asking for lore, summary, recap) or 'QUESTION' (specific detail).
Query: "{user_query}"
Respond with ONLY the word SUMMARIZE or QUESTION.
Classification:"""
    try:
        model = get_watsonx_model(max_tokens=10)
        response = model.generate(prompt=prompt)
        intent = response['results'][0]['generated_text'].strip().upper()
        if "SUMMARIZE" in intent:
            return "SUMMARIZE"
        return "QUESTION"
    except Exception as e:
        print(f"[Intent Classifier Error] {e}")
        return "QUESTION"

def get_rag_context(user_query, user_pause_time_sec):
    """
    Implements the Dynamic RAG Pipeline:
    1. Triggers the Spoiler Shield to drop future data.
    2. Separates the current active minute (Immediate Context).
    3. Finds the top 3 historically relevant chunks via hybrid scoring.
    """
    # 1. Enforce the Spoiler Shield
    safe_chunks = retrieve_safe_chunks(user_pause_time_sec)
    
    if not safe_chunks:
        return {
            "immediate": None,
            "historical": [],
            "combined_context": "No watched history available for this timestamp.",
            "intent": "UNKNOWN"
        }
        
    intent = classify_intent(user_query)
    print(f"[Intent Classifier] Query intent detected as: {intent}")
    
    immediate_chunk = None
    historical_pool = []
    
    # 3. Partition chunks into Immediate and Historical
    for chunk in safe_chunks:
        start = chunk['start_time_sec']
        end = chunk['end_time_sec']
        
        # Check if this chunk is what the user is currently watching
        if start <= user_pause_time_sec < end:
            immediate_chunk = chunk
        else:
            historical_pool.append(chunk)

    top_historical = []

    if intent == 'SUMMARIZE':
        print("[Intent Classifier] SUMMARIZE intent detected. Skipping vector search and returning all historical context.")
        top_historical = historical_pool.copy()
    else:
        # 2. Embed the user query for semantic search
        print(f"Generating query embedding for: '{user_query}'...")
        embeddings_client = get_watsonx_embeddings()
        query_embedding = []
        try:
            query_embedding = embeddings_client.embed_documents([user_query])[0]
        except Exception as e:
            print(f"[Embedding Error] Failed to embed query: {e}")
            
        # 4. Score and rank historical chunks using Semantic Search
        scored_history = []
        query_lower = user_query.lower()
        
        for chunk in historical_pool:
            score = calculate_relevance_score(
                query_embedding, 
                chunk.get('embedding', []),
                query_lower,
                chunk.get('entities', [])
            )
            scored_history.append((score, chunk))
            
        # Sort by score descending
        scored_history.sort(key=lambda x: x[0], reverse=True)
        
        print("\n--- [RAG Semantic Search Scores] ---")
        for score, chunk in scored_history:
            if score >= 60.0:
                min_start = chunk['start_time_sec'] // 60
                print(f"Minute {min_start:02d} | Score: {score:.2f} | Entities: {chunk.get('entities')}")
                top_historical.append(chunk)
        print("---------------------------\n")

    # 4. Formulate the clean prompt context
    context_lines = []
    
    if immediate_chunk:
        min_start = immediate_chunk['start_time_sec'] // 60
        context_lines.append(f"=== IMMEDIATE SCENE CONTEXT (MINUTE {min_start}) ===")
        context_lines.append(f"[Minute {min_start}]: {immediate_chunk['text']}")
    else:
        context_lines.append("=== IMMEDIATE SCENE CONTEXT (NO DIALOGUE) ===")
        context_lines.append("(No dialogue in the immediate past minute)")
        
    context_lines.append("\n=== HISTORICAL CONTEXT ===")
    if top_historical:
        # Sort historical chunks chronologically for logical LLM consumption
        top_historical.sort(key=lambda x: x['start_time_sec'])
        for chunk in top_historical:
            min_start = chunk['start_time_sec'] // 60
            context_lines.append(f"[Minute {min_start}]: {chunk['text']}")
    else:
        context_lines.append("(No relevant historical context found)")
        
    combined_context = "\n".join(context_lines)
    return {
        "immediate": immediate_chunk,
        "historical": top_historical,
        "combined_context": combined_context,
        "intent": intent
    }

if __name__ == "__main__":
    # Test connection and local scoring logic
    print("Testing Retrieval Engine...")
    test_query = "Who is Rohit and what is his website?"
    test_pause_time = "02:30"  # 150 seconds
    
    pause_sec = parse_time_str_to_seconds(test_pause_time)
    print(f"Simulating video paused at {test_pause_time} ({pause_sec} seconds)")
    
    result = get_rag_context(test_query, pause_sec)
    print("Combined Context sent to LLM:")
    print("====================================================")
    print(result['combined_context'])
    print("====================================================")
