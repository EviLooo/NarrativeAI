import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from retrieval_engine import get_rag_context

# Load environment variables
load_dotenv()

def get_watsonx_model(max_tokens=150):
    """Initializes and returns the watsonx.ai ModelInference client."""
    api_key = os.getenv("IBM_CLOUD_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL")
    
    if not api_key or not project_id or not url:
        raise ValueError("watsonx.ai credentials missing in .env")
        
    credentials = {
        "url": url,
        "apikey": api_key
    }
    
    # We use meta-llama/llama-3-3-70b-instruct as it has full text generation support and excellent reasoning
    model_id = "meta-llama/llama-3-3-70b-instruct"
    
    # Simple generation parameters for factual, concise RAG answers
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.TEMPERATURE: 0.2,  # Low temperature for high accuracy/factuality
        GenParams.TOP_P: 0.9
    }
    
    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=generate_params
    )
    return model

def generate_narrative_response(user_query, user_pause_time_sec):
    """
    Retrieves safe context based on timestamp and generates a spoiler-safe answer 
    using watsonx.ai.
    """
    # 1. Fetch filtered context using the retrieval engine (Spoiler Shield)
    rag_result = get_rag_context(user_query, user_pause_time_sec)
    context_text = rag_result["combined_context"]
    intent = rag_result.get("intent", "QUESTION")
    
    # Convert seconds to a human-readable timestamp (MM:SS)
    mins = int(user_pause_time_sec // 60)
    secs = int(user_pause_time_sec % 60)
    pause_time_str = f"{mins:02d}:{secs:02d}"
    
    # 2. Construct the strict prompt
    if intent == 'SUMMARIZE':
        system_prompt = f"""You are the AI Narrative Continuity Companion, helping a viewer watch "Mr. Robot" Season 1, Episode 1.
The viewer has paused the video at timestamp {pause_time_str}.

Here is the dialogue from the show that the viewer has watched so far (up to {pause_time_str}). Any events, characters, or details outside this dialogue have NOT happened yet and are considered SPOILERS:
---
{context_text}
---

Your task is to answer the viewer's request: "{user_query}"

Strict Rules:
1. Provide a cohesive, detailed narrative recap or lore explanation based ONLY on the dialogue context provided above.
2. If the request refers to future events, characters, or plot points that occur after {pause_time_str}, you MUST NOT reveal them. Instead, say exactly: "You haven't seen that yet! Keep watching."
3. Speak directly to the viewer in a helpful, conversational, yet spoiler-safe tone.
4. You are receiving a large block of chronological events. Write a comprehensive response matching the user's request.

Answer:"""
        max_tokens = 500
    else:
        system_prompt = f"""You are the AI Narrative Continuity Companion, helping a viewer watch "Mr. Robot" Season 1, Episode 1.
The viewer has paused the video at timestamp {pause_time_str}.

Here is the ONLY dialogue from the show that the viewer has watched so far (up to {pause_time_str}). Any events, characters, or details outside this dialogue have NOT happened yet and are considered SPOILERS:
---
{context_text}
---

Your task is to answer the viewer's question: "{user_query}"

Strict Rules:
1. Answer the question using ONLY the dialogue context provided above.
2. If the answer is not in the context, or if the question refers to future events, characters, or plot points that occur after {pause_time_str}, you MUST NOT reveal them. Instead, say exactly: "You haven't seen that yet! Keep watching."
3. Speak directly to the viewer in a helpful, conversational, yet spoiler-safe tone.
4. Keep your answer brief (1-3 sentences maximum).

Answer:"""
        max_tokens = 150

    # 3. Call the model
    try:
        model = get_watsonx_model(max_tokens=max_tokens)
        response = model.generate(prompt=system_prompt)
        answer = response['results'][0]['generated_text'].strip()
        return answer, context_text
    except Exception as e:
        return f"Error generating response: {e}", context_text

if __name__ == "__main__":
    print("Testing Generation Engine...")
    
    # Test case 1: Question about an event that has happened (Safe)
    q1 = "Who is Rohit and what did Elliot find out about him?"
    t1 = 150.0  # 2:30 (Rohit has been introduced and discussed)
    print(f"\nQuery: '{q1}' at paused time {t1} seconds.")
    ans1, ctx1 = generate_narrative_response(q1, t1)
    print(f"Response:\n{ans1}")
    print("-" * 50)
    
    # Test case 2: Question about a future character / event (Should trigger Spoiler Shield)
    # Mr. Robot is introduced around minute 4. If paused at minute 2 (120 sec), he is a spoiler.
    q2 = "Who is Mr. Robot?"
    t2 = 120.0  # 2:00
    print(f"\nQuery: '{q2}' at paused time {t2} seconds (Prior to introduction).")
    ans2, ctx2 = generate_narrative_response(q2, t2)
    print(f"Response:\n{ans2}")
    print("=========================================")
