import os
import json
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from retrieval_engine import tool_search_dialogue, tool_get_full_summary, tool_get_character_profile

load_dotenv()

# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# These are given to Llama 3.1 so it can decide autonomously
# which tool to call based on the user's question.
# ─────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_dialogue",
            "description": "Semantically searches the Mr. Robot episode dialogue database for chunks relevant to a specific factual question about events, plot points, or themes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant dialogue"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_summary",
            "description": "Returns all watched dialogue in chronological order. Use this when the user asks for a recap, summary, lore explanation, or what happened so far.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_character_profile",
            "description": "Retrieves all dialogue chunks mentioning a specific character by name. Use this when the user asks specifically about a character like Elliot, Angela, or Mr. Robot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_name": {
                        "type": "string",
                        "description": "The name of the character to look up"
                    }
                },
                "required": ["character_name"]
            }
        }
    }
]


def get_watsonx_model(max_tokens=600):
    """Initializes and returns the watsonx.ai ModelInference client."""
    api_key = os.getenv("IBM_CLOUD_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL")

    if not api_key or not project_id or not url:
        raise ValueError("watsonx.ai credentials missing in .env")

    model = ModelInference(
        model_id="meta-llama/llama-3-3-70b-instruct",
        credentials={"url": url, "apikey": api_key},
        project_id=project_id,
        params={
            GenParams.MAX_NEW_TOKENS: max_tokens,
            GenParams.TEMPERATURE: 0.2,
            GenParams.TOP_P: 0.9
        }
    )
    return model


def execute_tool(tool_name: str, tool_args: dict, timestamp: float) -> str:
    """Executes the tool chosen by the agent and returns the result as a string."""
    print(f"[Agent] Executing tool: {tool_name}({tool_args})")

    if tool_name == "search_dialogue":
        return tool_search_dialogue(tool_args.get("query", ""), timestamp)
    elif tool_name == "get_full_summary":
        return tool_get_full_summary(timestamp)
    elif tool_name == "get_character_profile":
        return tool_get_character_profile(tool_args.get("character_name", ""), timestamp)

    return "Unknown tool called."


def generate_narrative_response(user_query: str, user_pause_time_sec: float):
    """
    Agentic RAG Pipeline:
    Llama 3.1 autonomously decides which tool(s) to call,
    retrieves the spoiler-safe context, then generates the final answer.
    The Spoiler Shield is enforced inside every tool function.
    """
    mins = int(user_pause_time_sec // 60)
    secs = int(user_pause_time_sec % 60)
    pause_time_str = f"{mins:02d}:{secs:02d}"

    system_message = f"""You are the AI Narrative Continuity Companion for "Mr. Robot" Season 1, Episode 1.
The viewer has paused at {pause_time_str}. You have access to tools that query a spoiler-safe database.

Rules:
1. You MUST call a tool first before answering — never answer from memory.
2. Answer ONLY using the content returned by the tool.
3. If the tool returns nothing relevant, say exactly: "You haven't seen that yet! Keep watching."
4. Keep answers concise unless the user asks for a summary."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]

    model = get_watsonx_model()
    tool_context = ""

    # Agent loop — max 3 iterations to prevent infinite loops
    for iteration in range(3):
        print(f"[Agent] Loop iteration {iteration + 1}")
        response = model.chat(messages=messages, tools=TOOLS, tool_choice="auto")
        choice = response["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        message = choice["message"]

        # If the model wants to call a tool
        if finish_reason == "tool_calls" and message.get("tool_calls"):
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": message["tool_calls"]
            })

            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                except Exception:
                    tool_args = {}

                tool_result = execute_tool(tool_name, tool_args, user_pause_time_sec)
                tool_context = tool_result

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result
                })

        # Model is done — return final answer
        elif finish_reason == "stop":
            answer = message.get("content", "").strip()
            print(f"[Agent] Final answer generated after {iteration + 1} iteration(s).")
            return answer, tool_context

        else:
            break

    return "I was unable to retrieve context for your question. Please try again.", tool_context
