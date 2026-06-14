from Agent.state import AgentState

from Agent.utils import generate_content_with_retry

def get_db_and_models():
    import app
    return app.db, app.embeddings_collection, app.conversations_collection, app.gemini_35_model

def decompose_query_node(state: AgentState):
    _, _, _, gemini_model = get_db_and_models()

    query = state.get("english_query") or state["original_query"]
    logs = list(state.get("logs", []))
    logs.append("Decomposing user query into sub-queries...")
    
    decomposed = [query]
    if gemini_model:
        prompt = (
            "You are a Query Decomposer. Analyze the user query and break it down into 1 to 3 distinct, simple search queries "
            "that can be searched independently in a document vector database or web search engine. "
            "If the query is already simple and cannot be broken down further, return a JSON array containing just the query.\n\n"
            "Output format: Return ONLY a valid JSON list of strings. Do not use markdown blocks.\n\n"
            f"Query: {query}"
        )
        try:
            response = generate_content_with_retry(gemini_model, prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            import json
            decomposed = json.loads(text)
            if not isinstance(decomposed, list):
                decomposed = [query]
        except Exception as e:
            print(f"Error decomposing query: {e}")
            decomposed = [query]
            
    logs.append(f"Decomposed into sub-queries: {decomposed}")
    return {
        "decomposed_queries": decomposed,
        "logs": logs
    }

def decomposer_approval_node(state: AgentState):
    logs = list(state.get("logs", []))
    logs.append("Decomposed queries approved/modified by user.")
    return {"logs": logs}
