from Agent.state import AgentState
from Agent.utils import generate_content_with_retry

def get_db_and_models():
    import app
    return app.db, app.embeddings_collection, app.conversations_collection, app.gemini_25_model

def detect_language_node(state: AgentState):
    _, _, _, gemini_model = get_db_and_models()

    query = state["original_query"]
    logs = list(state.get("logs", []))
    logs.append("Detecting language of the user query...")
    
    if not gemini_model:
        return {"detected_language": "English", "english_query": query, "logs": logs + ["Warning: Gemini model not configured."]}
        
    prompt = f"Identify the language of the following user query. Return only the language name (e.g., 'English', 'Spanish', 'French', 'Hindi', etc.) as a single word without any punctuation.\n\nQuery: {query}"
    try:
        response = generate_content_with_retry(gemini_model, prompt)
        lang = response.text.strip().capitalize()
        if not lang or len(lang) > 20:
            lang = "English"
    except Exception as e:
        print(f"Error detecting language: {e}")
        lang = "English"
        
    logs.append(f"Language detected: {lang}")
    
    english_query = query
    if lang != "English":
        trans_prompt = f"Translate the following text into English. Return only the translation, nothing else.\n\nText: {query}"
        try:
            response = generate_content_with_retry(gemini_model, trans_prompt)
            english_query = response.text.strip()
            logs.append(f"Translated query to English: '{english_query}'")
        except Exception as e:
            print(f"Error translating query to English: {e}")
            logs.append("Warning: Could not translate query to English. Using original query.")
            
    return {
        "detected_language": lang,
        "english_query": english_query,
        "logs": logs
    }
def translate_response_node(state: AgentState):
    _, _, _, gemini_model = get_db_and_models()

    lang = state.get("detected_language", "English")
    answer = state.get("answer", "")
    logs = list(state.get("logs", []))
    
    if lang != "English" and gemini_model:
        logs.append(f"Translating final response back to {lang}...")
        prompt = f"Translate the following response into {lang}. Preserve all URLs, citations, and markdown formatting.\n\nResponse:\n{answer}"
        try:
            response = generate_content_with_retry(gemini_model, prompt)
            answer = response.text
            logs.append(f"Translation completed.")
        except Exception as e:
            print(f"Error translating response: {e}")
            logs.append("Warning: Could not translate response back to original language.")
            
    return {
        "answer": answer,
        "logs": logs
    }
