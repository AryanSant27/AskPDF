from Agent.state import AgentState
from bson.objectid import ObjectId
from Agent.utils import generate_content_with_retry

def get_db_and_models():
    import app
    return app.db, app.embeddings_collection, app.conversations_collection, app.gemini_31_model

def vector_search_node(state: AgentState):
    db, embeddings_collection, _, _ = get_db_and_models()
    from app import generate_embedding
    
    queries = state.get("decomposed_queries", [])
    pdf_id = state.get("pdf_id")
    logs = list(state.get("logs", []))
    logs.append(f"Searching PDF vector database for {len(queries)} sub-queries...")
    
    retrieved_texts = []
    seen_chunks = set()
    
    for idx, q in enumerate(queries):
        logs.append(f"Vector search for sub-query {idx+1}: '{q}'")
        try:
            # Generate embedding using the Google Embeddings API via app.py
            embedding = generate_embedding(q)
            if not embedding:
                continue

                
            pipeline = [
                {
                    "$vectorSearch": {
                        "queryVector": embedding,
                        "path": "embedding",
                        "numCandidates": 100,
                        "limit": 3,
                        "index": "pdf_embeddings_index"
                    }
                },
                {
                    "$match": {"pdf_id": ObjectId(pdf_id)}
                },
                {
                    "$project": {
                        "_id": 0,
                        "text": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            chunks = list(embeddings_collection.aggregate(pipeline))
            for chunk in chunks:
                text = chunk.get("text", "")
                if text and text not in seen_chunks:
                    seen_chunks.add(text)
                    retrieved_texts.append(text)
        except Exception as e:
            print(f"Vector search error for query '{q}': {e}")
            logs.append(f"Warning: Vector search query failed for '{q}'. Error: {e}")
            
    combined_context = "\n\n".join(retrieved_texts)
    logs.append(f"Retrieved {len(seen_chunks)} relevant chunks from PDF.")
    return {
        "vector_context": combined_context,
        "logs": logs
    }

def decide_web_search_node(state: AgentState):
    _, _, _, gemini_model = get_db_and_models()
    original_query = state.get("english_query") or state["original_query"]
    context = state.get("vector_context", "")
    logs = list(state.get("logs", []))
    logs.append("Evaluating retrieved PDF context for completeness...")
    
    web_needed = False
    web_queries = []
    
    if gemini_model:
        prompt = (
            "You are an information completeness analyzer. Analyze the user's query and the retrieved context from the PDF document.\n"
            "Determine if the PDF context contains enough information to answer the query, or if we need external current information from the web (e.g., current stock prices, recent news, external definitions or general facts not present in the document context).\n\n"
            "Output format: Return ONLY a valid JSON object with the following fields:\n"
            "{\n"
            "  \"web_needed\": true or false,\n"
            "  \"queries\": [\"search query 1\", \"search query 2\"] (only if web_needed is true, otherwise empty list)\n"
            "}\n\n"
            f"User Query: {original_query}\n\n"
            f"Retrieved PDF Context:\n{context[:3000]}"
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
            data = json.loads(text)
            web_needed = data.get("web_needed", False)
            web_queries = data.get("queries", [])
        except Exception as e:
            print(f"Error deciding web search: {e}")
            web_needed = False
            
    logs.append(f"Web search decision: needed={web_needed}, queries={web_queries}")
    return {
        "web_queries": web_queries if web_needed else [],
        "logs": logs
    }

def synthesis_node(state: AgentState):
    _, _, conversations_collection, gemini_model = get_db_and_models()
    
    original_query = state.get("english_query") or state["original_query"]
    vector_context = state.get("vector_context", "")
    scraped_data = state.get("scraped_data", [])
    conversation_id = state.get("conversation_id")
    logs = list(state.get("logs", []))
    
    logs.append("Synthesizing final answer from retrieved contexts...")
    
    chat_history = []
    if conversation_id:
        try:
            conversation = conversations_collection.find_one({"_id": ObjectId(conversation_id)})
            if conversation:
                chat_history = conversation.get('history', [])
        except Exception as e:
            print(f"Error retrieving history: {e}")
            
    web_context = ""
    if scraped_data:
        web_context_parts = []
        for idx, item in enumerate(scraped_data):
            web_context_parts.append(f"Source [{idx+1}]: {item['title']} ({item['url']})\nContent: {item['text']}")
        web_context = "\n\n".join(web_context_parts)
        
    prompt = f"""You are a helpful and detailed RAG AI assistant. Your task is to answer the user's query using the provided context and conversation history.

You have access to two sources of context:
1. Context extracted from a local PDF document.
2. Context scraped from web searches (if needed).

Answer the user's query thoroughly, drawing from both contexts if relevant.
When you use facts from the context, make sure to explicitly cite the source (e.g., cite the PDF if the info comes from PDF, or cite the Web page title/URL if it comes from the web).

PDF Context:
{vector_context}

Web Context:
{web_context}

Conversation History:
{chat_history}

User Query: {original_query}

Answer:"""

    answer = "No response generated."
    if gemini_model:
        try:
            response = generate_content_with_retry(gemini_model, prompt)
            answer = response.text
        except Exception as e:
            print(f"Error in synthesis: {e}")
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                answer = (
                    "⚠️ **Rate Limit Exceeded**: The Gemini API rate limits or free tier quota has been exhausted. "
                    "Please wait a few seconds and try resubmitting your query.\n\n"
                    "*Note: In Agentic RAG mode, a single query runs multiple LLM calls sequentially (Translation, Decomposition, Search Decision, and Synthesis), "
                    "which can trigger free tier API rate limits.*"
                )
            else:
                answer = f"An error occurred while generating the answer. Details: {str(e)}"
            
    logs.append("Final answer synthesized successfully.")
    return {
        "answer": answer,
        "logs": logs
    }
