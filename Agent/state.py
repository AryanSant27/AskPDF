from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    session_id: str
    user: str
    pdf_id: str
    conversation_id: Optional[str]
    original_query: str
    english_query: str
    detected_language: str
    current_step: str
    decomposed_queries: List[str]
    web_queries: List[str]
    vector_context: str
    scraped_data: List[Dict[str, str]]
    logs: List[str]
    options: Dict[str, bool]
    answer: str
