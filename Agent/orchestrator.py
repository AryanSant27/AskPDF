from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from Agent.state import AgentState
from Agent.translation_agent import detect_language_node, translate_response_node
from Agent.decomposer_agent import decompose_query_node, decomposer_approval_node
from Agent.web_scraper_agent import run_web_search_node, web_search_approval_node
from Agent.synthesis_agent import vector_search_node, decide_web_search_node, synthesis_node

def should_decomposer_hitl(state: AgentState):
    options = state.get("options", {})
    if options.get("hitl_decomposer", True):
        return "decomposer_approval_gate"
    else:
        return "vector_search"

def should_web_search(state: AgentState):
    web_queries = state.get("web_queries", [])
    options = state.get("options", {})
    
    if not web_queries:
        return "synthesis"
        
    if options.get("hitl_web", True):
        return "web_search_approval_gate"
    else:
        return "run_web_search"

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("detect_language", detect_language_node)
    workflow.add_node("decompose_query", decompose_query_node)
    workflow.add_node("decomposer_approval_gate", lambda state: {"logs": state.get("logs", []) + ["Waiting for user approval of decomposed sub-queries..."]})
    workflow.add_node("decomposer_approval", decomposer_approval_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("decide_web_search", decide_web_search_node)
    workflow.add_node("web_search_approval_gate", lambda state: {"logs": state.get("logs", []) + ["Waiting for user approval of web search queries..."]})
    workflow.add_node("web_search_approval", web_search_approval_node)
    workflow.add_node("run_web_search", run_web_search_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("translate_response", translate_response_node)
    
    workflow.add_edge(START, "detect_language")
    workflow.add_edge("detect_language", "decompose_query")
    
    workflow.add_conditional_edges(
        "decompose_query",
        should_decomposer_hitl,
        {
            "decomposer_approval_gate": "decomposer_approval_gate",
            "vector_search": "vector_search"
        }
    )
    
    workflow.add_edge("decomposer_approval_gate", "decomposer_approval")
    workflow.add_edge("decomposer_approval", "vector_search")
    
    workflow.add_edge("vector_search", "decide_web_search")
    
    workflow.add_conditional_edges(
        "decide_web_search",
        should_web_search,
        {
            "web_search_approval_gate": "web_search_approval_gate",
            "run_web_search": "run_web_search",
            "synthesis": "synthesis"
        }
    )
    
    workflow.add_edge("web_search_approval_gate", "web_search_approval")
    workflow.add_edge("web_search_approval", "run_web_search")
    
    workflow.add_edge("run_web_search", "synthesis")
    workflow.add_edge("synthesis", "translate_response")
    workflow.add_edge("translate_response", END)
    
    memory = MemorySaver()
    
    graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=["decomposer_approval_gate", "web_search_approval_gate"]
    )
    return graph

compiled_graph = build_graph()
