from langgraph.graph import StateGraph, END
from core.state import AgentState

# We must import config first to ensure env vars/keys are loaded
import core.config 

from nodes.io import vault_reader, link_writer, summary_reporter
from nodes.concepts import concept_extractor, concept_verifier
from nodes.relations import relationship_extractor, relationship_verifier
from nodes.quality import quality_checker, should_retry

# --- Build graph ---
graph = StateGraph(AgentState)

graph.add_node("vault_reader",           vault_reader)
graph.add_node("concept_extractor",      concept_extractor)
graph.add_node("concept_verifier",       concept_verifier)
graph.add_node("relationship_extractor", relationship_extractor)
graph.add_node("relationship_verifier",  relationship_verifier)
graph.add_node("quality_checker",        quality_checker)
graph.add_node("link_writer",            link_writer)
graph.add_node("summary_reporter",       summary_reporter)

graph.set_entry_point("vault_reader")

graph.add_edge("vault_reader",           "concept_extractor")
graph.add_edge("concept_extractor",      "concept_verifier")
graph.add_edge("concept_verifier",       "relationship_extractor")
graph.add_edge("relationship_extractor", "relationship_verifier")
graph.add_edge("relationship_verifier",  "quality_checker")
graph.add_conditional_edges("quality_checker", should_retry, {
    "concept_extractor": "concept_extractor",
    "relationship_extractor": "relationship_extractor",
    "link_writer": "link_writer"
})
graph.add_edge("link_writer",            "summary_reporter")
graph.add_edge("summary_reporter",       END)

app = graph.compile()

# --- Run it ---
if __name__ == "__main__":
    result = app.invoke({
        "notes": [], 
        "new_notes": [],
        "raw_concepts": [],
        "concepts": [], 
        "new_concepts": [],
        "raw_links": [],
        "links": [],
        "quality_score": 0.0, 
        "retry_count": 0,
        "dir": "",
        "retry_reason": None,
        "cache_path": "",
        "file_mtimes": {}
    })