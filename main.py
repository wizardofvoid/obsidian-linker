from langgraph.graph import StateGraph, END
from core.state import AgentState

# We must import config first to ensure env vars/keys are loaded
import core.config 

from nodes.io import vault_reader, link_writer, summary_reporter
from nodes.concepts import concept_extractor, concept_verifier
from nodes.relations import relationship_extractor, relationship_verifier

# --- Build graph ---
graph = StateGraph(AgentState)

graph.add_node("vault_reader",           vault_reader)
graph.add_node("concept_extractor",      concept_extractor)
graph.add_node("concept_verifier",       concept_verifier)
graph.add_node("relationship_extractor", relationship_extractor)
graph.add_node("relationship_verifier",  relationship_verifier)
graph.add_node("link_writer",            link_writer)
graph.add_node("summary_reporter",       summary_reporter)

graph.set_entry_point("vault_reader")

graph.add_edge("vault_reader",           "concept_extractor")
graph.add_edge("concept_extractor",      "concept_verifier")
graph.add_edge("concept_verifier",       "relationship_extractor")
graph.add_edge("relationship_extractor", "relationship_verifier")
graph.add_edge("relationship_verifier",  "link_writer")
graph.add_edge("link_writer",            "summary_reporter")
graph.add_edge("summary_reporter",       END)

app = graph.compile()

# --- Run it ---
if __name__ == "__main__":
    import asyncio
    import sys
    import os
    
    # Resolve directory from CLI arguments or environment variable
    vault_dir = r"C:\Users\saraf\Documents\VOID"
    if "--dir" in sys.argv:
        try:
            idx = sys.argv.index("--dir")
            if idx + 1 < len(sys.argv):
                vault_dir = sys.argv[idx + 1]
        except ValueError:
            pass
    elif "OBSIDIAN_VAULT_DIR" in os.environ:
        vault_dir = os.environ["OBSIDIAN_VAULT_DIR"]
        
    async def main():
        result = await app.ainvoke({
            "notes": [], 
            "new_notes": [],
            "raw_concepts": [],
            "concepts": [], 
            "new_concepts": [],
            "raw_links": [],
            "links": [],
            "dir": vault_dir,
            "cache_path": "",
            "file_hashes": {},
            "raw_tags_by_note": {},
            "tags_by_note": {}
        })
        return result
        
    asyncio.run(main())