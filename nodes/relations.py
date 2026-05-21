import json
from core.state import AgentState, RelationshipExtractionOutput
from core.utils import invoke_with_retry, minify_concepts
from core.config import LLM_RELATIONSHIP, LLM_VERIFICATION
from core.prompts import prompt_relation, prompt_relation_verify

def relationship_extractor(state: AgentState):
    print("\n--- [Relationship Extractor] ---")
    new_concepts = state.get("new_concepts", [])
    all_concepts = state.get("concepts", [])
    
    if not new_concepts:
        print("No new concepts extracted. Skipping relationship extraction.")
        return {"raw_links": [], "retry_count": state.get("retry_count", 0) + 1}
    
    try:
        existing_concepts = [c for c in all_concepts if c not in new_concepts]
        print(f"Extracting relationships involving {len(new_concepts)} new concepts vs {len(existing_concepts)} existing concepts...")
        
        minified_new = minify_concepts(new_concepts)
        minified_existing = minify_concepts(existing_concepts)
        
        extracted_output = invoke_with_retry(prompt_relation, LLM_RELATIONSHIP, RelationshipExtractionOutput, {
            "new_concepts": json.dumps(minified_new),
            "existing_concepts": json.dumps(minified_existing)
        })
        
        relations_list = [link.model_dump() for link in extracted_output.links]
        print(f"Total raw cross-note relationships extracted: {len(relations_list)}")
        return {"raw_links": relations_list, "retry_count": state.get("retry_count", 0) + 1}
    except Exception as e:
        print(f"Failed to extract cross-note relationships: {e}")
        return {"raw_links": [], "retry_count": state.get("retry_count", 0) + 1}

def relationship_verifier(state: AgentState):
    print("\n--- [Relationship Verifier] ---")
    raw_links = state.get("raw_links", [])
    concepts = state.get("concepts", [])
    
    if not raw_links or not concepts:
        print("No raw links or concepts to verify.")
        return {"links": state.get("links", [])}
    
    try:
        minified_concepts = minify_concepts(concepts)
        links_json = json.dumps(raw_links)
        print(f"Verifying {len(raw_links)} cross-note relationships...")
        
        verified_output = invoke_with_retry(prompt_relation_verify, LLM_VERIFICATION, RelationshipExtractionOutput, {
            "concepts": json.dumps(minified_concepts),
            "relationships": links_json
        })
        
        relations_list = [link.model_dump() for link in verified_output.links]
        print(f"Total verified cross-note relationships: {len(relations_list)}")
        return {"links": state.get("links", []) + relations_list}
    except Exception as e:
        print(f"Failed to verify relationships: {e}")
        return {"links": state.get("links", [])}
