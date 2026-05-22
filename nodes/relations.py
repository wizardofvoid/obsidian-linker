import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from core.state import AgentState, RelationshipExtractionOutput
from core.utils import invoke_with_retry, minify_concepts
from core.config import LLM_RELATIONSHIP, LLM_VERIFICATION, embeddings
from core.prompts import prompt_relation, prompt_relation_verify

def relationship_extractor(state: AgentState):
    print("\n--- [Relationship Extractor] ---")
    new_concepts = state.get("new_concepts", [])
    all_concepts = state.get("concepts", [])
    
    if not new_concepts:
        print("No new concepts extracted. Skipping relationship extraction.")
        return {"raw_links": [], "retry_count": state.get("retry_count", 0) + 1}
    
    try:
        # Load FAISS index
        directory_path = state.get("dir", "")
        faiss_path = Path(directory_path) / ".linker_faiss_index"
        
        retrieved_concepts = []
        if faiss_path.exists():
            vectorstore = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
            for new_concept in new_concepts:
                query = f"{new_concept['concept_name']}: {new_concept['explanation']}"
                # Retrieve top 10 relevant concepts per new concept
                results = vectorstore.similarity_search(query, k=10)
                
                for doc in results:
                    name = doc.metadata.get("name")
                    note = doc.metadata.get("note")
                    # Find the full concept dict in all_concepts
                    match = next((c for c in all_concepts if c["concept_name"] == name and c["source_note"] == note), None)
                    if match and match not in retrieved_concepts and match not in new_concepts:
                        retrieved_concepts.append(match)
        else:
            print("No FAISS index found. Falling back to all existing concepts.")
            retrieved_concepts = [c for c in all_concepts if c not in new_concepts]
            
        existing_concepts = retrieved_concepts
        print(f"Evaluating {len(new_concepts)} new concepts against {len(existing_concepts)} retrieved historical concepts...")
        
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
