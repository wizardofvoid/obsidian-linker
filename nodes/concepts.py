import json
from core.state import AgentState, ConceptExtractionOutput
from core.utils import invoke_with_retry
from core.config import LLM_EXTRACTION, LLM_VERIFICATION
from core.prompts import prompt_concept_extraction, prompt_concept_verification

def concept_extractor(state: AgentState):
    print("\n--- [Concept Extractor] ---")
    all_extracted_concepts = []

    for note in state.get("new_notes", []):
        try: 
            print(f"Extracting raw concepts from: {note['title']}...")
            extracted_output = invoke_with_retry(prompt_concept_extraction, LLM_EXTRACTION, ConceptExtractionOutput, {
                "text": note["content"]
            })

            for concept in extracted_output.concepts:
                concept_dict = concept.model_dump()
                concept_dict["source_note"] = note["title"]
                all_extracted_concepts.append(concept_dict)
                
        except Exception as e:
            print(f"Failed to extract concepts from '{note['title']}': {e}")
            continue
            
    print(f"Total raw concepts extracted: {len(all_extracted_concepts)}")
    return {"raw_concepts": all_extracted_concepts}

def concept_verifier(state: AgentState):
    print("\n--- [Concept Verifier] ---")
    all_verified_concepts = []

    for note in state.get("new_notes", []):
        try:
            note_raw_concepts = [c for c in state.get("raw_concepts", []) if c.get("source_note") == note["title"]]
            if not note_raw_concepts:
                continue
                
            print(f"Verifying concepts for: {note['title']}...")
            concepts_json = json.dumps(note_raw_concepts)
            verified_output = invoke_with_retry(prompt_concept_verification, LLM_VERIFICATION, ConceptExtractionOutput, {
                "text": note["content"], 
                "concepts": concepts_json
            })

            for concept in verified_output.concepts:
                concept_dict = concept.model_dump()
                concept_dict["source_note"] = note["title"]
                all_verified_concepts.append(concept_dict)
                
        except Exception as e:
            print(f"Failed to verify concepts from '{note['title']}': {e}")
            continue
            
    print(f"Total verified concepts from new notes: {len(all_verified_concepts)}")
    return {
        "new_concepts": all_verified_concepts,
        "concepts": state.get("concepts", []) + all_verified_concepts
    }
