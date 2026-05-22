import json
from pathlib import Path
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
from core.state import AgentState, ConceptExtractionOutput
from core.utils import invoke_with_retry
from core.config import LLM_EXTRACTION, LLM_VERIFICATION, embeddings
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
            
    if all_verified_concepts:
        directory_path = state.get("dir", "")
        faiss_path = Path(directory_path) / ".linker_faiss_index"
        
        docs = [
            Document(
                page_content=f"{c['concept_name']}: {c['explanation']}",
                metadata={"name": c['concept_name'], "note": c['source_note']}
            ) for c in all_verified_concepts
        ]
        
        print(f"Indexing {len(docs)} new concepts into FAISS...")
        
        def add_safely(store, documents):
            for i, d in enumerate(documents):
                store.add_documents([d])
                if (i + 1) % 10 == 0 or (i + 1) == len(documents):
                    print(f"  Indexed {i+1}/{len(documents)}...")

        if faiss_path.exists():
            try:
                vectorstore = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
                add_safely(vectorstore, docs)
                vectorstore.save_local(str(faiss_path))
            except Exception as e:
                print(f"Failed to load existing FAISS index. Rebuilding: {e}")
                vectorstore = FAISS.from_documents([docs[0]], embeddings)
                if len(docs) > 1:
                    add_safely(vectorstore, docs[1:])
                vectorstore.save_local(str(faiss_path))
        else:
            vectorstore = FAISS.from_documents([docs[0]], embeddings)
            if len(docs) > 1:
                add_safely(vectorstore, docs[1:])
            vectorstore.save_local(str(faiss_path))
            
    print(f"Total verified concepts from new notes: {len(all_verified_concepts)}")
    return {
        "new_concepts": all_verified_concepts,
        "concepts": state.get("concepts", []) + all_verified_concepts
    }
