import json
from pathlib import Path
from langchain_core.documents import Document
from core.state import AgentState, ConceptExtractionOutput
from core.utils import invoke_with_retry, abatch_invoke_with_retry
from core.config import LLM_EXTRACTION, LLM_VERIFICATION, embeddings
from core.prompts import prompt_concept_extraction, prompt_concept_verification

async def concept_extractor(state: AgentState):
    print("\n--- [Concept Extractor] ---")
    all_extracted_concepts = []
    raw_tags_by_note = state.get("raw_tags_by_note", {})

    new_notes = state.get("new_notes", [])
    if not new_notes:
        print("No new notes to extract concepts from.")
        return {"raw_concepts": [], "raw_tags_by_note": raw_tags_by_note}

    print(f"Extracting raw concepts from {len(new_notes)} notes concurrently...")
    inputs_list = [{"text": note["content"]} for note in new_notes]
    
    results = await abatch_invoke_with_retry(prompt_concept_extraction, LLM_EXTRACTION, ConceptExtractionOutput, inputs_list)
    
    for note, result in zip(new_notes, results):
        if isinstance(result, Exception):
            print(f"Failed to extract concepts from '{note['title']}': {result}")
            continue
            
        for concept in result.concepts:
            concept_dict = concept.model_dump()
            concept_dict["source_note"] = note["title"]
            all_extracted_concepts.append(concept_dict)
            
        if result.tags:
            raw_tags_by_note[note["title"]] = result.tags

    print(f"Total raw concepts extracted: {len(all_extracted_concepts)}")
    return {"raw_concepts": all_extracted_concepts, "raw_tags_by_note": raw_tags_by_note}

async def concept_verifier(state: AgentState):
    print("\n--- [Concept Verifier] ---")
    all_verified_concepts = []
    tags_by_note = state.get("tags_by_note", {})

    new_notes = state.get("new_notes", [])
    if not new_notes:
        return {"new_concepts": [], "concepts": state.get("concepts", []), "tags_by_note": tags_by_note}

    notes_with_concepts = []
    inputs_list = []

    for note in new_notes:
        note_raw_concepts = [c for c in state.get("raw_concepts", []) if c.get("source_note") == note["title"]]
        if not note_raw_concepts:
            continue
        
        notes_with_concepts.append(note)
        concepts_json = json.dumps(note_raw_concepts)
        tags_json = json.dumps(state.get("raw_tags_by_note", {}).get(note["title"], []))
        
        inputs_list.append({
            "text": note["content"], 
            "concepts": concepts_json,
            "tags": tags_json
        })

    if not inputs_list:
        print("No raw concepts found to verify.")
        return {"new_concepts": [], "concepts": state.get("concepts", []), "tags_by_note": tags_by_note}

    print(f"Verifying concepts for {len(inputs_list)} notes concurrently...")
    results = await abatch_invoke_with_retry(prompt_concept_verification, LLM_VERIFICATION, ConceptExtractionOutput, inputs_list)

    for note, result in zip(notes_with_concepts, results):
        if isinstance(result, Exception):
            print(f"Failed to verify concepts from '{note['title']}': {result}")
            continue

        for concept in result.concepts:
            concept_dict = concept.model_dump()
            concept_dict["source_note"] = note["title"]
            all_verified_concepts.append(concept_dict)
            
        if result.tags:
            tags_by_note[note["title"]] = result.tags
            
    if all_verified_concepts:
        directory_path = state.get("dir", "")
        faiss_path = Path(directory_path) / ".linker_faiss_index"
        
        docs = [
            Document(
                page_content=f"{c['concept_name']}: {c['explanation']}",
                metadata={"name": c['concept_name'], "note": c['source_note']}
            ) for c in all_verified_concepts
        ]
        
        print(f"Indexing {len(docs)} new concepts into Pinecone...")
        
        try:
            from pinecone import Pinecone as PineconeClient
            import os
            import uuid
            
            pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "obsidian-brain"))
            
            # Delete stale concepts for modified notes
            new_notes_titles = {note["title"] for note in new_notes}
            if new_notes_titles:
                print(f"Removing stale concepts for {len(new_notes_titles)} modified notes...")
                for title in new_notes_titles:
                    try:
                        index.delete(filter={"note": title}, namespace="obsidian")
                    except Exception as e:
                        pass # It's okay if there are no stale concepts
            
            # Generate embeddings and Upsert
            print("Generating embeddings for new concepts...")
            vectors = []
            for doc in docs:
                # We should use batch embeddings if possible, but for simplicity here:
                meta = doc.metadata.copy()
                meta["text"] = doc.page_content
                vectors.append((str(uuid.uuid4()), doc.page_content, meta))
            
            # Use embeddings.embed_documents for faster batching
            texts_to_embed = [v[1] for v in vectors]
            embeddings_list = embeddings.embed_documents(texts_to_embed)
            
            final_vectors = []
            for i, emb in enumerate(embeddings_list):
                final_vectors.append((vectors[i][0], emb, vectors[i][2]))
            
            for i in range(0, len(final_vectors), 50):
                batch = final_vectors[i:i+50]
                index.upsert(vectors=batch, namespace="obsidian")
                print(f"  Indexed {min(i + 50, len(final_vectors))}/{len(final_vectors)}...")
                
        except Exception as embed_e:
            print(f"Failed to index concepts into Pinecone: {embed_e}")
            
    print(f"Total verified concepts from new notes: {len(all_verified_concepts)}")
    return {
        "new_concepts": all_verified_concepts,
        "concepts": state.get("concepts", []) + all_verified_concepts,
        "tags_by_note": tags_by_note
    }
