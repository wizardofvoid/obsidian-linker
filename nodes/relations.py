import json
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
import os
from core.state import AgentState, RelationshipExtractionOutput
from core.utils import invoke_with_retry, minify_concepts
from core.config import LLM_RELATIONSHIP, LLM_VERIFICATION, embeddings
from core.prompts import prompt_relation, prompt_relation_verify

def relationship_extractor(state: AgentState):
    print("\n--- [Relationship Extractor] ---")
    new_concepts = state.get("new_concepts", [])
    all_concepts = state.get("concepts", [])
    
    # Smart Resumption: If no new concepts were extracted in this run (e.g. because we are resuming after an error),
    # but we have cached concepts and no links created yet, we can resume directly from here by treating cached concepts as new.
    if not new_concepts:
        cached_links = state.get("links", [])
        if all_concepts and not cached_links:
            print("No new concepts extracted, but cached concepts exist with 0 links. Resuming relationship extraction for all concepts...")
            new_concepts = all_concepts
        else:
            print("No new concepts extracted. Skipping relationship extraction.")
            return {"raw_links": []}
    
    try:
        # Load Pinecone index
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "obsidian-brain")
        from pinecone import Pinecone as PineconeClient
        pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(pinecone_index_name)
        
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace="obsidian",
            text_key="text"
        )
        
        # Group new concepts by their source note to process them note-by-note
        concepts_by_note = {}
        for c in new_concepts:
            note = c.get("source_note")
            if note:
                if note not in concepts_by_note:
                    concepts_by_note[note] = []
                concepts_by_note[note].append(c)
                
        all_relations = []
        print(f"Extracting relationships for {len(concepts_by_note)} notes...")
        
        for note_title, note_concepts in concepts_by_note.items():
            retrieved_concepts = []
            retrieved_names = set()
            
            for c in note_concepts:
                query = f"{c['concept_name']}: {c['explanation']}"
                results = vectorstore.similarity_search(query, k=10)
                
                for doc in results:
                    name = doc.metadata.get("name")
                    note = doc.metadata.get("note")
                    
                    # Exclude concepts from the same note
                    if note == note_title:
                        continue
                        
                    key = (name, note)
                    if key not in retrieved_names:
                        retrieved_names.add(key)
                        match = next((x for x in all_concepts if x["concept_name"] == name and x["source_note"] == note), None)
                        if match:
                            retrieved_concepts.append(match)
                            
            # Focus on top 15 most semantically similar concepts to keep prompt compact and precise
            retrieved_concepts = retrieved_concepts[:15]
            
            if not retrieved_concepts:
                continue
                
            print(f"  Note '{note_title}': evaluating {len(note_concepts)} concepts against {len(retrieved_concepts)} relevant concepts...")
            
            minified_new = minify_concepts(note_concepts)
            minified_existing = minify_concepts(retrieved_concepts)
            
            extracted_output = invoke_with_retry(prompt_relation, LLM_RELATIONSHIP, RelationshipExtractionOutput, {
                "new_concepts": json.dumps(minified_new),
                "existing_concepts": json.dumps(minified_existing)
            })
            
            if extracted_output and extracted_output.links:
                all_relations.extend([link.model_dump() for link in extracted_output.links])
                
        print(f"Total raw cross-note relationships extracted: {len(all_relations)}")
        return {"raw_links": all_relations}
    except Exception as e:
        print(f"Failed to extract cross-note relationships: {e}")
        return {"raw_links": []}

def relationship_verifier(state: AgentState):
    print("\n--- [Relationship Verifier] ---")
    raw_links = state.get("raw_links", [])
    concepts = state.get("concepts", [])
    
    if not raw_links or not concepts:
        print("No raw links or concepts to verify.")
        return {"links": state.get("links", [])}
    
    try:
        # We will verify raw_links in batches of 20 to avoid exceeding LLM context / token limits (Groq 413)
        batch_size = 20
        all_verified_links = []
        
        print(f"Verifying {len(raw_links)} cross-note relationships in batches of {batch_size}...")
        
        # Helper to map concept name to its dict for fast lookup
        concept_lookup = {c["concept_name"]: c for c in concepts}
        
        for i in range(0, len(raw_links), batch_size):
            batch_links = raw_links[i : i + batch_size]
            
            # Find unique concepts involved in this batch
            concepts_in_batch = set()
            for link in batch_links:
                if link.get("source"):
                    concepts_in_batch.add(link["source"])
                if link.get("target"):
                    concepts_in_batch.add(link["target"])
                    
            # Filter concepts to only those involved in this batch
            relevant_concepts = [concept_lookup[name] for name in concepts_in_batch if name in concept_lookup]
            
            minified_concepts = minify_concepts(relevant_concepts)
            links_json = json.dumps(batch_links)
            
            print(f"  Verifying batch {i // batch_size + 1}/{(len(raw_links) - 1) // batch_size + 1} ({len(batch_links)} links, {len(minified_concepts)} relevant concepts)...")
            
            verified_output = invoke_with_retry(prompt_relation_verify, LLM_VERIFICATION, RelationshipExtractionOutput, {
                "concepts": json.dumps(minified_concepts),
                "relationships": links_json
            })
            
            if verified_output and verified_output.links:
                all_verified_links.extend([link.model_dump() for link in verified_output.links])
        
        # Structural filtering
        note_titles = {note["title"] for note in state.get("notes", [])}
        title_map = {t.lower(): t for t in note_titles}
        concept_to_note = {c["concept_name"]: c["source_note"] for c in concepts}
        valid_links = []
        prevented = 0
        
        for link in all_verified_links:
            source_concept = link.get("source")
            target_concept = link.get("target")
            
            from_note = link.get("from_note") or concept_to_note.get(source_concept)
            to_note = link.get("to_note") or concept_to_note.get(target_concept)
            
            # Resolve suffix and case mismatches dynamically
            if from_note:
                if not from_note.lower().endswith(".md"):
                    from_note_with_ext = f"{from_note}.md"
                else:
                    from_note_with_ext = from_note
                resolved = title_map.get(from_note_with_ext.lower())
                if resolved:
                    from_note = resolved
                    
            if to_note:
                if not to_note.lower().endswith(".md"):
                    to_note_with_ext = f"{to_note}.md"
                else:
                    to_note_with_ext = to_note
                resolved = title_map.get(to_note_with_ext.lower())
                if resolved:
                    to_note = resolved
            
            if not from_note or not to_note:
                prevented += 1
                continue
            if from_note not in note_titles or to_note not in note_titles:
                prevented += 1
                continue
            if from_note == to_note:
                prevented += 1
                continue
                
            link["from_note"] = from_note
            link["to_note"] = to_note
            valid_links.append(link)
            
        print(f"Total verified cross-note relationships: {len(valid_links)} (prevented {prevented} invalid links)")
        return {
            "links": state.get("links", []) + valid_links,
            "new_links": valid_links
        }
    except Exception as e:
        print(f"Failed to verify relationships: {e}")
        return {"links": state.get("links", []), "new_links": []}
