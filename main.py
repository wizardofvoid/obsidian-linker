from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from pathlib import Path
from dotenv import load_dotenv
import os
import getpass
import json
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

groq_api_key = os.getenv("GROQ_API_KEY")

# --- Per-stage LLM configuration ---
# You can set different models for each stage via environment variables.
# Falls back to LLM_MODEL if not set.
LLM_EXTRACTION = os.getenv("LLM_EXTRACTION", LLM_MODEL)      # concept extraction
LLM_VERIFICATION = os.getenv("LLM_VERIFICATION", LLM_MODEL)   # concept & relationship verification
LLM_RELATIONSHIP = os.getenv("LLM_RELATIONSHIP", LLM_MODEL)   # relationship extraction

llm_extraction = ChatGroq(groq_api_key=groq_api_key, model=LLM_EXTRACTION, max_tokens=4096)
llm_verification = ChatGroq(groq_api_key=groq_api_key, model=LLM_VERIFICATION, max_tokens=4096)
llm_relationship = ChatGroq(groq_api_key=groq_api_key, model=LLM_RELATIONSHIP, max_tokens=4096)

parser = JsonOutputParser()

print(f"Models loaded:")
print(f"  Extraction:    {LLM_EXTRACTION}")
print(f"  Verification:  {LLM_VERIFICATION}")
print(f"  Relationship:  {LLM_RELATIONSHIP}")

def invoke_with_retry(chain, inputs, max_retries=3, base_delay=2.0):
    """Invoke a chain with exponential backoff retry on failure."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
            delay = base_delay * (2 ** attempt)
            print(f"  Attempt {attempt + 1} failed: {e}")
            print(f"  Retrying in {delay:.0f}s...")
            time.sleep(delay)

class Concept(TypedDict):
    concept_name: str
    category: str
    explanation: str
    important_keywords: List[str]
    related_concepts: List[str]
    importance_score: int
    source_note: str

# --- State ---
class AgentState(TypedDict):
    notes: List[dict]
    new_notes: List[dict]
    raw_concepts: List[dict]
    concepts: List[Concept]
    new_concepts: List[Concept]
    raw_links: List[dict]
    links: List[dict]
    quality_score: float
    retry_count: int
    dir: str
    retry_reason: Optional[List[str]]
    cache_path: str
    file_mtimes: dict

def vault_reader(state: AgentState):
    directory_path = state.get("dir", "")
    if not directory_path:
        directory_path = input("Please enter your Obsidian vault directory path: ").strip()
    
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"The path '{directory_path}' is not a valid directory.")
    
    cache_path = directory / ".linker_cache.json"
    cache = {"files": {}, "concepts": [], "links": []}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read cache: {e}")

    cached_files = cache.get("files", {})
    all_cached_concepts = cache.get("concepts", [])
    all_cached_links = cache.get("links", [])
    
    notes = []
    new_notes = []
    current_mtimes = {}
    current_titles = set()
    
    for file in directory.rglob("*.md"):
        title = file.name
        current_titles.add(title)
        mtime = file.stat().st_mtime
        current_mtimes[title] = mtime
        
        note_obj = {
            "title": title, 
            "path": str(file.resolve()),
            "content": file.read_text(encoding="utf-8")
        }
        notes.append(note_obj)
        
        if title not in cached_files or cached_files[title].get("mtime") != mtime:
            new_notes.append(note_obj)
            
    # Cleanup cache: remove deleted/modified notes from cached concepts and links
    valid_titles = current_titles - {n["title"] for n in new_notes}
    
    retained_concepts = [c for c in all_cached_concepts if c["source_note"] in valid_titles]
    retained_links = [l for l in all_cached_links if l.get("from_note") in valid_titles and l.get("to_note") in valid_titles]

    print(f"\n--- [Vault Reader] ---")
    print(f"Found {len(notes)} notes in vault.")
    print(f"Notes needing processing: {len(new_notes)}")
    if retained_concepts or retained_links:
        print(f"Loaded {len(retained_concepts)} cached concepts and {len(retained_links)} cached links.")
    
    return {
        "notes": notes, 
        "new_notes": new_notes,
        "dir": str(directory_path),
        "cache_path": str(cache_path),
        "file_mtimes": current_mtimes,
        "concepts": retained_concepts,
        "links": retained_links
    }

def concept_extractor(state: AgentState):
    print("\n--- [Concept Extractor] ---")
    prompt_concept_extraction = ChatPromptTemplate.from_messages([
        ("system", "You are an expert knowledge extraction system. Analyze the provided text and extract concepts as requested. Return ONLY the raw JSON array of concept objects. Do NOT write code or explanations outside of JSON."),
        ("user", """Analyze the following text and extract all important concepts, topics, technologies, methods, and entities.
                    For each extracted concept provide:
                    - concept_name
                    - category
                    - explanation
                    - important_keywords
                    - related_concepts
                    - importance_score (1-10)

                    Rules:
                    - Extract only the top 5-8 most significant concepts.
                    - Only extract concepts explicitly mentioned or strongly implied.
                    - Avoid duplicates.
                    - Keep explanations concise.

                    {format_instructions}

                    TEXT: {text}""")
    ])
    
    extraction_chain = prompt_concept_extraction | llm_extraction | parser
    all_extracted_concepts = []

    for note in state.get("new_notes", []):
        try: 
            print(f"Extracting raw concepts from: {note['title']}...")
            extracted_raw = invoke_with_retry(extraction_chain, {
                "text": note["content"],
                "format_instructions": parser.get_format_instructions()
            })
            verified_list = extracted_raw if isinstance(extracted_raw, list) else extracted_raw.get("concepts", [])

            for concept in verified_list:
                concept["source_note"] = note["title"]
                all_extracted_concepts.append(concept)
                
        except Exception as e:
            print(f"Failed to extract concepts from '{note['title']}': {e}")
            continue
            
    print(f"Total raw concepts extracted: {len(all_extracted_concepts)}")
    return {"raw_concepts": all_extracted_concepts}

def concept_verifier(state: AgentState):
    print("\n--- [Concept Verifier] ---")
    prompt_concept_verification = ChatPromptTemplate.from_messages([
        ("system", "You are a verification system. Clean and verify the extracted concepts. Do NOT write Python code or programs. Output ONLY the raw JSON array of concepts directly."),
        ("user", """Analyze the provided concepts extracted from the text below. Clean and verify them by:
            1. Removing any incorrect concepts that are not supported by the text.
            2. Removing duplicate or highly redundant concepts.
            3. Merging concepts that refer to the exact same entity/topic.
            4. Refining descriptions to be highly accurate and clear based on the text.

            {format_instructions}
            
            TEXT:
            {text}
            CONCEPTS:
            {concepts}""")
    ])
    
    verification_chain = prompt_concept_verification | llm_verification | parser
    all_verified_concepts = []

    for note in state.get("new_notes", []):
        try:
            # Filter raw concepts for this note
            note_raw_concepts = [c for c in state.get("raw_concepts", []) if c.get("source_note") == note["title"]]
            if not note_raw_concepts:
                continue
                
            print(f"Verifying concepts for: {note['title']}...")
            concepts_json = json.dumps(note_raw_concepts)
            verified_raw = invoke_with_retry(verification_chain, {
                "text": note["content"], 
                "concepts": concepts_json,
                "format_instructions": parser.get_format_instructions()
            })
            verified_list = verified_raw if isinstance(verified_raw, list) else verified_raw.get("concepts", [])

            for concept in verified_list:
                concept["source_note"] = note["title"]
                all_verified_concepts.append(concept)
                
        except Exception as e:
            print(f"Failed to verify concepts from '{note['title']}': {e}")
            continue
            
    print(f"Total verified concepts from new notes: {len(all_verified_concepts)}")
    return {
        "new_concepts": all_verified_concepts,
        "concepts": state.get("concepts", []) + all_verified_concepts
    }

def relationship_extractor(state: AgentState):
    print("\n--- [Relationship Extractor] ---")
    new_concepts = state.get("new_concepts", [])
    all_concepts = state.get("concepts", [])
    
    if not new_concepts:
        print("No new concepts extracted. Skipping relationship extraction.")
        return {"raw_links": [], "retry_count": state.get("retry_count", 0) + 1}
    
    prompt_relation = ChatPromptTemplate.from_messages([
        ("system", "You are a relationship extraction engine that finds connections BETWEEN different notes in a knowledge base. Return ONLY the raw JSON array of relationship objects. Do NOT write code."),
        ("user", """Below are two lists of concepts extracted from notes in a knowledge base.
Each concept has a 'source_note' field indicating which note it came from.

YOUR TASK: Find semantic relationships that involve AT LEAST ONE concept from the NEW CONCEPTS list.
You can link New ↔ New, or New ↔ Existing, but DO NOT link Existing ↔ Existing (we already know those).

Allowed relationship types:
- uses, depends_on, extends, similar_to, part_of
- implemented_with, alternative_to, improves, causes

For each relationship, include:
- "source": the concept name (from one note)
- "target": the concept name (from a DIFFERENT note)
- "relationship": one of the allowed types
- "evidence": brief explanation of why these concepts are related
- "from_note": the source_note of the source concept
- "to_note": the source_note of the target concept

Rules:
- ONLY extract relationships between concepts from DIFFERENT notes.
- AT LEAST ONE concept in the relationship MUST be from the NEW CONCEPTS list.
- Do NOT create relationships between concepts from the same note.
- Focus on meaningful semantic connections.

{format_instructions}

NEW CONCEPTS:
{new_concepts}

EXISTING CONCEPTS:
{existing_concepts}""")
    ])
    
    relation_chain = prompt_relation | llm_relationship | parser
    
    try:
        existing_concepts = [c for c in all_concepts if c not in new_concepts]
        print(f"Extracting relationships involving {len(new_concepts)} new concepts vs {len(existing_concepts)} existing concepts...")
        extracted_relations = invoke_with_retry(relation_chain, {
            "new_concepts": json.dumps(new_concepts),
            "existing_concepts": json.dumps(existing_concepts),
            "format_instructions": parser.get_format_instructions()
        })
        relations_list = extracted_relations if isinstance(extracted_relations, list) else extracted_relations.get("links", [])
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
        return {"links": []}
    
    prompt_relation_verify = ChatPromptTemplate.from_messages([
        ("system", "You are a link verification engine. Verify the proposed cross-note relationships. Do NOT write Python code. Output ONLY the raw JSON array of verified relationships directly."),
        ("user", """Below are proposed relationships between concepts from different notes.
Verify each relationship:
1. Remove any invalid or unsupported relationships.
2. Remove duplicate relationships.
3. Ensure the source and target concepts actually exist in the concepts list.
4. Ensure the relationship type is semantically correct.

{format_instructions}

CONCEPTS:
{concepts}

RELATIONSHIPS:
{relationships}""")
    ])
    
    verify_chain = prompt_relation_verify | llm_verification | parser
    
    try:
        concepts_json = json.dumps(concepts)
        links_json = json.dumps(raw_links)
        print(f"Verifying {len(raw_links)} cross-note relationships...")
        
        verified_relations = invoke_with_retry(verify_chain, {
            "concepts": concepts_json,
            "relationships": links_json,
            "format_instructions": parser.get_format_instructions()
        })
        relations_list = verified_relations if isinstance(verified_relations, list) else verified_relations.get("links", [])
        print(f"Total verified cross-note relationships: {len(relations_list)}")
        return {"links": state.get("links", []) + relations_list}
    except Exception as e:
        print(f"Failed to verify relationships: {e}")
        return {"links": state.get("links", [])}

def quality_checker(state: AgentState):
    print("\n--- [Quality Checker] ---")
    links = state["links"]
    note_titles = {note["title"] for note in state["notes"]}
    
    # Map each concept name to its source note filename for fast lookup
    concept_to_note = {c["concept_name"]: c["source_note"] for c in state["concepts"]}
    
    issues = []
    valid_links = []
    
    for link in links:
        source_concept = link.get("source")
        target_concept = link.get("target")
        
        # Use from_note/to_note if provided by the LLM, otherwise look up from concepts
        from_note = link.get("from_note") or concept_to_note.get(source_concept)
        to_note = link.get("to_note") or concept_to_note.get(target_concept)
        
        # 1. Verify we can resolve both concepts to notes
        if not from_note:
            issues.append(f"Source concept '{source_concept}' not found in extracted concepts.")
            continue
        if not to_note:
            issues.append(f"Target concept '{target_concept}' not found in extracted concepts.")
            continue
            
        # 2. Check both notes actually exist in the vault
        if from_note not in note_titles:
            issues.append(f"Source note '{from_note}' (for concept '{source_concept}') does not exist in vault.")
            continue
        if to_note not in note_titles:
            issues.append(f"Target note '{to_note}' (for concept '{target_concept}') does not exist in vault.")
            continue
            
        # 3. Check a note isn't linking to itself
        if from_note == to_note:
            issues.append(f"Self-link prevented: '{source_concept}' and '{target_concept}' are both in '{from_note}'.")
            continue
        
        # Attach resolved note info to the link for downstream use
        link["from_note"] = from_note
        link["to_note"] = to_note
        valid_links.append(link)
    
    # Score = ratio of valid links to total links
    total = len(links)
    score = len(valid_links) / total if total > 0 else 0.0
    
    print(f"Assessed quality: Score = {score:.2f} ({len(valid_links)}/{total} links valid).")
    if issues:
        print(f"Detected {len(issues)} issues:")
        for issue in issues[:5]:
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more.")
            
    # Save back to cache
    cache_path = state.get("cache_path")
    if cache_path:
        cache_data = {
            "files": {title: {"mtime": mtime} for title, mtime in state["file_mtimes"].items()},
            "concepts": state["concepts"],
            "links": valid_links
        }
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            print(f"Saved cache to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
        
    return {
        "quality_score": score,
        "links": valid_links,
        "retry_reason": issues if issues else None
    }

def link_writer(state: AgentState):
    print("\n--- [Link Writer] ---")
    links = state["links"]
    concepts = state["concepts"]
    # 1. Map note title to its absolute path and original content
    note_to_path = {n["title"]: n["path"] for n in state["notes"]}
    note_to_content = {n["title"]: n["content"] for n in state["notes"]}
    
    # 2. Group links by their source note (from_note → set of target note titles)
    links_by_source = {}
    for link in links:
        from_note = link.get("from_note")
        to_note = link.get("to_note")
        
        if from_note and to_note:
            if from_note not in links_by_source:
                links_by_source[from_note] = set()
            # Strip '.md' for the wikilink format
            target_title = to_note.replace(".md", "")
            links_by_source[from_note].add(target_title)
            
    # 4. Write links to files atomically
    for note_title, targets in links_by_source.items():
        file_path_str = note_to_path.get(note_title)
        original_content = note_to_content.get(note_title)
        
        if not file_path_str or not original_content:
            continue
            
        file_path = Path(file_path_str)
        
        # Strip previous related links section if it exists
        parts = original_content.split("\n## Related Links")
        base_content = parts[0].rstrip()
        
        # Build new section
        sorted_targets = sorted(list(targets))
        links_block = "\n".join(f"- [[{target}]]" for target in sorted_targets)
        new_content = f"{base_content}\n\n## Related Links\n{links_block}\n"
        
        # Only write if content actually changed
        if new_content == original_content:
            continue
            
        # Safe Atomic Write
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        try:
            temp_path.write_text(new_content, encoding="utf-8")
            temp_path.replace(file_path)
            print(f"Successfully wrote {len(targets)} links to {note_title}")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            print(f"Failed to write links to {note_title}: {e}")
            
    return {}

def summary_reporter(state: AgentState):
    print("\n--- [Summary Reporter] ---")
    print("Done! Links created:", state["links"])
    return {}

# --- Conditional edge ---
def should_retry(state: AgentState):
    if state["retry_count"] >= 3:
        print(f"Max retries reached ({state['retry_count']}). Proceeding to link writer.")
        return "link_writer"
    
    # If concepts are empty, retrying relationships is pointless — restart from extraction
    if not state.get("concepts"):
        print(f"No concepts found. Retrying from concept extraction (retry {state['retry_count']}/3)...")
        return "concept_extractor"
    
    if state["quality_score"] < 0.7:
        print(f"Quality score low ({state['quality_score']:.2f}), retrying relationship extraction (retry {state['retry_count']}/3)...")
        return "relationship_extractor"
    
    return "link_writer"

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
        "raw_concepts": [],
        "concepts": [], 
        "raw_links": [],
        "links": [],
        "quality_score": 0.0, 
        "retry_count": 0,
        "dir": "",
        "retry_reason": None
    })