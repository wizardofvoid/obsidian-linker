import json
from core.state import AgentState

def quality_checker(state: AgentState):
    print("\n--- [Quality Checker] ---")
    links = state.get("links", [])
    note_titles = {note["title"] for note in state.get("notes", [])}
    
    concept_to_note = {c["concept_name"]: c["source_note"] for c in state.get("concepts", [])}
    
    issues = []
    valid_links = []
    
    for link in links:
        source_concept = link.get("source")
        target_concept = link.get("target")
        
        from_note = link.get("from_note") or concept_to_note.get(source_concept)
        to_note = link.get("to_note") or concept_to_note.get(target_concept)
        
        if not from_note:
            issues.append(f"Source concept '{source_concept}' not found in extracted concepts.")
            continue
        if not to_note:
            issues.append(f"Target concept '{target_concept}' not found in extracted concepts.")
            continue
            
        if from_note not in note_titles:
            issues.append(f"Source note '{from_note}' (for concept '{source_concept}') does not exist in vault.")
            continue
        if to_note not in note_titles:
            issues.append(f"Target note '{to_note}' (for concept '{target_concept}') does not exist in vault.")
            continue
            
        if from_note == to_note:
            issues.append(f"Self-link prevented: '{source_concept}' and '{target_concept}' are both in '{from_note}'.")
            continue
        
        link["from_note"] = from_note
        link["to_note"] = to_note
        valid_links.append(link)
    
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
            "files": {title: {"mtime": mtime} for title, mtime in state.get("file_mtimes", {}).items()},
            "concepts": state.get("concepts", []),
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

def should_retry(state: AgentState):
    retry_count = state.get("retry_count", 0)
    quality_score = state.get("quality_score", 0.0)
    
    if retry_count >= 3:
        print(f"Max retries reached ({retry_count}). Proceeding to link writer.")
        return "link_writer"
    
    if not state.get("concepts"):
        print(f"No concepts found. Retrying from concept extraction (retry {retry_count}/3)...")
        return "concept_extractor"
    
    if quality_score < 0.7:
        print(f"Quality score low ({quality_score:.2f}), retrying relationship extraction (retry {retry_count}/3)...")
        return "relationship_extractor"
    
    return "link_writer"
