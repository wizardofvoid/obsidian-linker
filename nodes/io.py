from pathlib import Path
import json
from core.state import AgentState

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

def link_writer(state: AgentState):
    print("\n--- [Link Writer] ---")
    links = state.get("links", [])
    note_to_path = {n["title"]: n["path"] for n in state.get("notes", [])}
    note_to_content = {n["title"]: n["content"] for n in state.get("notes", [])}
    
    links_by_source = {}
    for link in links:
        from_note = link.get("from_note")
        to_note = link.get("to_note")
        
        if from_note and to_note:
            if from_note not in links_by_source:
                links_by_source[from_note] = set()
            target_title = to_note.replace(".md", "")
            links_by_source[from_note].add(target_title)
            
    for note_title, targets in links_by_source.items():
        file_path_str = note_to_path.get(note_title)
        original_content = note_to_content.get(note_title)
        
        if not file_path_str or not original_content:
            continue
            
        file_path = Path(file_path_str)
        
        parts = original_content.split("\n## Related Links")
        base_content = parts[0].rstrip()
        
        sorted_targets = sorted(list(targets))
        links_block = "\n".join(f"- [[{target}]]" for target in sorted_targets)
        new_content = f"{base_content}\n\n## Related Links\n{links_block}\n"
        
        if new_content == original_content:
            continue
            
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
    print("Done! Links created:", len(state.get("links", [])))
    return {}
