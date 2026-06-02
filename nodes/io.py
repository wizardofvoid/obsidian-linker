from pathlib import Path
import json
import hashlib
import re
from core.state import AgentState

def strip_auto_sections(content: str) -> str:
    """Safely extracts the core content of a markdown note, removing only the ## Tags and 
    ## Related Links sections (and their contents up to the next root-level heading or EOF),
    preserving any manually added content below them."""
    lines = content.splitlines()
    in_code_block = False
    filtered_lines = []
    skip_mode = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            
        if not in_code_block:
            # Check if this line starts an auto-generated section
            if stripped == "## Tags" or stripped == "## Related Links":
                skip_mode = True
                continue
            # If in skip mode, check if we hit the next heading
            elif skip_mode and re.match(r"^#+\s", stripped):
                skip_mode = False
                
        if not skip_mode:
            filtered_lines.append(line)
            
    return "\n".join(filtered_lines).rstrip()

def get_core_hash(content: str) -> str:
    """Returns an MD5 hash of the note content, ignoring Related Links and Tags sections."""
    core_content = strip_auto_sections(content)
    return hashlib.md5(core_content.encode('utf-8')).hexdigest()

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
    cached_tags = cache.get("tags", {})
    
    notes = []
    new_notes = []
    current_hashes = {}
    current_titles = set()
    
    for file in directory.rglob("*.md"):
        title = file.name
        current_titles.add(title)
        
        content = file.read_text(encoding="utf-8")
        content_hash = get_core_hash(content)
        current_hashes[title] = content_hash
        
        note_obj = {
            "title": title, 
            "path": str(file.resolve()),
            "content": content
        }
        notes.append(note_obj)
        
        if title not in cached_files or cached_files[title].get("hash") != content_hash:
            new_notes.append(note_obj)
            
    valid_titles = current_titles - {n["title"] for n in new_notes}
    
    retained_concepts = [c for c in all_cached_concepts if c["source_note"] in valid_titles]
    retained_links = [l for l in all_cached_links if l.get("from_note") in valid_titles and l.get("to_note") in valid_titles]
    retained_tags = {k: v for k, v in cached_tags.items() if k in valid_titles}

    print(f"\n--- [Vault Reader] ---")
    print(f"Found {len(notes)} notes in vault.")
    print(f"Notes needing processing: {len(new_notes)}")
    if retained_concepts or retained_links or retained_tags:
        print(f"Loaded {len(retained_concepts)} cached concepts, {len(retained_links)} cached links, and {len(retained_tags)} cached tags.")
    
    return {
        "notes": notes, 
        "new_notes": new_notes,
        "dir": str(directory_path),
        "cache_path": str(cache_path),
        "file_hashes": current_hashes,
        "concepts": retained_concepts,
        "links": retained_links,
        "tags_by_note": retained_tags,
        "raw_tags_by_note": {}
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
            target_title = to_note[:-3] if to_note.lower().endswith(".md") else to_note
            links_by_source[from_note].add(target_title)
            
    tags_by_note = state.get("tags_by_note", {})

    new_notes_titles = {note["title"] for note in state.get("new_notes", [])}
    new_links = state.get("new_links", [])
    new_link_sources = {link["from_note"] for link in new_links if link.get("from_note")}
    allowed_updates = new_notes_titles.union(new_link_sources)

    # Gather all notes that need to be updated with links OR tags, filtered to only those modified/updated in this run
    all_notes_to_update = set(links_by_source.keys()).union(tags_by_note.keys())
    all_notes_to_update = all_notes_to_update.intersection(allowed_updates)

    for note_title in all_notes_to_update:
        file_path_str = note_to_path.get(note_title)
        original_content = note_to_content.get(note_title)
        
        if not file_path_str or not original_content:
            continue
            
        file_path = Path(file_path_str)
        
        base_content = strip_auto_sections(original_content)
        
        new_content = base_content
        
        # Add tags if they exist
        note_tags = tags_by_note.get(note_title)
        if note_tags:
            tags_block = " ".join(note_tags)
            new_content += f"\n\n## Tags\n{tags_block}"

        # Add links if they exist
        targets = links_by_source.get(note_title)
        if targets:
            sorted_targets = sorted(list(targets))
            links_block = "\n".join(f"- [[{target}]]" for target in sorted_targets)
            new_content += f"\n\n## Related Links\n{links_block}"
        
        new_content += "\n"
        
        if new_content == original_content:
            continue
            
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        try:
            temp_path.write_text(new_content, encoding="utf-8")
            temp_path.replace(file_path)
            
            updates = []
            if targets:
                updates.append(f"{len(targets)} links")
            if note_tags:
                updates.append(f"{len(note_tags)} tags")
            print(f"Successfully wrote {' and '.join(updates)} to {note_title}")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            print(f"Failed to write links to {note_title}: {e}")
            
    return {}

def summary_reporter(state: AgentState):
    print("\n--- [Summary Reporter] ---")
    print("Done! Links created:", len(state.get("links", [])))
    
    # Save back to cache
    cache_path = state.get("cache_path")
    if cache_path:
        cache_data = {
            "files": {title: {"hash": file_hash} for title, file_hash in state.get("file_hashes", {}).items()},
            "concepts": state.get("concepts", []),
            "links": state.get("links", []),
            "tags": state.get("tags_by_note", {})
        }
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            print(f"Saved cache to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            
    return {}
