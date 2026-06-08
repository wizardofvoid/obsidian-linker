from langgraph.graph import StateGraph, END
from core.state import AgentState

# We must import config first to ensure env vars/keys are loaded
import core.config 

from nodes.io import vault_reader, link_writer, summary_reporter
from nodes.concepts import concept_extractor, concept_verifier
from nodes.relations import relationship_extractor, relationship_verifier

# --- Build graph ---
graph = StateGraph(AgentState)

graph.add_node("vault_reader",           vault_reader)
graph.add_node("concept_extractor",      concept_extractor)
graph.add_node("concept_verifier",       concept_verifier)
graph.add_node("relationship_extractor", relationship_extractor)
graph.add_node("relationship_verifier",  relationship_verifier)
graph.add_node("link_writer",            link_writer)
graph.add_node("summary_reporter",       summary_reporter)

graph.set_entry_point("vault_reader")

graph.add_edge("vault_reader",           "concept_extractor")
graph.add_edge("concept_extractor",      "concept_verifier")
graph.add_edge("concept_verifier",       "relationship_extractor")
graph.add_edge("relationship_extractor", "relationship_verifier")
graph.add_edge("relationship_verifier",  "link_writer")
graph.add_edge("link_writer",            "summary_reporter")
graph.add_edge("summary_reporter",       END)

app = graph.compile()

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import asyncio
import os
import subprocess
from pathlib import Path

fastapi_app = FastAPI(title="Obsidian Linker API")

class SyncRequest(BaseModel):
    github_url: str = None
    github_token: str = None
    
def sync_vault(repo_url: str, token: str, vault_dir: Path) -> bool:
    if not repo_url:
        return False
    clean_repo_url = repo_url.replace("https://", "").replace("http://", "")
    auth_url = f"https://{token}@{clean_repo_url}" if token else f"https://{clean_repo_url}"
    git_dir = vault_dir / ".git"
    try:
        if vault_dir.exists() and git_dir.exists():
            subprocess.run(["git", "-C", str(vault_dir), "pull"], check=True)
        else:
            vault_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", auth_url, str(vault_dir)], check=True)
        return True
    except Exception as e:
        print(f"Git sync failed: {e}")
        return False

async def run_pipeline(vault_dir: str):
    await app.ainvoke({
        "notes": [], 
        "new_notes": [],
        "raw_concepts": [],
        "concepts": [], 
        "new_concepts": [],
        "raw_links": [],
        "links": [],
        "dir": vault_dir,
        "cache_path": "",
        "file_hashes": {},
        "raw_tags_by_note": {},
        "tags_by_note": {}
    })

@fastapi_app.post("/sync")
async def sync_endpoint(req: SyncRequest, background_tasks: BackgroundTasks):
    vault_dir = Path("/tmp/ObsidianVault") if os.name != 'nt' else Path(os.environ.get("TEMP", "C:/temp")) / "ObsidianVault"
    
    # Use request body or fallback to environment variables
    repo_url = req.github_url or os.getenv("OBSIDIAN_REPO_URL")
    token = req.github_token or os.getenv("GITHUB_TOKEN")
    
    if repo_url:
        success = sync_vault(repo_url, token, vault_dir)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to sync GitHub repository.")
    else:
        # Fallback for local testing if no repo is provided
        vault_dir = Path(os.getenv("OBSIDIAN_VAULT_DIR", str(vault_dir)))
        if not vault_dir.exists():
            raise HTTPException(status_code=400, detail="No GitHub URL provided and local vault dir does not exist.")

    # Run the heavy LangGraph pipeline in the background so the HTTP request doesn't timeout
    background_tasks.add_task(run_pipeline, str(vault_dir))
    return {"status": "success", "message": "Obsidian sync and LangGraph extraction started in the background."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)