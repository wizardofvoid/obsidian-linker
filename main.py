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
    
def sync_vault(repo_url: str, token: str, vault_dir: Path) -> tuple[bool, str]:
    if not repo_url:
        return False, "repo_url is empty"
    repo_url = repo_url.strip('"').strip("'")
    clean_repo_url = repo_url.replace("https://", "").replace("http://", "")
    auth_url = f"https://{token}@{clean_repo_url}" if token else f"https://{clean_repo_url}"
    git_dir = vault_dir / ".git"
    try:
        if vault_dir.exists() and git_dir.exists():
            subprocess.run(["git", "-C", str(vault_dir), "reset", "--hard", "HEAD"], check=True)
            subprocess.run(["git", "-C", str(vault_dir), "clean", "-fd"], check=True)
            subprocess.run(["git", "-C", str(vault_dir), "pull"], check=True)
        else:
            import shutil
            if vault_dir.exists():
                shutil.rmtree(vault_dir)
            vault_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", auth_url, str(vault_dir)], check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        error_msg = f"Git command failed: {e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)}"
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Git sync failed: {e}"
        print(error_msg)
        return False, error_msg

def push_vault(vault_dir: Path) -> bool:
    git_dir = vault_dir / ".git"
    if not git_dir.exists():
        print("Not a git repository, skipping push.")
        return False
    try:
        # Check if there are changes
        status = subprocess.run(
            ["git", "-C", str(vault_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        if not status.stdout.strip():
            print("No changes to push.")
            return True
            
        print("Changes detected. Committing and pushing to GitHub...")
        # Configure temporary user
        subprocess.run(["git", "-C", str(vault_dir), "config", "user.name", "Obsidian Linker Agent"], check=True)
        subprocess.run(["git", "-C", str(vault_dir), "config", "user.email", "agent@obsidianlinker.com"], check=True)
        
        # Add, commit and push
        subprocess.run(["git", "-C", str(vault_dir), "add", "."], check=True)
        subprocess.run(["git", "-C", str(vault_dir), "commit", "-m", "Auto-update links and concept cache [Render]"], check=True)
        subprocess.run(["git", "-C", str(vault_dir), "push"], check=True)
        print("Successfully pushed changes to GitHub.")
        return True
    except Exception as e:
        print(f"Git push failed: {e}")
        return False


# Global state for task status tracking
task_status = {"status": "idle", "error": None}


async def run_pipeline(vault_dir: str):
    global task_status
    task_status["status"] = "running"
    task_status["error"] = None
    try:
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
        push_vault(Path(vault_dir))
        task_status["status"] = "completed"
    except Exception as e:
        task_status["status"] = "failed"
        task_status["error"] = str(e)
        print(f"Pipeline failed: {e}")


@fastapi_app.post("/sync")
async def sync_endpoint(req: SyncRequest, background_tasks: BackgroundTasks):
    global task_status
    task_status = {"status": "running", "error": None}
    
    vault_dir = Path("/tmp/ObsidianVault") if os.name != 'nt' else Path(os.environ.get("TEMP", "C:/temp")) / "ObsidianVault"
    
    # Use request body or fallback to environment variables
    repo_url = req.github_url or os.getenv("OBSIDIAN_REPO_URL")
    token = req.github_token or os.getenv("GITHUB_TOKEN")
    
    if repo_url:
        success, error_msg = sync_vault(repo_url, token, vault_dir)
        if not success:
            task_status = {"status": "failed", "error": f"Failed to sync GitHub repository: {error_msg}"}
            raise HTTPException(status_code=500, detail=f"Failed to sync GitHub repository: {error_msg}")
    else:
        # Fallback for local testing if no repo is provided
        vault_dir = Path(os.getenv("OBSIDIAN_VAULT_DIR", str(vault_dir)))
        if not vault_dir.exists():
            task_status = {"status": "failed", "error": "No GitHub URL provided and local vault dir does not exist."}
            raise HTTPException(status_code=400, detail="No GitHub URL provided and local vault dir does not exist.")

    # Run the heavy LangGraph pipeline in the background so the HTTP request doesn't timeout
    background_tasks.add_task(run_pipeline, str(vault_dir))
    return {"status": "success", "message": "Obsidian sync and LangGraph extraction started in the background."}


@fastapi_app.get("/status")
async def status_endpoint():
    return task_status

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)