import os
from dotenv import load_dotenv
import getpass
import time
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
# Load all API keys
GROQ_API_KEYS = [v.strip() for k, v in os.environ.items() if "GROQ_API_KEY" in k and v.strip()]
if not GROQ_API_KEYS:
    key = getpass.getpass("Enter your Groq API key: ")
    GROQ_API_KEYS.append(key)

# Load all Google API keys
GOOGLE_API_KEYS = [v.strip() for k, v in os.environ.items() if "GOOGLE_API_KEY" in k and v.strip()]
if not GOOGLE_API_KEYS:
    key = getpass.getpass("Enter your Google API key for embeddings: ")
    GOOGLE_API_KEYS.append(key)

current_google_key_index = 0

class RotatingGoogleEmbeddings(Embeddings):
    def __init__(self, model_name: str = "models/gemini-embedding-2"):
        self.model_name = model_name
        
    def _get_embedding_instance(self):
        global current_google_key_index
        key = GOOGLE_API_KEYS[current_google_key_index]
        return GoogleGenerativeAIEmbeddings(model=self.model_name, google_api_key=key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Call embed_query for each text to guarantee we get exactly one embedding per text,
        # avoiding the langchain-google-genai batch embedding bug in this environment.
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        global current_google_key_index
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                emb = self._get_embedding_instance()
                return emb.embed_query(text)
            except Exception as e:
                err_msg = str(e).lower()
                if "rate limit" in err_msg or "429" in err_msg or "resource_exhausted" in err_msg:
                    if len(GOOGLE_API_KEYS) > 1:
                        print(f"  Google API rate limit hit on Key {current_google_key_index + 1}. Switching to next key...")
                        current_google_key_index = (current_google_key_index + 1) % len(GOOGLE_API_KEYS)
                        if attempt < max_retries - 1:
                            continue
                    else:
                        print("  Google API rate limit hit. No alternative keys to switch to.")
                
                if attempt == max_retries - 1:
                    raise
                    
                delay = base_delay * (2 ** attempt)
                print(f"  Embedding query failed (Attempt {attempt+1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)

embeddings = RotatingGoogleEmbeddings(model_name="models/gemini-embedding-2")

# --- Per-stage LLM configuration ---
LLM_EXTRACTION = os.getenv("LLM_EXTRACTION", LLM_MODEL)
LLM_VERIFICATION = os.getenv("LLM_VERIFICATION", LLM_MODEL)
LLM_RELATIONSHIP = os.getenv("LLM_RELATIONSHIP", LLM_MODEL)

print(f"Models loaded:")
print(f"  Extraction:    {LLM_EXTRACTION}")
print(f"  Verification:  {LLM_VERIFICATION}")
print(f"  Relationship:  {LLM_RELATIONSHIP}")
print(f"Loaded {len(GROQ_API_KEYS)} Groq API keys and {len(GOOGLE_API_KEYS)} Google API keys for rotation.")
