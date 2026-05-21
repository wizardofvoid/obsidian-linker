from dotenv import load_dotenv
import os
import getpass

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
# Load all API keys
GROQ_API_KEYS = [v.strip() for k, v in os.environ.items() if "GROQ_API_KEY" in k and v.strip()]
if not GROQ_API_KEYS:
    key = getpass.getpass("Enter your Groq API key: ")
    GROQ_API_KEYS.append(key)

# --- Per-stage LLM configuration ---
LLM_EXTRACTION = os.getenv("LLM_EXTRACTION", LLM_MODEL)
LLM_VERIFICATION = os.getenv("LLM_VERIFICATION", LLM_MODEL)
LLM_RELATIONSHIP = os.getenv("LLM_RELATIONSHIP", LLM_MODEL)

print(f"Models loaded:")
print(f"  Extraction:    {LLM_EXTRACTION}")
print(f"  Verification:  {LLM_VERIFICATION}")
print(f"  Relationship:  {LLM_RELATIONSHIP}")
print(f"Loaded {len(GROQ_API_KEYS)} API keys for rotation.")
