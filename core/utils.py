import time
from typing import List
from langchain_groq import ChatGroq
from core.config import GROQ_API_KEYS

current_key_index = 0

def minify_concepts(concepts: List[dict]) -> List[dict]:
    """Strips bulky fields (keywords, related_concepts) to drastically reduce token usage."""
    return [
        {
            "name": c.get("concept_name"),
            "note": c.get("source_note"),
            "desc": c.get("explanation")
        }
        for c in concepts
    ]

def invoke_with_retry(prompt, model_name, parser, inputs, max_retries=5, base_delay=2.0):
    """Invoke a chain with exponential backoff and API key rotation on failure."""
    global current_key_index
    for attempt in range(max_retries):
        try:
            api_key = GROQ_API_KEYS[current_key_index]
            llm = ChatGroq(groq_api_key=api_key, model=model_name, max_tokens=4096)
            chain = prompt | llm | parser
            return chain.invoke(inputs)
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "429" in err_msg:
                if len(GROQ_API_KEYS) > 1:
                    print(f"  Rate limit hit on Key {current_key_index + 1}. Switching to next key...")
                    current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                    if attempt < max_retries - 1:
                        continue # Retry immediately with new key
                else:
                    print("  Rate limit hit. No alternative keys to switch to.")
                    
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
                
            delay = base_delay * (2 ** attempt)
            print(f"  Attempt {attempt + 1} failed: {e}")
            print(f"  Retrying in {delay:.0f}s...")
            time.sleep(delay)
