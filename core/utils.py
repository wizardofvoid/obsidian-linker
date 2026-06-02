import time
import asyncio
from typing import List
from langchain_groq import ChatGroq
from core.config import GROQ_API_KEYS

current_key_index = 0
key_rotation_lock = None

def get_key_rotation_lock():
    global key_rotation_lock
    if key_rotation_lock is None:
        key_rotation_lock = asyncio.Lock()
    return key_rotation_lock

def minify_concepts(concepts: List[dict]) -> List[dict]:
    """Strips bulky fields (keywords, related_concepts) to drastically reduce token usage."""
    return [
        {
            "concept_name": c.get("concept_name"),
            "source_note": c.get("source_note"),
            "explanation": c.get("explanation")
        }
        for c in concepts
    ]

def invoke_with_retry(prompt, model_name, pydantic_schema, inputs, max_retries=None, base_delay=2.0):
    """Invoke a chain with exponential backoff, structured outputs, and API key rotation."""
    global current_key_index
    if max_retries is None:
        max_retries = max(len(GROQ_API_KEYS), 5)
    for attempt in range(max_retries):
        try:
            api_key = GROQ_API_KEYS[current_key_index]
            llm = ChatGroq(groq_api_key=api_key, model=model_name, max_tokens=1024)
            structured_llm = llm.with_structured_output(pydantic_schema)
            chain = prompt | structured_llm
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
            # Truncate error message to avoid terminal spam
            err_str = str(e)
            if len(err_str) > 200:
                err_str = err_str[:200] + "... [truncated]"
            print(f"  Attempt {attempt + 1} failed: {err_str}")
            print(f"  Retrying in {delay:.0f}s...")
            time.sleep(delay)

async def async_invoke_with_retry(prompt, model_name, pydantic_schema, inputs, max_retries=None, base_delay=2.0):
    """Async wrapper for invoking a chain with exponential backoff and API key rotation."""
    global current_key_index
    if max_retries is None:
        max_retries = max(len(GROQ_API_KEYS), 5)
    for attempt in range(max_retries):
        key_used = current_key_index
        try:
            api_key = GROQ_API_KEYS[key_used]
            llm = ChatGroq(groq_api_key=api_key, model=model_name, max_tokens=1024)
            structured_llm = llm.with_structured_output(pydantic_schema)
            chain = prompt | structured_llm
            return await chain.ainvoke(inputs)
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "429" in err_msg:
                if len(GROQ_API_KEYS) > 1:
                    lock = get_key_rotation_lock()
                    async with lock:
                        if current_key_index == key_used:
                            next_key = (current_key_index + 1) % len(GROQ_API_KEYS)
                            print(f"  Rate limit hit on Key {current_key_index + 1}. Switching to Key {next_key + 1}...")
                            current_key_index = next_key
                        else:
                            print(f"  Rate limit hit on Key {key_used + 1}, but it was already rotated by another task. Using current Key {current_key_index + 1}...")
                    
                    if attempt < max_retries - 1:
                        continue # Retry immediately with new key
                else:
                    print("  Rate limit hit. No alternative keys to switch to.")
                    
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
                
            delay = base_delay * (2 ** attempt)
            err_str = str(e)
            if len(err_str) > 200:
                err_str = err_str[:200] + "... [truncated]"
            print(f"  Attempt {attempt + 1} failed: {err_str}")
            print(f"  Retrying in {delay:.0f}s...")
            await asyncio.sleep(delay)

async def abatch_invoke_with_retry(prompt, model_name, pydantic_schema, inputs_list, max_retries=None, base_delay=2.0):
    """Concurrently invoke multiple inputs using the async retry logic."""
    if max_retries is None:
        max_retries = max(len(GROQ_API_KEYS), 5)
    tasks = [
        async_invoke_with_retry(prompt, model_name, pydantic_schema, inputs, max_retries, base_delay)
        for inputs in inputs_list
    ]
    # return_exceptions=True prevents one failed request from bringing down the entire batch
    return await asyncio.gather(*tasks, return_exceptions=True)
