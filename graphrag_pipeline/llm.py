"""
LLM Client and Prompt Management
- Groq API client with retry logic
- Prompt template loading
"""

import time
from pathlib import Path
from functools import lru_cache
from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_RETRIES, GROQ_BASE_BACKOFF_SECONDS


# ============================================================
# GROQ CLIENT INITIALIZATION
# ============================================================

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None


# ============================================================
# LLM CALL WITH RETRY LOGIC
# ============================================================

def llm_call(prompt: str) -> str:
    """Call Groq LLM with retry logic and return generated text"""
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY is not set. Please configure your Groq credentials.")
    
    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            choice = response.choices[0].message.content if response.choices else ""
            return choice or ""
        except Exception as e:
            err_msg = str(e)
            print(f"LLM call error (attempt {attempt}/{GROQ_MAX_RETRIES}): {err_msg}")
            if attempt == GROQ_MAX_RETRIES:
                break
            backoff = GROQ_BASE_BACKOFF_SECONDS * attempt
            time.sleep(backoff)
    
    return ""


# ============================================================
# PROMPT TEMPLATE LOADING
# ============================================================

@lru_cache(maxsize=4)
def _load_prompt_template(filename: str) -> str:
    """Load prompt template from external file with caching"""
    template_path = Path(__file__).resolve().parent / filename
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8").strip()


def generate_answer(query, fused_context):
    """Generate final answer using LLM with prompt template"""
    template = _load_prompt_template("graph_rag_prompt.md")
    prompt = (
        f"{template}\n\n"
        f"Context:\n{fused_context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    
    return llm_call(prompt)
