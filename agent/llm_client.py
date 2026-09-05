"""
llm_client.py — Sub-member A

Sends the built prompt to Gemini over HTTPS and parses the JSON response
back into a Python dict. Raises LLMCallError on any failure so
reasoning.py can catch it and fall back to Sub-member B's rule engine.
"""

import os
import json
import requests

try:
    from agent.prompts import build_prompt
except ImportError:
    from prompts import build_prompt

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# Keep the request tight — we don't need a long response, just one JSON object
REQUEST_TIMEOUT_SECONDS = 8


def _get_api_key() -> str | None:
    """Retrieve API key from environment variables or .env file if available."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    if key:
        return key.strip()
    
    # Check for .env file in project root or current working dir
    env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.getcwd(), ".env")
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k in ("GEMINI_API_KEY", "LLM_API_KEY") and v:
                            os.environ[k] = v
                            return v
            except Exception:
                pass
    return None


class LLMCallError(Exception):
    """Raised whenever the LLM path can't be trusted to produce a result —
    network failure, bad status code, or a response that isn't valid JSON
    in the expected shape. reasoning.py catches this and calls the
    fallback instead."""
    pass


def call_llm(context: dict) -> dict:
    """
    Takes the context dict, builds the prompt, calls Gemini, and returns
    a dict with keys: decision, reasoning, offer.

    Raises LLMCallError if anything goes wrong, so the caller can fall
    back to fallback_rules.fallback_score() without crashing the app.
    """
    api_key = _get_api_key()
    if not api_key:
        raise LLMCallError("Missing GEMINI_API_KEY or LLM_API_KEY environment variable")

    prompt = build_prompt(context)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,       # low temperature — we want consistent judgment, not creativity
            "response_mime_type": "application/json",
        },
    }

    try:
        response = requests.post(
            f"{GEMINI_URL}?key={api_key}",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise LLMCallError(f"Network error calling Gemini: {e}") from e

    if response.status_code != 200:
        raise LLMCallError(
            f"Gemini returned status {response.status_code}: {response.text[:200]}"
        )

    try:
        data = response.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, ValueError) as e:
        raise LLMCallError(f"Unexpected Gemini response shape: {e}") from e

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise LLMCallError(f"Gemini response wasn't valid JSON: {e}") from e

    for key in ("decision", "reasoning", "offer"):
        if key not in result:
            raise LLMCallError(f"Gemini response missing required key: {key}")

    return result
