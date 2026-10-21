"""Talks to the LLM. Everything that calls a model goes through this file.

Two providers, picked automatically:
- Online: Groq's free hosted Llama models, used when a GROQ_API_KEY is set
  (in Streamlit secrets or an environment variable).
- Local: Ollama on your own computer, used when no key is set.

Reliability features:
- JSON mode forces valid JSON, temperature 0 keeps answers consistent
- one automatic retry if the reply still isn't valid JSON
- if the main online model hits its daily limit, a smaller one takes over
- a disk cache: the same question never costs a second model call
"""
import hashlib
import json
import os
from pathlib import Path

import requests

LOCAL_CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
CONTEXT_TOKENS = 8192  # Ollama's default window is small; CVs need more room

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]  # main, then fallback

CACHE_DIR = Path(__file__).parent / "cache"


def _groq_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def online():
    return bool(_groq_key())


def provider_note():
    if online():
        return "Runs on Llama 3.3 hosted by Groq. Your CV is sent to Groq for analysis and isn't stored by this app."
    return "Runs on a local AI model, so nothing leaves your computer."


def _cache_file(kind, payload):
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{kind}-{key}.json"


def _chat_groq(text):
    last = None
    for model in GROQ_MODELS:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {_groq_key()}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": text}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        if resp.status_code == 429:  # rate or daily limit: try the next model
            last = resp
            continue
        if resp.status_code == 401:
            raise RuntimeError("The Groq API key was rejected. Check GROQ_API_KEY in the app's secrets.")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError("The free AI limit is used up for now. Try again in a few minutes. "
                       f"(Groq said: {last.text[:200] if last is not None else 'rate limited'})")


def _chat_ollama(text):
    import ollama
    resp = ollama.chat(
        model=LOCAL_CHAT_MODEL,
        messages=[{"role": "user", "content": text}],
        format="json",
        options={"temperature": 0, "num_ctx": CONTEXT_TOKENS},
    )
    return resp["message"]["content"]


def ask_json(prompt, use_cache=True):
    """Send a prompt and return the model's reply parsed as JSON."""
    provider = "groq" if online() else "ollama:" + LOCAL_CHAT_MODEL
    cache_file = _cache_file("chat", {"provider": provider, "prompt": prompt})
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    data, last_error = None, None
    for attempt in range(2):
        text = prompt
        if attempt == 1:
            text += "\n\nYour previous reply was not valid JSON. Reply with valid JSON only."
        reply = _chat_groq(text) if online() else _chat_ollama(text)
        try:
            data = json.loads(reply)
            break
        except json.JSONDecodeError as err:
            last_error = err
    if data is None:
        raise ValueError(f"The model didn't return valid JSON: {last_error}")

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def embed(texts, model=EMBED_MODEL):
    """Turn a list of texts into embedding vectors (local Ollama only)."""
    import ollama
    resp = ollama.embed(model=model, input=texts)
    return resp["embeddings"]


def _installed_models():
    import ollama
    names = set()
    for m in ollama.list().models:
        name = getattr(m, "model", None) or m.get("name", "")
        names.add(name.split(":")[0])
    return names


def check_setup():
    """Return (chat_ready, embeddings_ready, message) for the setup banner."""
    if online():
        # Groq has no embedding models, so ranking uses keyword similarity online
        return True, False, ""
    try:
        names = _installed_models()
    except Exception:
        return False, False, (
            "Ollama isn't running. Open the Ollama app from the Start menu, then refresh this page."
        )
    if LOCAL_CHAT_MODEL not in names:
        return False, False, f"The AI model isn't downloaded yet. In a terminal, run:  ollama pull {LOCAL_CHAT_MODEL}"
    if EMBED_MODEL not in names:
        return True, False, (
            f"Job ranking is using keyword matching. For smarter ranking, run:  ollama pull {EMBED_MODEL}"
        )
    return True, True, ""
