"""Talks to the LLM. Everything that calls a model goes through this file.

Providers, picked automatically from the keys you set (in Streamlit secrets
or environment variables):
- Groq (free hosted open-source models) when GROQ_API_KEY is set
- Google Gemini when GEMINI_API_KEY is set (and no Groq key)
- Ollama on your own computer when no key is set

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
# Tried in order. Groq retires models from time to time, so a retired or
# rate-limited model is skipped automatically. Add a GROQ_MODEL secret to try another first.
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"  # override with a GEMINI_MODEL secret

CACHE_DIR = Path(__file__).parent / "cache"


def _secret(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _groq_key():
    return _secret("GROQ_API_KEY")


def provider():
    if _secret("GROQ_API_KEY"):
        return "groq"
    if _secret("GEMINI_API_KEY"):
        return "gemini"
    return "ollama"


def online():
    return provider() != "ollama"


def provider_note():
    name = provider()
    if name == "groq":
        return "Runs on open-source AI models hosted by Groq. Your CV is sent to Groq for analysis and isn't stored by this app."
    if name == "gemini":
        return "Runs on Google Gemini. Your CV is sent to Google for analysis and isn't stored by this app."
    return "Runs on a local AI model, so nothing leaves your computer."


def _cache_file(kind, payload):
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{kind}-{key}.json"


def _chat_groq(text):
    models = GROQ_MODELS
    preferred = _secret("GROQ_MODEL")
    if preferred:
        models = [preferred] + [m for m in GROQ_MODELS if m != preferred]
    problems = []
    for model in models:
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
        if resp.status_code == 401:
            raise RuntimeError("The Groq API key was rejected. Check GROQ_API_KEY in the app's secrets.")
        if resp.status_code == 400 and "json_validate_failed" in resp.text:
            return ""  # the model's JSON was invalid; ask_json retries once
        if resp.status_code in (404, 429) or (resp.status_code == 400 and "model" in resp.text.lower()):
            problems.append(f"{model}: {resp.status_code}")  # retired, unavailable or rate limited
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    if all(p.endswith("429") for p in problems):
        raise RuntimeError("The free AI limit is used up for now. Try again in a few minutes.")
    raise RuntimeError("None of the Groq models answered (" + ", ".join(problems) + "). "
                       "Add a GROQ_MODEL secret with a current model name from console.groq.com/docs/models.")


def _chat_gemini(text):
    model = _secret("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    resp = requests.post(
        GEMINI_URL.format(model=model),
        headers={"x-goog-api-key": _secret("GEMINI_API_KEY")},
        json={
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        },
        timeout=120,
    )
    if resp.status_code in (400, 401, 403) and "API key" in resp.text:
        raise RuntimeError("The Gemini API key was rejected. Check GEMINI_API_KEY in the app's secrets.")
    if resp.status_code == 404:
        raise RuntimeError(f"Gemini model '{model}' isn't available to your key. Add a GEMINI_MODEL secret "
                           "with a current Flash model name from Google AI Studio.")
    if resp.status_code == 402:
        raise RuntimeError("Google says this key's project has no credit left. Create a key in a new project "
                           "in Google AI Studio, or use a free Groq key instead.")
    if resp.status_code == 429:
        raise RuntimeError("Gemini's free limit is used up for now. Try again in a few minutes.")
    resp.raise_for_status()
    try:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini returned no answer: {resp.text[:200]}")


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
    name = provider()
    tag = {"groq": "groq", "gemini": "gemini:" + (_secret("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL)}.get(
        name, "ollama:" + LOCAL_CHAT_MODEL)
    cache_file = _cache_file("chat", {"provider": tag, "prompt": prompt})
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    data, last_error = None, None
    for attempt in range(2):
        text = prompt
        if attempt == 1:
            text += "\n\nYour previous reply was not valid JSON. Reply with valid JSON only."
        chat = {"groq": _chat_groq, "gemini": _chat_gemini}.get(name, _chat_ollama)
        reply = chat(text)
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
        # online ranking uses keyword similarity, so no embedding model is needed
        return True, False, ""
    try:
        names = _installed_models()
    except Exception:
        return False, False, (
            "No AI is connected. Online: add GROQ_API_KEY or GEMINI_API_KEY in the app's Secrets settings. "
            "On your laptop: open the Ollama app from the Start menu, then refresh this page."
        )
    if LOCAL_CHAT_MODEL not in names:
        return False, False, f"The AI model isn't downloaded yet. In a terminal, run:  ollama pull {LOCAL_CHAT_MODEL}"
    if EMBED_MODEL not in names:
        return True, False, (
            f"Job ranking is using keyword matching. For smarter ranking, run:  ollama pull {EMBED_MODEL}"
        )
    return True, True, ""
