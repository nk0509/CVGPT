# CV GPT

Upload a CV and CV GPT will:

1. **Rank job listings** against it and show your top 5 matches
2. **Score your fit** for any listed job, or any role you type or paste
3. **Suggest CV changes** for that role, as before/after rewrites that never invent experience

Everything runs on a local open-source LLM through [Ollama](https://ollama.com), so it costs nothing to run and no CV leaves your machine.

## How it works

```
CV (PDF/DOCX/TXT) ──► text extraction ──► LLM ──► structured profile (JSON)
                                │
Job listings ──► embeddings (computed once, cached)
                                │
                     cosine similarity + skill overlap ──► Top 5 (instant, no LLM calls)
                                │
              chosen role ──► LLM ──► match score, breakdown, gaps
                                │
                    gaps ──► LLM ──► grounded CV rewrites
```

| File | What it does |
|---|---|
| `app.py` | Streamlit interface with three pages: Top matches, Match score, CV changes |
| `cv_parser.py` | Reads PDF/Word/text CVs and extracts a structured profile with the LLM |
| `jobs.py` | Loads listings and ranks them with embeddings plus skill overlap |
| `analysis.py` | Detailed match scoring and CV rewrite prompts, with output validation |
| `llm.py` | The only file that talks to the model: JSON mode, retries, disk cache |
| `data/jobs.json` | 30 data, analytics and AI roles in Ireland and Bengaluru (Oct 2026) |

## Design decisions

- **Retrieve, then reason.** Ranking every job with an LLM would be slow, so jobs are ranked with cheap signals (embedding similarity and skill overlap), and only the job you open gets a detailed LLM score. This is the same retrieve-then-rerank pattern used in RAG.
- **Reliable structured output.** Every model call uses JSON mode at temperature 0. Replies are validated and clamped into a fixed shape, with one automatic retry if the JSON is invalid.
- **Context control.** CVs and job descriptions are trimmed to fixed limits, and the model's context window is set explicitly, so prompts never silently overflow.
- **Grounded rewrites.** The rewrite prompt forbids inventing employers, dates, tools or numbers. Missing figures come back as `[placeholders]` for the user to fill.
- **Caching.** Model answers and job embeddings are cached on disk, so repeating a request is instant and free.
- **Provider-agnostic.** All model access is in `llm.py`, so moving to Azure OpenAI, Gemini or Claude means changing one file.

## Run it locally (Windows)

1. Install [Ollama](https://ollama.com) and [Python 3.11+](https://www.python.org).
2. Download the models (one time):
   ```
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Install the Python packages:
   ```
   pip install -r requirements.txt
   ```
4. Start the app:
   ```
   streamlit run app.py
   ```
   It opens in your browser. Upload a CV, or switch on the sample CV.

## Limitations

- A small local model (Llama 3.2, 3B) is fast and free but less precise than large hosted models. Treat scores as guidance.
- Job listings are a fixed snapshot. A live jobs API (such as Adzuna) is the next step.

## Next steps

- Live job listings from a jobs API, refreshed daily
- An evaluation set of hand-scored CV and job pairs to measure scoring accuracy
- FastAPI backend with PostgreSQL and pgvector, deployed with Docker
