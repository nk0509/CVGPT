"""Step 2: load job listings and rank them against a CV without any LLM calls.

Ranking is two cheap signals blended together:
- semantic similarity between CV and job (embeddings, or a keyword fallback)
- skill overlap: which of the job's skills the CV actually mentions
Only the jobs you open get the slower, detailed LLM score (see analysis.py).
"""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from llm import CACHE_DIR, EMBED_MODEL, embed

DATA_DIR = Path(__file__).parent / "data"

# Canonical skill -> words that count as a mention of it
SKILLS = {
    "Python": ["python"],
    "SQL": ["sql", "mysql", "sql server", "postgresql", "postgres"],
    "R": ["r"],
    "Excel": ["excel"],
    "Power BI": ["power bi"],
    "Tableau": ["tableau"],
    "DAX": ["dax"],
    "Spark": ["spark", "pyspark"],
    "Databricks": ["databricks"],
    "Azure": ["azure"],
    "AWS": ["aws"],
    "Google Cloud": ["gcp", "google cloud", "vertex ai", "bigquery"],
    "Azure Data Factory": ["azure data factory", "adf"],
    "Microsoft Fabric": ["fabric"],
    "Machine learning": ["machine learning", "ml"],
    "Deep learning": ["deep learning", "cnn", "cnns"],
    "NLP": ["nlp", "natural language processing", "text classification", "sentiment"],
    "Computer vision": ["computer vision", "vision", "opencv", "yolo", "yolov5"],
    "LLMs": ["llm", "llms", "large language model", "large language models"],
    "Generative AI": ["generative ai", "genai", "gen ai"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "Statistics": ["statistics", "statistical"],
    "A/B testing": ["a/b testing", "a/b tests", "a/b test", "experimentation", "experiments"],
    "Forecasting": ["forecasting", "forecast"],
    "Recommender systems": ["recommendation", "recommender", "recommendations"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow", "keras"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "Transformers": ["transformer", "transformers", "bert", "hugging face"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "MLOps": ["mlops", "mlflow", "model monitoring"],
    "CI/CD": ["ci/cd", "ci cd"],
    "Data modelling": ["data modelling", "data modeling", "dimensional modelling", "dimensional modeling"],
    "Data warehousing": ["data warehouse", "data warehousing", "lakehouse"],
    "ETL / pipelines": ["etl", "data pipeline", "data pipelines", "pipelines"],
    "Dashboards": ["dashboard", "dashboards", "reporting"],
    "Data visualisation": ["visualisation", "visualization", "matplotlib"],
    "Stakeholder management": ["stakeholder", "stakeholders", "client-facing"],
    "SAP": ["sap"],
    "Agile": ["agile", "scrum", "sprint", "sprints"],
    "Search & ranking": ["search", "ranking"],
    "AI agents": ["agentic", "multi-agent", "agents"],
}


def _norm(text):
    """Lowercase and turn punctuation into spaces so phrases match reliably."""
    return " " + re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip() + " "


_ALIASES = {skill: [_norm(a) for a in aliases] for skill, aliases in SKILLS.items()}


def find_skills(text):
    padded = _norm(text)
    return {skill for skill, aliases in _ALIASES.items() if any(a in padded for a in aliases)}


def load_jobs():
    return json.loads((DATA_DIR / "jobs.json").read_text(encoding="utf-8"))


def job_text(job):
    return f"{job['title']} at {job['company']} ({job['location']}). {job['summary']}"


# ---------- similarity ----------

def _cosine(matrix, vector):
    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(vector, dtype=float)
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    denom[denom == 0] = 1.0
    return matrix @ vector / denom


def _job_embeddings(jobs):
    """Embed every job once and keep the vectors on disk."""
    texts = [job_text(j) for j in jobs]
    key = hashlib.sha256((EMBED_MODEL + "".join(texts)).encode("utf-8")).hexdigest()[:16]
    cache_file = CACHE_DIR / f"job-embeddings-{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    vectors = embed(texts)
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(vectors), encoding="utf-8")
    return vectors


def _tfidf_similarity(cv_text, texts):
    """Keyword-based fallback when no embedding model is installed."""
    docs = [_norm(t).split() for t in texts]
    cv_words = _norm(cv_text).split()
    n = len(docs) + 1
    df = Counter(w for d in docs + [cv_words] for w in set(d))
    vocab = {w: i for i, w in enumerate(df)}

    def vec(words):
        v = np.zeros(len(vocab))
        for w, c in Counter(words).items():
            v[vocab[w]] = c * math.log(n / df[w])
        return v

    return _cosine([vec(d) for d in docs], vec(cv_words))


def rank_jobs(cv_text, jobs, use_embeddings=True):
    """Score every job against the CV and return them best first."""
    texts = [job_text(j) for j in jobs]
    method = "keywords"
    if use_embeddings:
        try:
            sims = _cosine(_job_embeddings(jobs), embed([cv_text[:8000]])[0])
            # embedding similarity for related texts sits roughly between 0.35 and 0.8
            scaled = np.clip((sims - 0.35) / 0.45, 0, 1)
            method = "embeddings"
        except Exception:
            use_embeddings = False
    if not use_embeddings:
        sims = _tfidf_similarity(cv_text, texts)
        scaled = np.clip(sims / 0.3, 0, 1)

    cv_skills = find_skills(cv_text)
    ranked = []
    for job, text, sim in zip(jobs, texts, scaled):
        wanted = find_skills(text)
        matched = sorted(wanted & cv_skills)
        missing = sorted(wanted - cv_skills)
        overlap = len(matched) / len(wanted) if wanted else 0.0
        quick = round(100 * (0.5 * float(sim) + 0.5 * overlap))
        ranked.append({**job, "quick_score": quick, "matched": matched, "missing": missing})
    ranked.sort(key=lambda j: j["quick_score"], reverse=True)
    return ranked, method
