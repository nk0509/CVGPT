"""CV GPT: upload a CV, see your best-matching jobs, score your fit, get CV edits.

Run with:  streamlit run app.py
"""
import hashlib
from pathlib import Path

import streamlit as st

from analysis import score_match, suggest_changes
from cv_parser import extract_text, parse_cv
from jobs import find_skills, job_text, load_jobs, rank_jobs
from llm import check_setup, provider_note

PAGES = ["Top matches", "Match score", "CV changes"]
SAMPLE_CV = Path(__file__).parent / "data" / "sample_cv.txt"

st.set_page_config(page_title="CV GPT", layout="wide")
st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("target", None)
st.session_state.setdefault("results", {})


# ---------- cached work ----------

@st.cache_data(show_spinner=False)
def cached_parse(cv_text):
    return parse_cv(cv_text)


@st.cache_data(show_spinner=False)
def cached_rank(cv_text, use_embeddings):
    return rank_jobs(cv_text, load_jobs(), use_embeddings)


# ---------- helpers ----------

def go_to(page, target=None):
    """Button callback: switch page and remember which role we're looking at."""
    st.session_state.page = page
    if target is not None:
        st.session_state.target = target


def target_info(target, jobs_by_id):
    """Return (label, text sent to the model, cache key) for the current target."""
    if not target:
        return None, None, None
    if target["kind"] == "job":
        job = jobs_by_id.get(target["id"])
        if not job:
            return None, None, None
        return f"{job['title']} at {job['company']}", job_text(job), "job:" + job["id"]
    role, jd = target.get("role", ""), target.get("jd", "")
    text = f"Role: {role or '(not named)'}\n"
    text += f"Job description:\n{jd}" if jd else "No job description given: judge against typical requirements for this role."
    key = "role:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return role or "the pasted job description", text, key


def score_label(score):
    if score >= 75:
        return "Strong match"
    if score >= 50:
        return "Partial match"
    return "Weak match"


def show_list(title, items, empty="None"):
    st.markdown(f"**{title}**")
    if items:
        st.markdown("\n".join(f"- {i}" for i in items))
    else:
        st.caption(empty)


# ---------- header and setup check ----------

st.title("CV GPT")
st.caption("Upload your CV to find your best-matching roles, score your fit for any role, and get tailored CV edits. "
           + provider_note())

chat_ready, embed_ready, setup_msg = check_setup()
if not chat_ready:
    st.error(setup_msg)
    st.stop()
if setup_msg:
    st.info(setup_msg)

# ---------- sidebar: the CV ----------

with st.sidebar:
    st.header("Your CV")
    upload = st.file_uploader("PDF, Word (.docx) or .txt", type=["pdf", "docx", "txt"], key="cv_upload")
    use_sample = st.toggle("Use the sample CV instead", value=upload is None and SAMPLE_CV.exists(),
                           help="A made-up CV for trying the app without your own.")

cv_text, cv_name = None, None
if upload is not None and not use_sample:
    try:
        cv_text, cv_name = extract_text(upload, upload.name), upload.name
    except Exception as err:
        st.sidebar.error(f"Couldn't read that file: {err}")
elif use_sample and SAMPLE_CV.exists():
    cv_text, cv_name = SAMPLE_CV.read_text(encoding="utf-8"), "Sample CV (made up)"

if not cv_text:
    st.info("Upload your CV in the sidebar, or switch on the sample CV, to get started.")
    st.stop()
if len(cv_text.split()) < 50:
    st.error("That file has almost no readable text. If it's a scanned image, save it as a Word file or a text-based PDF.")
    st.stop()

with st.sidebar:
    with st.spinner("Reading your CV..."):
        try:
            profile = cached_parse(cv_text)
        except Exception as err:
            st.error(f"The AI couldn't read the CV: {err}")
            st.stop()
    st.success(f"Loaded: {cv_name}")
    if profile["name"]:
        st.markdown(f"**{profile['name']}**")
    if profile["headline"]:
        st.caption(profile["headline"])
    if profile["total_years_experience"]:
        st.markdown(f"Experience: about {profile['total_years_experience']:g} years")
    skills = sorted(find_skills(cv_text))
    if skills:
        st.markdown("**Skills found:** " + ", ".join(skills))
    with st.expander("Extracted CV text"):
        st.text(cv_text[:6000])

# ---------- page switcher ----------

jobs = load_jobs()
jobs_by_id = {j["id"]: j for j in jobs}
ranked, method = cached_rank(cv_text, embed_ready)

st.radio("Page", PAGES, key="page", horizontal=True, label_visibility="collapsed")
page = st.session_state.page
results = st.session_state.results

# ---------- page 1: top matches ----------

if page == "Top matches":
    st.subheader("Your top 5 matches")
    st.caption(f"Quick fit blends {'meaning-based similarity' if method == 'embeddings' else 'keyword similarity'} "
               f"with skill overlap across {len(jobs)} listings. Open a job for the detailed AI score.")
    for rank, job in enumerate(ranked[:5], start=1):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"**{rank}. {job['title']}**  \n{job['company']} · {job['location']}"
                            + (f" · {job['salary']}" if job.get("salary") else ""))
                st.progress(job["quick_score"] / 100, text=f"Quick fit: {job['quick_score']}%")
                st.caption("You have: " + (", ".join(job["matched"]) or "no listed skills"))
                if job["missing"]:
                    st.caption("Missing: " + ", ".join(job["missing"]))
            with right:
                target = {"kind": "job", "id": job["id"]}
                st.button("Full score", key=f"score-{job['id']}", on_click=go_to, args=("Match score", target),
                          width="stretch")
                st.button("Tailor CV", key=f"tailor-{job['id']}", on_click=go_to, args=("CV changes", target),
                          width="stretch")
                if job.get("url"):
                    st.link_button("Open job", job["url"], width="stretch")

    st.subheader("All open roles")
    c1, c2 = st.columns([3, 1])
    query = c1.text_input("Search title, company or skill", key="job_search")
    region = c2.selectbox("Location", ["All", "Ireland", "India"], key="job_region")
    shown = [j for j in ranked
             if (region == "All" or j["country"] == region)
             and (not query or query.lower() in (j["title"] + j["company"] + j["summary"]).lower())]
    st.dataframe(
        [{"Quick fit %": j["quick_score"], "Role": j["title"], "Company": j["company"],
          "Location": j["location"], "Posted": j["posted"], "Link": j.get("url", "")} for j in shown],
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")},
        hide_index=True, width="stretch",
    )

# ---------- page 2: match score ----------

elif page == "Match score":
    st.subheader("How well does your CV match a role?")
    current = st.session_state.target
    ids = [j["id"] for j in jobs]
    default_index = ids.index(current["id"]) + 1 if current and current["kind"] == "job" else 0
    c1, c2 = st.columns(2)
    picked = c1.selectbox("Pick a listed job", [None] + ids, index=default_index,
                          format_func=lambda i: "Choose a job..." if i is None
                          else f"{jobs_by_id[i]['title']} · {jobs_by_id[i]['company']}")
    role = c2.text_input("Or type any role", placeholder="e.g. Junior Data Analyst at a retailer")
    jd = st.text_area("Paste the full job description (optional, gives a sharper score)", height=120)

    if role or jd:
        st.session_state.target = {"kind": "role", "role": role, "jd": jd}
    elif picked:
        st.session_state.target = {"kind": "job", "id": picked}
    else:
        st.session_state.target = None
    label, target_text, key = target_info(st.session_state.target, jobs_by_id)

    if st.button("Score my CV", type="primary", disabled=not key):
        with st.spinner("Comparing your CV with the role. This can take up to a minute..."):
            try:
                results["score:" + key] = score_match(cv_text, target_text)
            except Exception as err:
                st.error(f"Scoring failed: {err}")

    res = results.get("score:" + key) if key else None
    if res:
        st.divider()
        m1, m2 = st.columns([1, 2])
        m1.metric(f"Match for {label}", f"{res['score']}%", score_label(res["score"]), delta_color="off")
        with m2:
            st.markdown(res["verdict"])
            for part, value in res["breakdown"].items():
                st.progress(value / 100, text=f"{part.capitalize()}: {value}")
        c1, c2 = st.columns(2)
        with c1:
            show_list("On your CV", res["matched"])
            show_list("Strengths", res["strengths"])
        with c2:
            show_list("Missing or not shown", res["missing"], "Nothing major")
            show_list("Gaps", res["gaps"], "Nothing major")
        st.button("See CV changes for this role", on_click=go_to, args=("CV changes",), type="primary")
        st.caption("AI estimate. Treat it as a guide, not a hiring decision.")
    elif not key:
        st.caption("Pick a job or type a role, then press Score my CV.")

# ---------- page 3: CV changes ----------

else:
    st.subheader("What to change on your CV")
    label, target_text, key = target_info(st.session_state.target, jobs_by_id)
    if not key:
        st.info("Choose a role first: press Tailor CV on a job, or pick one on the Match score page.")
        st.stop()
    st.markdown(f"Target: **{label}**")
    scored = results.get("score:" + key)

    if st.button("Suggest CV changes", type="primary"):
        with st.spinner("Rewriting your CV for this role. This can take up to a minute and a half..."):
            try:
                results["changes:" + key] = suggest_changes(cv_text, target_text, scored["missing"] if scored else None)
            except Exception as err:
                st.error(f"Couldn't generate changes: {err}")

    res = results.get("changes:" + key)
    if res:
        if res["projected_score"] is not None:
            before = f"{scored['score']}% now → " if scored else ""
            st.caption(f"Estimated match: {before}{res['projected_score']}% after these edits")
        if res["headline"]:
            st.markdown(f"**Headline:** {res['headline']}")
        if res["summary"]:
            st.markdown("**Professional summary**")
            st.write(res["summary"])
        st.caption("Anything in [square brackets] is a figure only you know. Fill it in or delete it.")

        for change in res["changes"]:
            with st.container(border=True):
                st.markdown(f"**{change['section']}** · {change['priority']} priority")
                before_col, after_col = st.columns(2)
                before_col.caption("Now")
                before_col.write(change["current"])
                after_col.caption("Change to")
                after_col.write(change["suggested"])
                if change["why"]:
                    st.caption(change["why"])

        c1, c2 = st.columns(2)
        with c1:
            show_list("Keywords to add (only if true)", res["add_keywords"])
        with c2:
            show_list("Cut or shorten", res["remove"], "Nothing")

        lines = [f"CV changes for {label}", "", f"HEADLINE: {res['headline']}", "", "SUMMARY:", res["summary"], ""]
        for c in res["changes"]:
            lines += [f"[{c['priority'].upper()}] {c['section']}", f"Now: {c['current']}",
                      f"Change to: {c['suggested']}", f"Why: {c['why']}", ""]
        lines += ["Add keywords: " + ", ".join(res["add_keywords"]), "Cut: " + "; ".join(res["remove"])]
        st.download_button("Download these changes (.txt)", "\n".join(lines), file_name="cv_changes.txt")
