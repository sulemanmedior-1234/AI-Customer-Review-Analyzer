"""
AI Customer Review Analyzer
Upload a CSV of product reviews. The app classifies each review's sentiment,
finds the most common complaints, writes a summary of the feedback, and shows
charts.

Uses the Groq REST API directly through `requests`, so the `groq` package is
not needed.

Run locally:  streamlit run app.py
"""

import json
import os
import re

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AI Review Analyzer", page_icon="📊", layout="wide")

# HELPERS START
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
BATCH_SIZE = 20
MAX_TEXT_CHARS = 350

SENTIMENTS = ["Positive", "Neutral", "Negative"]
CATEGORIES = [
    "Product Quality",
    "Price / Value",
    "Shipping / Delivery",
    "Customer Service",
    "Ease of Use",
    "Performance",
    "Packaging",
    "Size / Fit",
    "Other",
]

TEXT_HINTS = ["review", "text", "comment", "feedback", "body", "content", "description", "message"]
RATING_HINTS = ["rating", "stars", "star", "score"]
DATE_HINTS = ["date", "time", "created", "posted"]


def get_api_key(sidebar_value):
    """Priority: sidebar input -> st.secrets -> environment variable."""
    if sidebar_value and sidebar_value.strip():
        return sidebar_value.strip()
    try:
        secret = st.secrets["GROQ_API_KEY"]
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


def read_csv(uploaded):
    """Read an uploaded CSV, trying common encodings."""
    last_error = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise ValueError("Could not read the CSV file: " + str(last_error))


def find_column(columns, hints):
    lowered = {c: str(c).strip().lower() for c in columns}
    for hint in hints:
        for col, name in lowered.items():
            if name == hint:
                return col
    for hint in hints:
        for col, name in lowered.items():
            if hint in name:
                return col
    return None


def detect_columns(df):
    """Guess the review text, rating and date columns."""
    text_col = find_column(df.columns, TEXT_HINTS)
    if text_col is None:
        best, best_len = None, 0
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                avg = df[col].astype(str).str.len().mean()
                if avg > best_len:
                    best, best_len = col, avg
        text_col = best
    rating_col = find_column(df.columns, RATING_HINTS)
    date_col = find_column(df.columns, DATE_HINTS)
    return text_col, rating_col, date_col


def sentiment_from_rating(rating):
    try:
        r = float(rating)
    except Exception:
        return None
    if r >= 4:
        return "Positive"
    if r <= 2:
        return "Negative"
    return "Neutral"


def build_classify_messages(batch):
    """batch is a list of (id, text) pairs."""
    system_msg = (
        "You are a careful customer-feedback analyst. You classify product reviews. "
        "Reply with valid JSON only."
    )
    lines = [str(i) + ": " + t.replace("\n", " ")[:MAX_TEXT_CHARS] for i, t in batch]
    user_msg = (
        "Classify each numbered review below.\n\n"
        "For each review return:\n"
        '- "id": the number of the review\n'
        '- "sentiment": exactly one of Positive, Neutral, Negative\n'
        '- "category": the main topic the review is about, exactly one of: '
        + ", ".join(CATEGORIES) + "\n\n"
        'Return JSON of this shape: {"results": [{"id": 1, "sentiment": "Negative", "category": "Shipping / Delivery"}]}\n'
        "Include every review exactly once.\n\n"
        "REVIEWS:\n" + "\n".join(lines)
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def extract_json(text):
    """Parse JSON even if the model wrapped it in code fences or extra text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("The AI did not return valid JSON.")


def normalize_label(value, allowed, default):
    value = str(value).strip().lower()
    for item in allowed:
        if value == item.lower():
            return item
    for item in allowed:
        if value and (value in item.lower() or item.lower() in value):
            return item
    return default


def parse_classification(raw, batch):
    """Return {id: (sentiment, category)} for the ids in batch."""
    data = extract_json(raw)
    items = data.get("results", []) if isinstance(data, dict) else data
    wanted = {i for i, _ in batch}
    out = {}
    for item in items or []:
        try:
            rid = int(item["id"])
        except Exception:
            continue
        if rid in wanted:
            out[rid] = (
                normalize_label(item.get("sentiment", ""), SENTIMENTS, "Neutral"),
                normalize_label(item.get("category", ""), CATEGORIES, "Other"),
            )
    return out


def build_summary_messages(total, counts, top_complaints, negatives, positives):
    system_msg = (
        "You are a senior customer-experience analyst. You write clear, honest summaries "
        "of customer feedback. Use ONLY the information given. Do not invent numbers."
    )
    user_msg = (
        "Here are statistics and sample reviews from " + str(total) + " customer reviews.\n\n"
        "Sentiment counts: " + json.dumps(counts) + "\n"
        "Most common complaint categories (negative reviews): " + json.dumps(top_complaints) + "\n\n"
        "SAMPLE NEGATIVE REVIEWS:\n" + "\n".join("- " + t for t in negatives) + "\n\n"
        "SAMPLE POSITIVE REVIEWS:\n" + "\n".join("- " + t for t in positives) + "\n\n"
        "Write a report in markdown with exactly these sections:\n"
        "### Overall summary\n(2-4 sentences)\n"
        "### Top complaints\n(a bullet list; for each complaint say what customers actually say)\n"
        "### What customers love\n(a bullet list)\n"
        "### Recommended actions\n(3-5 specific, practical bullet points)\n"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def friendly_error(exc):
    msg = str(exc)
    low = msg.lower()
    if "401" in msg or "invalid api key" in low or "invalid_api_key" in low:
        return "Your Groq API key looks invalid. Please check it and try again."
    if "429" in msg or "rate limit" in low:
        return (
            "Groq rate limit reached. Please wait a minute and try again, "
            "or analyze fewer reviews."
        )
    if "decommissioned" in low or "does not exist" in low or "404" in msg:
        return (
            "That model is not available on your account. Choose another model in the "
            "sidebar. Details: " + msg[:300]
        )
    if "connection" in low or "timeout" in low or "timed out" in low:
        return "Could not reach the Groq API. Check your internet connection and retry."
    if "json" in low:
        return "The AI reply could not be read. Please click Analyze again."
    return "Something went wrong: " + msg
# HELPERS END


@st.cache_data(ttl=600, show_spinner=False)
def fetch_models(api_key):
    """Ask Groq which chat models this API key can use right now."""
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": "Bearer " + api_key},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        skip = ("whisper", "guard", "tts", "playai", "orpheus", "safeguard", "embed")
        ids = [
            m["id"]
            for m in r.json().get("data", [])
            if m.get("active", True) and not any(s in m["id"].lower() for s in skip)
        ]
        return sorted(ids)
    except Exception:
        return []


def call_groq(api_key, model, messages, json_mode):
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)

    # Some models do not support JSON mode: retry once without it.
    if resp.status_code == 400 and "response_format" in resp.text:
        payload.pop("response_format")
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)

    if resp.status_code != 200:
        raise RuntimeError("Groq API error " + str(resp.status_code) + ": " + resp.text[:300])
    choices = resp.json().get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def classify_all(api_key, model, texts, progress):
    """Classify all reviews in batches. Returns {index: (sentiment, category)}."""
    results = {}
    batches = [list(range(i, min(i + BATCH_SIZE, len(texts)))) for i in range(0, len(texts), BATCH_SIZE)]
    for n, idxs in enumerate(batches):
        batch = [(i + 1, texts[i]) for i in idxs]
        try:
            raw = call_groq(api_key, model, build_classify_messages(batch), True)
            parsed = parse_classification(raw, batch)
        except RuntimeError:
            raise
        except Exception:
            parsed = {}
        for rid, value in parsed.items():
            results[rid - 1] = value
        progress.progress((n + 1) / len(batches), text="Analyzing reviews... " + str(n + 1) + "/" + str(len(batches)))
    return results


SAMPLE_CSV = (
    "review,rating,date\n"
    '"Great product, works exactly as described and feels well made.",5,2026-08-03\n'
    '"Arrived two weeks late and the box was damaged.",2,2026-08-09\n'
    '"Customer support never answered my emails. Very disappointing.",1,2026-08-14\n'
    '"Good value for the price, I would buy again.",4,2026-09-02\n'
    '"It stopped working after a week. Poor quality.",1,2026-09-11\n'
    '"Easy to set up and simple to use.",5,2026-09-18\n'
    '"Okay product, nothing special.",3,2026-10-01\n'
    '"Too expensive for what you get.",2,2026-10-05\n'
    '"Fast delivery and excellent packaging!",5,2026-10-12\n'
    '"The size runs small, I had to return it.",2,2026-10-20\n'
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
for key, value in {"analysis": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    key_input = st.text_input(
        "Groq API key (optional)",
        type="password",
        help="Leave empty to use st.secrets or the GROQ_API_KEY environment variable.",
    )
    api_key = get_api_key(key_input)
    if api_key:
        st.success("API key found ✅")
    else:
        st.warning("No API key found yet")

    live_models = fetch_models(api_key) if api_key else []
    model_options = live_models or MODELS
    default_index = 0
    for preferred in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
        if preferred in model_options:
            default_index = model_options.index(preferred)
            break
    model = st.selectbox("Groq model", model_options, index=default_index)
    if live_models:
        st.caption("Models loaded from your Groq account.")

    max_reviews = st.slider("Max reviews to analyze", 10, 300, 100, step=10)
    st.caption("More reviews take longer and use more of your Groq limit.")

    st.divider()
    st.download_button(
        "📄 Download sample CSV",
        data=SAMPLE_CSV,
        file_name="sample_reviews.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("📊 AI Customer Review Analyzer")
st.write(
    "Upload a CSV of product reviews. The app finds the sentiment of each review, "
    "the most common complaints, and writes a summary with recommended actions."
)

uploaded = st.file_uploader("Upload reviews (CSV)", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to begin, or download the sample CSV from the sidebar to try the app.")
    st.stop()

try:
    df = read_csv(uploaded)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("The CSV file is empty.")
    st.stop()

text_guess, rating_guess, date_guess = detect_columns(df)
none_label = "(none)"
cols = list(df.columns)

st.subheader("1. Check your columns")
c1, c2, c3 = st.columns(3)
with c1:
    text_col = st.selectbox(
        "Review text column", cols, index=cols.index(text_guess) if text_guess in cols else 0
    )
with c2:
    rating_options = [none_label] + cols
    rating_col = st.selectbox(
        "Rating column (optional)",
        rating_options,
        index=rating_options.index(rating_guess) if rating_guess in rating_options else 0,
    )
with c3:
    date_options = [none_label] + cols
    date_col = st.selectbox(
        "Date column (optional)",
        date_options,
        index=date_options.index(date_guess) if date_guess in date_options else 0,
    )

with st.expander("Preview of your file", expanded=False):
    st.dataframe(df.head(10))

if st.button("✨ Analyze reviews", type="primary"):
    work = pd.DataFrame({"Review": df[text_col].astype(str).str.strip()})
    work["Rating"] = pd.to_numeric(df[rating_col], errors="coerce") if rating_col != none_label else None
    work["Date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col != none_label else pd.NaT
    work = work[(work["Review"] != "") & (work["Review"].str.lower() != "nan")].reset_index(drop=True)

    if not api_key:
        st.error(
            "No Groq API key found. Enter it in the sidebar, or add GROQ_API_KEY "
            "to your Streamlit secrets."
        )
    elif work.empty:
        st.warning("No review text was found in the selected column.")
    else:
        if len(work) > max_reviews:
            st.warning(
                "Your file has " + str(len(work)) + " reviews. Only the first "
                + str(max_reviews) + " will be analyzed (change this in the sidebar)."
            )
            work = work.head(max_reviews).reset_index(drop=True)

        try:
            progress = st.progress(0.0, text="Analyzing reviews...")
            found = classify_all(api_key, model, work["Review"].tolist(), progress)
            progress.empty()

            sentiments, categories, missing = [], [], 0
            for i in range(len(work)):
                if i in found:
                    s, c = found[i]
                else:
                    missing += 1
                    s = sentiment_from_rating(work.loc[i, "Rating"]) or "Neutral"
                    c = "Other"
                sentiments.append(s)
                categories.append(c)
            work["Sentiment"] = sentiments
            work["Category"] = categories

            with st.spinner("Writing the summary..."):
                negs = work[work["Sentiment"] == "Negative"]["Review"].head(25).str.slice(0, 250).tolist()
                poss = work[work["Sentiment"] == "Positive"]["Review"].head(15).str.slice(0, 250).tolist()
                counts = work["Sentiment"].value_counts().to_dict()
                top = (
                    work[work["Sentiment"] == "Negative"]["Category"].value_counts().head(5).to_dict()
                )
                summary = call_groq(
                    api_key,
                    model,
                    build_summary_messages(len(work), counts, top, negs, poss),
                    False,
                )

            st.session_state["analysis"] = {"data": work, "summary": summary, "missing": missing}
        except Exception as exc:
            progress_error = friendly_error(exc)
            st.error(progress_error)

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
analysis = st.session_state["analysis"]
if analysis is not None:
    data = analysis["data"]
    total = len(data)

    st.success("Analysis complete! 🎉")
    if analysis["missing"]:
        st.caption(
            str(analysis["missing"]) + " review(s) could not be classified by the AI "
            "and were labelled using their rating (or Neutral)."
        )

    pos = int((data["Sentiment"] == "Positive").sum())
    neg = int((data["Sentiment"] == "Negative").sum())
    neu = int((data["Sentiment"] == "Neutral").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reviews analyzed", total)
    m2.metric("Positive", str(round(pos / total * 100)) + "%")
    m3.metric("Negative", str(round(neg / total * 100)) + "%")
    if data["Rating"].notna().any():
        m4.metric("Average rating", format(data["Rating"].mean(), ".2f"))
    else:
        m4.metric("Neutral", str(round(neu / total * 100)) + "%")

    tab_sum, tab_charts, tab_data = st.tabs(["📝 Summary", "📈 Charts", "🧾 All reviews"])

    with tab_sum:
        st.markdown(analysis["summary"])

    with tab_charts:
        left, right = st.columns(2)
        with left:
            st.subheader("Sentiment")
            sent_counts = data["Sentiment"].value_counts().reindex(SENTIMENTS).fillna(0)
            st.bar_chart(sent_counts)
        with right:
            st.subheader("Most common complaints")
            comp = data[data["Sentiment"] == "Negative"]["Category"].value_counts()
            if comp.empty:
                st.info("No negative reviews were found.")
            else:
                st.bar_chart(comp)

        left2, right2 = st.columns(2)
        with left2:
            if data["Rating"].notna().any():
                st.subheader("Rating distribution")
                st.bar_chart(data["Rating"].dropna().round().astype(int).value_counts().sort_index())
        with right2:
            if data["Date"].notna().any():
                st.subheader("Sentiment over time")
                timed = data.dropna(subset=["Date"]).copy()
                timed["Month"] = timed["Date"].dt.strftime("%Y-%m")
                trend = timed.groupby(["Month", "Sentiment"]).size().unstack(fill_value=0)
                trend = trend.reindex(columns=SENTIMENTS, fill_value=0)
                st.bar_chart(trend)

    with tab_data:
        choice = st.multiselect("Filter by sentiment", SENTIMENTS, default=SENTIMENTS)
        shown = data[data["Sentiment"].isin(choice)]
        st.dataframe(shown)
        st.download_button(
            "⬇️ Download results (CSV)",
            data=data.to_csv(index=False),
            file_name="review_analysis.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Download summary (.md)",
            data=analysis["summary"],
            file_name="review_summary.md",
            mime="text/markdown",
        )
