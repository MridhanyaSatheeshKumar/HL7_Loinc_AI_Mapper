import pickle
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("data/processed/loinc_cleaned.csv")

with open("data/processed/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/processed/matrix.pkl", "rb") as f:
    X = pickle.load(f)


# ✅ Clean camelCase (important for your Excel data)
def clean_input(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.lower()


# ✅ Expand query with domain knowledge
def expand_query(query):
    mapping = {
        "cycling": "bicycle riding distance exercise physical activity",
        "floors climbed": "stairs climbed stair climbing physical activity",
        "stairs": "stairs climbed physical activity",
        "walking": "walking distance steps physical activity",
        "running": "running distance speed physical activity",
        "step count": "steps walking activity count",
        "vo2 max": "oxygen consumption fitness exercise capacity",
    }

    query = query.lower()

    for key in mapping:
        if key in query:
            query += " " + mapping[key]

    return query

# ✅ Add unit/context awareness
def enrich_query(query):
    if "distance" in query:
        query += " distance length"
    if "count" in query or "steps" in query:
        query += " count number"
    if "vo2" in query:
        query += " oxygen consumption fitness"

    return query


# ✅ Boost relevant concepts
def boost_score(row):
    text = row["LONG_COMMON_NAME"].lower()

    score = 0

    # ✅ Boost good concepts
    if "activity" in text:
        score += 0.3
    if "physical" in text:
        score += 0.2
    if "distance" in text:
        score += 0.2

    # ❌ Penalize survey/question-based LOINC
    if "promis" in text:
        score -= 0.4
    if "does your" in text or "limit you" in text:
        score -= 0.3

    return score

# 🔍 Main search function
def search_loinc(query, top_k=5):
    query = clean_input(query)
    query = expand_query(query)
    query = enrich_query(query)

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X)[0]

    top_idx = scores.argsort()[::-1][:top_k]

    results = df.iloc[top_idx].copy()
    results["score"] = scores[top_idx]

    # Boost scores
    results["score"] = results.apply(
        lambda row: row["score"] + boost_score(row),
        axis=1
    )

    # Re-rank
    results = results.sort_values(by="score", ascending=False)

    return results[["LOINC_NUM", "LONG_COMMON_NAME", "score"]]

def map_loinc(code_value, record_name):
    query = f"{code_value} {record_name}"
    results = search_loinc(query, top_k=1)

    best = results.iloc[0]

    return {
        "query": query,
        "loinc_num": best["LOINC_NUM"],
        "loinc_name": best["LONG_COMMON_NAME"],
        "score": best["score"]
    }
