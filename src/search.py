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


# ✅ Clean camelCase
def clean_input(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(text))
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
        "vo2 max": "oxygen consumption fitness exercise capacity"
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
    if "energy" in query or "calorie" in query:
        query += " energy expenditure calories"
    if "time" in query:
        query += " duration time"

    return query


# ✅ Filter bad LOINC rows (KEY IMPROVEMENT)
def is_valid_loinc(row):
    name = row["LONG_COMMON_NAME"].lower()

    # ❌ Remove surveys / questionnaires
    if any(x in name for x in ["promis", "phenx", "questionnaire", "survey"]):
        return False

    # ❌ Remove lab-specific irrelevant biology
    if any(x in name for x in ["enzyme", "serum", "plasma", "urine", "blood"]):
        return False

    return True


# ✅ Boost relevant concepts
def boost_score(row, query=""):
    text = row["LONG_COMMON_NAME"].lower()
    q = query.lower()

    score = 0

    # ✅ Boost activity-related terms
    if "activity" in text:
        score += 0.3
    if "physical" in text:
        score += 0.2
    if "exercise" in text:
        score += 0.3

    # ✅ Distance-related boost
    if "distance" in q and ("distance" in text or "walk" in text or "run" in text):
        score += 0.4

    # ✅ Energy/calorie boost
    if "energy" in q or "calorie" in q:
        if "energy" in text or "calorie" in text:
            score += 0.4

    # ✅ Steps / stairs boost
    if "step" in q or "stairs" in q or "flights" in q:
        if "step" in text or "stairs" in text or "climb" in text:
            score += 0.4

    # ✅ Time boost
    if "time" in q or "duration" in q:
        if "time" in text or "duration" in text:
            score += 0.3

    # ❌ Penalize survey/question-based LOINC
    if "promis" in text or "phenx" in text:
        score -= 0.5
    if "does your" in text or "limit you" in text:
        score -= 0.4

    # ❌ Penalize vague short labels
    if len(text.split()) < 3:
        score -= 0.2

    return score


# 🔍 Main search function
def search_loinc(query, top_k=10):
    query = clean_input(query)
    query = expand_query(query)
    query = enrich_query(query)

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X)[0]

    top_k = 3
    top_idx = scores.argsort()[::-1][:top_k]
    results = df.iloc[top_idx].copy()

    results["score"] = scores[top_idx]
    # add metadata extraction
    results["component"] = df.iloc[top_idx]["LONG_COMMON_NAME"]  # or component column if exists
    results["system"] = df.iloc[top_idx]["Code system"]          # if you have Code system column

    # 🔥 Filter bad rows BEFORE boosting
    results = results[results.apply(is_valid_loinc, axis=1)]

    # fallback if empty
    if results.empty:
        top_idx = scores.argsort()[::-1][:5]
        results = df.iloc[top_idx].copy()
        results["score"] = scores[top_idx]

    # Boost scores
    results["score"] = results.apply(
        lambda row: row["score"] + boost_score(row, query),
        axis=1
    )

    # Re-rank
    results = results.sort_values(by="score", ascending=False)

    return results[["LOINC_NUM", "LONG_COMMON_NAME", "score"]]

def structured_score(query, metadata):
    score = 0
    q = query.lower()
    component = str(metadata.get("component", "")).lower()
    system = str(metadata.get("system", "")).lower()

    if "distance" in q and "distance" in component:
        score += 2
    if "energy" in q and "energy" in component:
        score += 2
    if "step" in q and "step" in component:
        score += 2
    if "glucose" in q and "glucose" in component:
        score += 2
    if "blood" in q and "blood" in system:
        score += 1

    return score

# 🎯 Final mapping function
def map_loinc(code_value, record_name, top_k=3):
    query = f"{code_value} {record_name}"

    # Step 1: get initial candidates (semantic search)
    results = search_loinc(query, top_k=20)

    candidates = []

    # Step 2: apply structured scoring
    for _, row in results.iterrows():
        meta = {
            "component": row.get("LONG_COMMON_NAME", ""),
            "system": row.get("SYSTEM", ""),
            "class": row.get("CLASS", "")
        }

        score = structured_score(query, meta)

        candidates.append({
            "LOINC_NUM": row["LOINC_NUM"],
            "LONG_COMMON_NAME": row["LONG_COMMON_NAME"],
            "score": score
        })

    # Step 3: sort by structured score
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    # Step 4: return top K (instead of just 1)
    return candidates[:top_k]

    return (
        best["LOINC_NUM"],
        best["LONG_COMMON_NAME"],
        best["score"]
    )
