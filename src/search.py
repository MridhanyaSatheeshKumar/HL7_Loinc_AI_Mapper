import pickle
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

print("Loading LOINC search resources...")

df = pd.read_csv("data/processed/loinc_cleaned.csv")

with open("data/processed/embeddings.pkl", "rb") as f:
    X = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Search system ready.")


def clean_input(text):

    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(text))

    return text.lower()


def expand_query(query):

    mapping = {

        "cycling": "bicycle riding exercise physical activity",

        "stairs": "stairs climbed stair climbing physical activity",

        "walking": "walking distance steps physical activity",

        "running": "running distance speed physical activity",

        "step": "steps walking activity count",

        "vo2": "oxygen consumption fitness exercise capacity",

        "calories": "energy expenditure calories burned",

        "distance": "length distance travel"

    }

    query = query.lower()

    for key in mapping:

        if key in query:

            query += " " + mapping[key]

    return query


def enrich_query(query):

    if "distance" in query:

        query += " length measurement"

    if "steps" in query:

        query += " step count walking"

    if "energy" in query or "calorie" in query:

        query += " energy expenditure calories"

    if "time" in query:

        query += " duration time measurement"

    return query


def is_valid_loinc(row):

    name = row["LONG_COMMON_NAME"].lower()

    bad_terms = [

        "promis",
        "phenx",
        "questionnaire",
        "survey",
        "assessment"

    ]

    for term in bad_terms:

        if term in name:

            return False

    return True


def boost_score(row, query):

    text = row["LONG_COMMON_NAME"].lower()

    q = query.lower()

    score = 0


    if "activity" in text:

        score += 0.25

    if "exercise" in text:

        score += 0.25

    if "physical" in text:

        score += 0.2


    if "distance" in q and "distance" in text:

        score += 0.4


    if "energy" in q and "energy" in text:

        score += 0.4


    if "step" in q and ("step" in text or "walk" in text):

        score += 0.4


    if "time" in q and "time" in text:

        score += 0.3


    return float(score)


def confidence_level(score):

    if score > 0.80:

        return "High"

    elif score > 0.65:

        return "Medium"

    elif score > 0.50:

        return "Low"

    else:

        return "Very Low"


def match_status(score):

    if score > 0.80:

        return "GOOD_MATCH"

    elif score > 0.65:

        return "REVIEW"

    elif score > 0.50:

        return "LOW_CONFIDENCE"

    else:

        return "NO_MATCH"


def search_loinc(query, top_k=10):

    query = clean_input(query)

    query = expand_query(query)

    query = enrich_query(query)

    query_vec = model.encode([query])

    scores = cosine_similarity(query_vec, X)[0]

    top_idx = scores.argsort()[::-1][:top_k]

    results = df.iloc[top_idx].copy()

    results["similarity"] = scores[top_idx]


    results = results[results.apply(is_valid_loinc, axis=1)]


    if results.empty:

        return pd.DataFrame([{

            "LOINC_NUM":"NONE",

            "LONG_COMMON_NAME":"No suitable LOINC found",

            "similarity":0.0,

            "boost":0.0,

            "final_score":0.0,

            "confidence":"Very Low",

            "status":"NO_MATCH"

        }])


    results["boost"] = results.apply(

        lambda row: boost_score(row, query),

        axis=1

    )


    results["final_score"] = results["similarity"] + results["boost"]


    results["confidence"] = results["final_score"].apply(

        confidence_level

    )


    results["status"] = results["final_score"].apply(

        match_status

    )


    results = results.sort_values(

        by="final_score",

        ascending=False

    )


    if results.iloc[0]["final_score"] < 0.50:

        return pd.DataFrame([{

            "LOINC_NUM":"NONE",

            "LONG_COMMON_NAME":"No suitable LOINC found",

            "similarity":results.iloc[0]["similarity"],

            "boost":results.iloc[0]["boost"],

            "final_score":results.iloc[0]["final_score"],

            "confidence":"Very Low",

            "status":"NO_MATCH"

        }])


    return results.head(3)[[

        "LOINC_NUM",

        "LONG_COMMON_NAME",

        "similarity",

        "boost",

        "final_score",

        "confidence",

        "status"

    ]]


def map_loinc(code_value, record_name, top_k=3):

    query = f"{code_value} {record_name}"

    results = search_loinc(query, top_k)

    candidates = []

    for _, row in results.iterrows():

        candidates.append({

            "LOINC_NUM": row.get("LOINC_NUM","NONE"),

            "LONG_COMMON_NAME": row.get("LONG_COMMON_NAME",""),

            "score": row.get("final_score",0),

            "confidence": row.get("confidence","Very Low"),

            "status": row.get("status","NO_MATCH")

        })

    return candidates
