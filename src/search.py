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

def classify_domain(text):

    t = text.lower()

    # Measurement domains (LOINC likely exists)
    measurement_keywords = [

        "rate","pressure","temperature","distance","energy",
        "oxygen","vo2","heart","pulse","cadence","speed",
        "power","glucose","weight","height","bmi",
        "respiratory","flow","volume","sleep","snore",
        "blood","body","water","nutrition","vitamin"
    ]

    # Activity domains (sometimes LOINC exists)
    activity_keywords = [

        "running","walking","swimming","exercise",
        "fitness","training","sports","workout",
        "cycling","rowing","yoga"
    ]

    # Symptoms (sometimes LOINC)
    symptom_keywords = [

        "pain","fever","fatigue","nausea","cough",
        "dizziness","vomiting","headache"
    ]

    # Emotions (LOINC usually no)
    emotion_keywords = [

        "happy","sad","angry","anxious","stressed",
        "worried","joyful","frustrated","content"
    ]

    # Lifestyle categories (no LOINC)
    lifestyle_keywords = [

        "family","work","community","travel",
        "hobbies","money","dating","weather"
    ]


    if any(x in t for x in measurement_keywords):

        return "MEASUREMENT"

    if any(x in t for x in activity_keywords):

        return "ACTIVITY"

    if any(x in t for x in symptom_keywords):

        return "SYMPTOM"

    if any(x in t for x in emotion_keywords):

        return "EMOTION"

    if any(x in t for x in lifestyle_keywords):

        return "LIFESTYLE"

    return "UNKNOWN"

def expand_query(query):

    mapping = {

        "cycling": "bicycle riding exercise physical activity",

        "stairs": "stairs climbed stair climbing physical activity",

        "walking": "walking distance steps physical activity",

        "running": "running distance speed physical activity",

        "step": "steps walking activity count",

        "vo2": "oxygen consumption fitness exercise capacity",

        "calories": "energy expenditure calories burned",

        "distance": "length distance travel",

        "heart": "heart rate cardiac pulse cardiovascular",

        "sleep": "sleep stage rem deep light",

        "mood": "emotion mental stress anxiety",

        "temperature": "body temperature measurement",

        "respiratory": "breathing respiration oxygen"
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


    # Activity domain
    if any(x in q for x in ["activity","exercise","fitness"]):

        if any(x in text for x in ["activity","exercise","physical"]):

            score += 0.35


    # Energy domain
    if any(x in q for x in ["energy","calorie","burn"]):

        if any(x in text for x in ["energy","calorie","metabolic"]):

            score += 0.50


    # Distance domain
    if "distance" in q:

        if any(x in text for x in ["distance","walk","run","travel"]):

            score += 0.45


    # Step domain
    if any(x in q for x in ["step","stairs","flights"]):

        if any(x in text for x in ["step","stairs","climb","walk"]):

            score += 0.45


    # Heart domain
    if any(x in q for x in ["heart","cardio","pulse","hr"]):

        if any(x in text for x in ["heart","cardiac","pulse"]):

            score += 0.50


    # Oxygen fitness domain
    if any(x in q for x in ["vo2","oxygen","fitness"]):

        if any(x in text for x in ["oxygen","vo2","fitness"]):

            score += 0.55


    # Speed / cadence domain
    if any(x in q for x in ["speed","cadence","pace","power"]):

        if any(x in text for x in ["speed","cadence","power"]):

            score += 0.45


    # Time domain
    if any(x in q for x in ["time","duration","minutes"]):

        if any(x in text for x in ["time","duration"]):

            score += 0.35


    # Sleep domain
    if "sleep" in q:

        if "sleep" in text:

            score += 0.50


    # Mood / mental domain (important for your dataset)
    if any(x in q for x in ["mood","emotion","stress","anxiety"]):

        if any(x in text for x in ["mood","emotion","mental"]):

            score += 0.40


    # Temperature domain
    if "temperature" in q:

        if "temperature" in text:

            score += 0.50


    # Penalize surveys
    if any(x in text for x in ["promis","phenx","survey"]):

        score -= 0.5


    return float(score)

def confidence_level(score):

    if score > 0.85:

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

    domain = classify_domain(query)


    # Skip domains that shouldn't map to LOINC
    if domain in ["EMOTION","LIFESTYLE"]:

        return [{

            "LOINC_NUM":"NONE",

            "LONG_COMMON_NAME":"Not appropriate for LOINC",

            "score":0,

            "confidence":"Very Low",

            "status":"DOMAIN_SKIP",

            "domain":domain

        }]


    results = search_loinc(query, top_k)

    candidates = []


    for _, row in results.iterrows():

        candidates.append({

            "LOINC_NUM": row.get("LOINC_NUM","NONE"),

            "LONG_COMMON_NAME": row.get("LONG_COMMON_NAME",""),

            "score": row.get("final_score",0),

            "confidence": row.get("confidence","Very Low"),

            "status": row.get("status","NO_MATCH"),

            "domain":domain

        })


    return candidates
