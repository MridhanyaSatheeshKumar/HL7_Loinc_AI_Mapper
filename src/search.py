import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/processed/loinc_cleaned.csv")

with open("data/processed/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/processed/matrix.pkl", "rb") as f:
    X = pickle.load(f)

def search_loinc(query, top_k=5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X)[0]

    top_idx = scores.argsort()[::-1][:top_k]

    results = df.iloc[top_idx].copy()
    results["score"] = scores[top_idx]

    return results[["LOINC_NUM", "LONG_COMMON_NAME", "score"]]
