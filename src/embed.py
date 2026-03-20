from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

df = pd.read_csv("data/processed/loinc_cleaned.csv")

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["text"])

# Save
with open("data/processed/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("data/processed/matrix.pkl", "wb") as f:
    pickle.dump(X, f)

print("TF-IDF embeddings saved!")
