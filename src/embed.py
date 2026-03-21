import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load cleaned data
df = pd.read_csv("data/processed/loinc_cleaned.csv")

# Create TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["text"])

# Save vectorizer + matrix
with open("data/processed/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("data/processed/matrix.pkl", "wb") as f:
    pickle.dump(X, f)

print("TF-IDF embeddings saved!")
