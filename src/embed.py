import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

print("Loading cleaned LOINC data...")

# Load cleaned data
df = pd.read_csv("data/processed/loinc_cleaned.csv")

print("Loading sentence transformer model...")

# Fast and accurate small model
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Preparing text for embedding...")

# Build better semantic text (natural sentence style works better)
df["search_text"] = (
    df["LONG_COMMON_NAME"].fillna("") +
    " component " + df["COMPONENT"].fillna("") +
    " system " + df["SYSTEM"].fillna("") +
    " method " + df["METHOD_TYP"].fillna("")
)

print("Creating embeddings...")

# Create embeddings
embeddings = model.encode(
    df["search_text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Saving embeddings...")

# Save embeddings
with open("data/processed/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Saving model reference...")

# Optional: save model (not strictly required)
with open("data/processed/embedding_model.pkl", "wb") as f:
    pickle.dump("all-MiniLM-L6-v2", f)

print("Sentence transformer embeddings saved successfully!")
print(f"Total LOINC entries embedded: {len(embeddings)}")
