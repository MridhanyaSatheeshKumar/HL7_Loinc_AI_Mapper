import pandas as pd

# Load data
df = pd.read_csv("data/raw/loinc.csv", low_memory=False)

# Keep relevant columns
df = df[[
    "LOINC_NUM",
    "LONG_COMMON_NAME",
    "COMPONENT",
    "SYSTEM",
    "METHOD_TYP"
]]

# Drop missing
df = df.dropna(subset=["LONG_COMMON_NAME"])

# ✅ Keep only activity/exercise-related
df = df[df["LONG_COMMON_NAME"].str.contains(
    "activity|exercise|physical|distance|steps|stairs|oxygen",
    case=False, na=False
)]

# ❌ Remove clinical measurements/tests/noise
df = df[~df["LONG_COMMON_NAME"].str.contains(
    "post|after|panel|rate|fev|time|volume|pressure|test|measure|level",
    case=False, na=False
)]

# ❌ Remove generic clinical notes
df = df[~df["LONG_COMMON_NAME"].str.contains(
    "finding|note|narrative|patient",
    case=False, na=False
)]

# Combine into one searchable text
df["text"] = (
    df["LONG_COMMON_NAME"] + " | " +
    df["COMPONENT"].fillna("") + " | " +
    df["SYSTEM"].fillna("") + " | " +
    df["METHOD_TYP"].fillna("")
)

# Keep small subset for speed (you can increase later)
df = df.head(1000)

# Save
df.to_csv("data/processed/loinc_cleaned.csv", index=False)

print("Cleaned + filtered data saved!")
