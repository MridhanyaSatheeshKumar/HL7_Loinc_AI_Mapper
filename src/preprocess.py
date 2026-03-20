import pandas as pd

df = pd.read_csv("data/raw/loinc.csv")

df = df[[
    "LOINC_NUM",
    "LONG_COMMON_NAME",
    "COMPONENT",
    "SYSTEM",
    "METHOD_TYP"
]]

df = df.dropna(subset=["LONG_COMMON_NAME"])

df["text"] = (
    df["LONG_COMMON_NAME"] + " | " +
    df["COMPONENT"].fillna("") + " | " +
    df["SYSTEM"].fillna("") + " | " +
    df["METHOD_TYP"].fillna("")
)

df = df.head(1000)

df.to_csv("data/processed/loinc_cleaned.csv", index=False)

print("Cleaned data saved!")
