import pandas as pd

print("Loading LOINC database...")

# Load data
df = pd.read_csv("data/raw/loinc.csv", low_memory=False)

print("Selecting relevant columns...")

df = df[[
    "LOINC_NUM",
    "LONG_COMMON_NAME",
    "COMPONENT",
    "SYSTEM",
    "METHOD_TYP",
    "CLASS"
]]

# Remove rows without names
df = df.dropna(subset=["LONG_COMMON_NAME"])


print("Filtering PGHD related concepts...")

# ✅ Keep PGHD domains (expanded properly)
df = df[df["LONG_COMMON_NAME"].str.contains(

    "activity|exercise|physical|distance|steps|stairs|walking|running|"
    "energy|calorie|metabolic|fitness|vo2|oxygen|"
    "heart|cardiac|pulse|blood pressure|"
    "sleep|rest|"
    "speed|cadence|power|pace|"
    "temperature|respiratory|breathing|"
    "effort|endurance",

    case=False,
    na=False
)]


print("Removing noisy clinical/lab concepts...")

# ❌ Remove lab tests
df = df[~df["LONG_COMMON_NAME"].str.contains(

    "serum|plasma|urine|blood test|antibody|enzyme|culture|"
    "specimen|lab|panel|gene|dna|rna|tumor|cancer|biopsy",

    case=False,
    na=False
)]

# ❌ Remove surveys/questionnaires
df = df[~df["LONG_COMMON_NAME"].str.contains(

    "promis|phenx|questionnaire|survey|assessment",

    case=False,
    na=False
)]

# ❌ Remove administrative notes
df = df[~df["LONG_COMMON_NAME"].str.contains(

    "note|narrative|report|finding|interpretation",

    case=False,
    na=False
)]


print("Building search text...")

# Combine into natural sentence (better embeddings)
df["text"] = (

    df["LONG_COMMON_NAME"].fillna("") +

    " component " + df["COMPONENT"].fillna("") +

    " system " + df["SYSTEM"].fillna("") +

    " method " + df["METHOD_TYP"].fillna("")

)


print("Removing duplicates...")

df = df.drop_duplicates(subset=["LOINC_NUM"])


print("Dataset size after filtering:", len(df))


print("Saving cleaned dataset...")

df.to_csv(

    "data/processed/loinc_cleaned.csv",

    index=False

)

print("Preprocessing complete!")
