import pandas as pd

# ==============================
# LOAD DATA
# ==============================

pred_df = pd.read_excel("data/output/phr_mapped.xlsx")

true_df = pd.read_excel(
    "data/input/LoincSubmission-PGHDterms_Updated_5-16.xlsx",
    sheet_name="PGHR Code Mapping Table"
)

print("Pred columns:", pred_df.columns.tolist())
print("True columns:", true_df.columns.tolist())

# ==============================
# MERGE
# ==============================

df = pd.merge(
    pred_df,
    true_df[["Code value", "LOINC"]],
    on="Code value",
    how="left"
)

print("\nMerged columns:", df.columns.tolist())


# ==============================
# CLEAN LOINC FUNCTION
# ==============================

def clean_codes(code_str):
    """
    Convert messy LOINC field into list of valid codes
    """
    if pd.isna(code_str):
        return []

    code_str = str(code_str)

    # FIX datetime issue like: 8302-02-01 00:00:00
    code_str = code_str.split(" ")[0]

    code_str = code_str.replace("[", "").replace("]", "").strip()

    parts = [c.strip() for c in code_str.replace(";", ",").split(",")]

    # keep only valid-looking codes (contains '-')
    parts = [c for c in parts if "-" in c]

    return parts


# ==============================
# MATCHING LOGIC
# ==============================

def exact_match(row):
    true_codes = clean_codes(row["LOINC"])
    pred_code = str(row["LOINC_top1"]).strip()

    return pred_code in true_codes

def top3_match(row):

    true_codes = clean_codes(row["LOINC"])

    preds = [
        str(row.get("LOINC_top1", "")).strip(),
        str(row.get("LOINC_top2", "")).strip(),
        str(row.get("LOINC_top3", "")).strip()
    ]

    for pred in preds:
        if pred in true_codes:
            return True

    return False

def partial_match(row):
    true_codes = clean_codes(row["LOINC"])
    pred_prefix = str(row["LOINC_top1"]).split("-")[0]

    for code in true_codes:
        if code.startswith(pred_prefix):
            return True
    return False


# APPLY FIRST
df["exact_match"] = df.apply(exact_match, axis=1)
df["top3_match"] = df.apply(top3_match, axis=1)
df["partial_match"] = df.apply(partial_match, axis=1)


# ==============================
# TYPE CHECK
# ==============================

def type_check(row):
    name = str(row["LOINC_name_1"]).lower()

    if any(x in name for x in ["survey", "question", "promis", "panel"]):
        return "survey"
    elif any(x in name for x in ["enzyme", "serum", "blood", "plasma"]):
        return "lab"
    elif any(x in name for x in ["activity", "distance", "energy", "steps", "cadence", "mass", "height"]):
        return "measurement"
    else:
        return "clear"


df["type"] = df.apply(type_check, axis=1)


# ==============================
# FINAL LABEL
# ==============================

def final_label(row):
    if row["exact_match"]:
        return "correct"
    elif row["partial_match"]:
        return "close"
    else:
        return "wrong"


df["final_result"] = df.apply(final_label, axis=1)


# ==============================
# FILTER VALID ROWS
# ==============================

valid_df = df[df["LOINC"].notna() & (df["LOINC"] != "-")]

print("\nValid rows:", len(valid_df))


# ==============================
# METRICS
# ==============================

total = len(valid_df)
exact = valid_df["exact_match"].sum()
partial = valid_df["partial_match"].sum()
top3 = valid_df["top3_match"].sum()

print("Top-3:", top3, f"({top3/total:.2%})")

print("\n RESULTS (VALID ROWS ONLY):")
print("Total valid:", total)

if total > 0:
    print("Exact:", exact, f"({exact/total:.2%})")
    print("Partial:", partial, f"({partial/total:.2%})")
else:
    print("No valid rows found!")


# ==============================
# SAMPLE ROWS
# ==============================

print("\n SAMPLE ROWS:")
print(valid_df[["Code value", "LOINC", "LOINC_top1"]].head(10))

# ==============================
# BREAKDOWNS
# ==============================

print("\nFinal result breakdown:")
print(valid_df["final_result"].value_counts())

print("\nType breakdown:")
print(df["type"].value_counts())


# ==============================
# SAVE
# ==============================

df.to_excel("data/output/phr_evaluated.xlsx", index=False)
print("\nSaved evaluated file!")
