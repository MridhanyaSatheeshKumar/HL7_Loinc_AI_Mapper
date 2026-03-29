import pandas as pd
from search import map_loinc

# 📥 Load input Excel
df = pd.read_excel(
    "data/input/LoincSubmission_sheet.xlsx",
    sheet_name="PGHR Code Mapping Table"
)

print("Columns:", df.columns)

results = []

for _, row in df.iterrows():
    code = str(row.get("Code value", ""))

    # Prefer Android, fallback to iOS
    record_name = row.get("Health Connect (Android16)")
    if pd.isna(record_name) or record_name == "":
        record_name = row.get("HealthKit (iOS26)")

    if pd.isna(record_name):
        record_name = ""

    record_name = str(record_name)

    try:
        # 🔍 Get Top 3 matches
        top_matches = map_loinc(code, record_name, top_k=3)

        # 🛡️ Safety: ensure always 3 results
        while len(top_matches) < 3:
            top_matches.append({
                "LOINC_NUM": "",
                "LONG_COMMON_NAME": "",
                "score": 0
            })

        # 📦 Store results
        results.append({

            "Code value": code,
            "Record": record_name,


            "LOINC_top1": top_matches[0].get("LOINC_NUM","NONE"),
            "LOINC_name_1": top_matches[0].get("LONG_COMMON_NAME",""),
            "score_1": top_matches[0].get("score",0),
            "confidence_1": top_matches[0].get("confidence","Very Low"),
            "status_1": top_matches[0].get("status","UNKNOWN"),
            "domain_1": top_matches[0].get("domain","UNKNOWN"),


            "LOINC_top2": top_matches[1].get("LOINC_NUM","NONE"),
            "LOINC_name_2": top_matches[1].get("LONG_COMMON_NAME",""),
            "score_2": top_matches[1].get("score",0),
            "confidence_2": top_matches[1].get("confidence","Very Low"),
            "status_2": top_matches[1].get("status","UNKNOWN"),
            "domain_2": top_matches[1].get("domain","UNKNOWN"),


            "LOINC_top3": top_matches[2].get("LOINC_NUM","NONE"),
            "LOINC_name_3": top_matches[2].get("LONG_COMMON_NAME",""),
            "score_3": top_matches[2].get("score",0),
            "confidence_3": top_matches[2].get("confidence","Very Low"),
            "status_3": top_matches[2].get("status","UNKNOWN"),
            "domain_3": top_matches[2].get("domain","UNKNOWN"),


            "primary_domain": top_matches[0].get("domain","UNKNOWN")

        })


    except Exception as e:
        print(f"❌ Error on {code} | {record_name}: {e}")

# 💾 Save output
final_df = pd.DataFrame(results)
final_df.to_excel("data/output/phr_mapped.xlsx", index=False)

print("Done! Output saved to data/output/phr_mapped.xlsx")
