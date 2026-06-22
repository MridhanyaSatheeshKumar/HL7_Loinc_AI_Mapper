import pandas as pd
from search import map_loinc
from explain import build_explanation

# Load input Excel
df_map = pd.read_excel(
    "data/input/LoincSubmission-PGHDterms_Updated_5-16.xlsx",
    sheet_name="PGHR Code Mapping Table"
)

df_req = pd.read_excel(
    "data/input/LoincSubmission-PGHDterms_Updated_5-16.xlsx",
    sheet_name="Requested LOINCs - Original"
)

sdk_df = pd.read_csv(
    "data/processed/sdk_metadata.csv"
)

df = df_map.merge(
    df_req,
    left_on="Code value",
    right_on="Local observation code",
    how="left"
)

print("Columns:", df.columns)

results = []

for _, row in df.iterrows():

    code = str(row.get("Code value", ""))

    sdk_row = sdk_df[
        sdk_df["apple_code_value"] == code
    ]

    # SDK metadata
    apple_desc = ""
    android_desc = ""
    sdk_unit = ""
    aggregation = ""
    category = ""

    if not sdk_row.empty:

        apple_desc = str(
            sdk_row.iloc[0]["apple_description"]
        )

        android_desc = str(
            sdk_row.iloc[0]["android_description"]
        )

        if android_desc == "nan":
            android_desc = ""

        sdk_unit = str(
            sdk_row.iloc[0]["unit"]
        )

        aggregation = str(
            sdk_row.iloc[0]["aggregation"]
        )

        category = str(
            sdk_row.iloc[0]["category"]
        )

    # Source records
    android_name = str(
        row.get("Health Connect (Android16)", "")
    )

    ios_name = str(
        row.get("HealthKit (iOS26)", "")
    )

    record_name = f"{android_name} {ios_name}"

    description = str(
        row.get("Observation description", "")
    )

    source = str(
        row.get("Reference Info/URL", "")
    )

    sdk_text = " ".join([
        apple_desc,
        android_desc,
        sdk_unit,
        aggregation,
        category
    ])

    try:

        top_matches = map_loinc(
            code,
            record_name,
            description,
            sdk_text,
            source,
            top_k=3
        )

        while len(top_matches) < 3:
            top_matches.append({
                "LOINC_NUM": "",
                "LONG_COMMON_NAME": "",
                "score": 0,
                "confidence": "Very Low",
                "status": "NO_MATCH",
                "domain": "UNKNOWN"
            })

        explanation_1 = build_explanation(
            code,
            record_name,
            top_matches[0].get("LOINC_NUM",""),
            top_matches[0].get("LONG_COMMON_NAME","")
        )

        explanation_2 = build_explanation(
            code,
            record_name,
            top_matches[1].get("LOINC_NUM",""),
            top_matches[1].get("LONG_COMMON_NAME","")
        )

        explanation_3 = build_explanation(
            code,
            record_name,
            top_matches[2].get("LOINC_NUM",""),
            top_matches[2].get("LONG_COMMON_NAME","")
        )

        results.append({

            "Code value": code,
            "Record": record_name,

            "LOINC_top1": top_matches[0].get("LOINC_NUM", "NONE"),
            "LOINC_name_1": top_matches[0].get("LONG_COMMON_NAME", ""),
            "Explanation_1": explanation_1,
            "score_1": top_matches[0].get("score", 0),
            "confidence_1": top_matches[0].get("confidence", "Very Low"),
            "status_1": top_matches[0].get("status", "UNKNOWN"),
            "domain_1": top_matches[0].get("domain", "UNKNOWN"),

            "LOINC_top2": top_matches[1].get("LOINC_NUM", "NONE"),
            "LOINC_name_2": top_matches[1].get("LONG_COMMON_NAME", ""),
            "Explanation_2": explanation_2,
            "score_2": top_matches[1].get("score", 0),
            "confidence_2": top_matches[1].get("confidence", "Very Low"),
            "status_2": top_matches[1].get("status", "UNKNOWN"),
            "domain_2": top_matches[1].get("domain", "UNKNOWN"),

            "LOINC_top3": top_matches[2].get("LOINC_NUM", "NONE"),
            "LOINC_name_3": top_matches[2].get("LONG_COMMON_NAME", ""),
            "Explanation_3": explanation_3,
            "score_3": top_matches[2].get("score", 0),
            "confidence_3": top_matches[2].get("confidence", "Very Low"),
            "status_3": top_matches[2].get("status", "UNKNOWN"),
            "domain_3": top_matches[2].get("domain", "UNKNOWN"),

            "primary_domain": top_matches[0].get(
                "domain",
                "UNKNOWN"
            )

        })

    except Exception as e:

        print(
            f"Error on {code} | {record_name}: {e}"
        )

# Save output
final_df = pd.DataFrame(results)

final_df.to_excel(
    "data/output/phr_mapped.xlsx",
    index=False
)

print(
    "Done! Output saved to data/output/phr_mapped.xlsx"
)
