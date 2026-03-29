import pandas as pd

df = pd.read_excel("data/output/phr_mapped.xlsx")

total = len(df)

mapped = len(df[df["LOINC_top1"] != "NONE"])

no_match = len(df[df["status_1"] == "NO_MATCH"])

domain_skip = len(df[df["status_1"] == "DOMAIN_SKIP"])

good_match = len(df[df["status_1"] == "GOOD_MATCH"])

review = len(df[df["status_1"] == "REVIEW"])


print("\n===== LOINC Mapping Evaluation =====\n")

print("Total PGHD terms:", total)

print("\nMapping results:")

print("Mapped:", mapped)

print("No match:", no_match)

print("Domain skipped:", domain_skip)

print("\nQuality:")

print("Good matches:", good_match)

print("Needs review:", review)


coverage = mapped / total * 100

print("\nCoverage %:", round(coverage,2))


print("\n===== Domain Distribution =====\n")

print(df["primary_domain"].value_counts())

print("\n===== Coverage by Domain =====\n")

for domain in df["primary_domain"].unique():

    domain_df = df[df["primary_domain"] == domain]

    mapped_domain = len(domain_df[domain_df["LOINC_top1"] != "NONE"])

    total_domain = len(domain_df)

    coverage = mapped_domain / total_domain * 100

    print(domain, ":", round(coverage,1), "%")
