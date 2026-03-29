# HL7 LOINC AI Mapper

## Overview
This project automates mapping PGHD observations to LOINC codes using NLP similarity search.

## Problem
Manual LOINC mapping is slow and difficult. This tool suggests candidate mappings automatically.

## Approach
Pipeline:

1 Clean LOINC database
2 Create searchable text
3 TF-IDF embeddings
4 Cosine similarity search
5 Domain scoring
6 Excel output

## Project structure

data/
src/
output/

## How to run

Step 1:
python src/preprocess.py

Step 2:
python src/embed.py

Step 3:
python src/map_excel.py

## Output

phr_mapped.xlsx contains top 3 LOINC suggestions.

## Future improvements

Sentence transformers
LOINC hierarchy
FHIR terminology server
