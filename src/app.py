import streamlit as st
from search import search_loinc

st.title("LOINC Mapping Assistant")

query = st.text_input("Enter activity")

if query:
    results = search_loinc(query)
    st.write(results)
