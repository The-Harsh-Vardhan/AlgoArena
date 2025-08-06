import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="ML Model Comparison – Tabular Data", layout="wide")

st.title("📊 Tabular ML Model Comparison")
st.subheader("See how different models performed on the Adult dataset")

# Load results
results_path = "../01_Tabular_Data/tabular_results.json"
if not os.path.exists(results_path):
    st.error("Results file not found! Please run the Jupyter Notebook first.")
    st.stop()

with open(results_path, "r") as f:
    results = json.load(f)

df = pd.DataFrame(results).T.sort_values(by="accuracy", ascending=False)

# Clean column names
df.columns = [col.capitalize() for col in df.columns]

# Display table
st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"))

# Bar chart: Accuracy
st.subheader("📈 Accuracy Comparison")
st.bar_chart(df["Accuracy"])

# Add more metrics
with st.expander("📌 View All Metrics", expanded=False):
    st.line_chart(df)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Dataset: Adult Income Dataset")