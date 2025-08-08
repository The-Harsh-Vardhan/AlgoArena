import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ML Model Comparison – Tabular Data", layout="wide")

st.title("📊 Tabular ML Model Comparison")
st.subheader("See how different models performed on the Adult dataset")

# Load results
results_path = "../01_Tabular_Data/tabular/tabular_results.json"
if not os.path.exists(results_path):
    st.error("Results file not found! Please run the Jupyter Notebook first.")
    st.info("Expected file location: " + os.path.abspath(results_path))
    st.stop()

try:
    with open(results_path, "r") as f:
        results = json.load(f)
except Exception as e:
    st.error(f"Error loading results: {e}")
    st.stop()

df = pd.DataFrame(results).T.sort_values(by="accuracy", ascending=False)

# Clean column names
df.columns = [col.capitalize() for col in df.columns]

# Display metrics overview
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Best Model", df.index[0])
with col2:
    st.metric("Best Accuracy", f"{df['Accuracy'].max():.3f}")
with col3:
    st.metric("Best F1-Score", f"{df['F1_score'].max():.3f}")
with col4:
    st.metric("Models Compared", len(df))

# Display table
st.subheader("📊 Model Performance Table")
st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

# Create interactive plots
col1, col2 = st.columns(2)

with col1:
    # Bar chart: Accuracy
    st.subheader("📈 Accuracy Comparison")
    fig_acc = px.bar(
        x=df.index, 
        y=df["Accuracy"], 
        title="Model Accuracy Comparison",
        labels={"x": "Models", "y": "Accuracy"}
    )
    fig_acc.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_acc, use_container_width=True)

with col2:
    # Bar chart: F1-Score
    st.subheader("🎯 F1-Score Comparison")
    fig_f1 = px.bar(
        x=df.index, 
        y=df["F1_score"], 
        title="Model F1-Score Comparison",
        labels={"x": "Models", "y": "F1-Score"},
        color=df["F1_score"],
        color_continuous_scale="viridis"
    )
    fig_f1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_f1, use_container_width=True)

# Add more metrics
with st.expander("📌 View All Metrics", expanded=False):
    # Radar chart for comprehensive comparison
    st.subheader("🕸️ Comprehensive Model Comparison")
    
    # Select top 3 models for radar chart
    top_models = df.head(3)
    
    fig_radar = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_score']
    
    for model in top_models.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=[top_models.loc[model, metric] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=model
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Top 3 Models - All Metrics Comparison"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Line chart for all metrics
    st.subheader("📊 All Metrics Line Chart")
    st.line_chart(df[metrics])

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Dataset: Adult Income Dataset")