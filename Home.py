import streamlit as st
import pandas as pd
import plotly.express as px
from utils.load_data import load_all_tables

st.set_page_config(page_title="Forecast Dashboard", layout="centered")

# --- LOAD DATA ---
try:
    train_df, test_pred_df, result_df, model_df = load_all_tables()
except Exception as e:
    st.error(f"❌ Data loading failed: {e}")
    st.stop()

# --- MAIN PAGE ---
st.title("📊 Forecast Dashboard")

st.markdown("""
This dashboard helps explore forecast accuracy and metrics across different product and store partitions.
Use the sidebar to select specific groups or customize your analysis.
""")

# --- SAMPLE METRICS OVERVIEW ---
st.subheader("Sample of Loaded Data")
st.write("### Train Data")
st.dataframe(train_df.head(10))

st.write("### Forecast Results")
st.dataframe(result_df.head(10))
