import streamlit as st
import pandas as pd
import plotly.express as px
from utils.load_data import load_all_tables, preprocess_model_data, calculate_average_rank

st.set_page_config(page_title="Feature Importance Explorer")

# --- LOAD DATA ---
try:
    _, _, _, model_df = load_all_tables()
except Exception as e:
    st.error(f"❌ Data loading failed: {e}")
    st.stop()

# --- PREPROCESSING ---
model_df, feature_df = preprocess_model_data(model_df)

# --- UI ---
st.title("🔍 Feature Importance Visualization")

col1, col2 = st.columns([2, 1])
with col1:
    partitions = sorted(model_df["GROUP_IDENTIFIER_STRING"].unique())
    selected_partition = st.selectbox("Select Partition (optional)", [None] + partitions)
with col2:
    top_n = st.slider("Top N Features", min_value=5, max_value=50, value=10)

# --- PLOTTING ---
def plot_feature_importance(df, is_aggregated=True, top_n=20):
    if is_aggregated:
        df = df.sort_values("AVERAGE_RANK", ascending=True).head(top_n)
        fig = px.bar(
            df,
            x="AVERAGE_RANK",
            y="FEATURE",
            orientation="h",
            title="Top Feature Importance (Aggregated)",
            labels={"FEATURE": "Feature", "AVERAGE_RANK": "Average Rank"},
        )
        fig.update_layout(yaxis=dict(categoryorder="total descending"))
    else:
        df = df.sort_values("IMPORTANCE", ascending=False).head(top_n)
        fig = px.bar(
            df,
            x="IMPORTANCE",
            y="FEATURE",
            orientation="h",
            title="Top Feature Importance (Single Partition)",
            labels={"FEATURE": "Feature", "IMPORTANCE": "Importance"},
        )
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))

    fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))
    return fig

if selected_partition:
    filtered_df = feature_df[feature_df["GROUP_IDENTIFIER_STRING"] == selected_partition]
    fig = plot_feature_importance(filtered_df, is_aggregated=False, top_n=top_n)
else:
    feature_df, avg_rank_df = calculate_average_rank(feature_df)
    fig = plot_feature_importance(avg_rank_df, is_aggregated=True, top_n=top_n)

st.plotly_chart(fig, use_container_width=True)

# --- TABLES ---
with st.expander("Show Underlying Data"):
    if selected_partition:
        st.dataframe(filtered_df.sort_values("IMPORTANCE", ascending=False))
    else:
        tabs = st.tabs(["Average Importance", "All Features"])
        tabs[0].dataframe(avg_rank_df.sort_values("AVERAGE_RANK", ascending=True))
        tabs[1].dataframe(feature_df.sort_values("IMPORTANCE", ascending=False))
