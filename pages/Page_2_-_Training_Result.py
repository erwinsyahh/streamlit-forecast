import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all_tables, compute_row_metrics

st.title("Partition Validation Insights")

try:
    _, test_pred_df, _, _ = load_all_tables()
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

validation_df = compute_row_metrics(test_pred_df)
partitions = sorted(validation_df["GROUP_IDENTIFIER_STRING"].unique())
partition = st.selectbox("Choose Partition", partitions)
partition_df = validation_df[validation_df["GROUP_IDENTIFIER_STRING"] == partition]
partition_df["FUTURE_DTTM"] = pd.to_datetime(partition_df["FUTURE_DTTM"])

st.subheader("Line Chart")
st.line_chart(partition_df, x="FUTURE_DTTM", y=["ACTUAL", "PREDICTED"])

st.subheader("Predicted vs Actual")
fig = px.scatter(partition_df, x="ACTUAL", y="PREDICTED", trendline="ols")
fig.add_trace(go.Scatter(
    x=[partition_df["ACTUAL"].min(), partition_df["ACTUAL"].max()],
    y=[partition_df["ACTUAL"].min(), partition_df["ACTUAL"].max()],
    mode="lines", name="Ideal", line=dict(color="black", dash="dash")
))
st.plotly_chart(fig, use_container_width=True)