import streamlit as st
from utils.load_data import load_all_tables, compute_row_metrics, compute_partition_metrics, compute_overall_metrics
import plotly.express as px

st.title("Forecast Model Summary")
try:
    _, test_pred_df, _, _ = load_all_tables()
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

val_df = compute_row_metrics(test_pred_df)
partition_metrics = compute_partition_metrics(val_df)
overall = compute_overall_metrics(partition_metrics)

# Metric Distribution plot with dynamic filtering
metric = st.selectbox("Metric", ["MAPE", "MAE", "RMSE"])
st.subheader(f"{metric} Distribution")
distribution_df = partition_metrics.copy()

# Add a slider to filter outliers
value_min, value_max = st.slider(
    f"Filter {metric} range in plot:",
    float(distribution_df[metric].min()),
    float(distribution_df[metric].max()),
    (float(distribution_df[metric].min()), float(distribution_df[metric].max())),
)

# Filter the DataFrame based on the slider values
filtered_df = distribution_df[
    (distribution_df[metric] >= value_min) & (distribution_df[metric] <= value_max)
]

fig = px.box(
    filtered_df,
    x=metric,  # Horizontal orientation
    points="all",  # Show individual data points as dots
    title=f"{metric} Distribution ({value_min:.2f} - {value_max:.2f})",
    labels={metric: metric, "GROUP_IDENTIFIER_STRING": "Partition"},
    hover_data=["GROUP_IDENTIFIER_STRING"],  # Add this for hover info
)

fig.update_layout(template="plotly_white", showlegend=True)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Overall Aggregated Metrics")
st.dataframe(overall)

metric_choice = st.selectbox("Sort by", ["MAPE", "MAE", "RMSE"])
col1, col2 = st.columns(2)

with col1:
    st.subheader("Best Partitions")
    st.dataframe(partition_metrics.sort_values(by=metric_choice, ascending=True).head())
with col2:
    st.subheader("Worst Partitions")
    st.dataframe(partition_metrics.sort_values(by=metric_choice, ascending=False).head())
    
