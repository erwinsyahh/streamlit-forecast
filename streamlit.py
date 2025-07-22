import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from snowflake.snowpark import Session
import plotly.express as px
import plotly.graph_objects as go
import json

# --- 1. Establish Snowflake Session ---
st.title("Sales Volume Forecast Evaluation Dashboard")

try:
    session = Session.builder.config("connection_name", "poc_nojo").create()
    st.success("✅ Successfully connected to Snowflake")
except Exception as e:
    st.error(f"❌ Failed to connect to Snowflake: {e}")
    st.stop()

# --- 2. Load Data ---
with st.spinner("Loading tables from Snowflake..."):
    train_df = session.table("POC_NOJORONO.ML_FORECAST.DAILY_PARTITIONED_SAMPLE_DATA").to_pandas()
    test_pred_df = session.table("POC_NOJORONO.NOTEBOOKS.VALIDATION_PREDS_FROM_TEST_MODEL_1").to_pandas()
    result_df = session.table("POC_NOJORONO.NOTEBOOKS.FORECAST_RESULTS").to_pandas()
    model_bin_df = session.table("POC_NOJORONO.NOTEBOOKS.MODEL_STORAGE_TEST_MODEL_1").to_pandas()
    final_sdf_df = session.table("POC_NOJORONO.NOTEBOOKS.FEATURE_TABLES").to_pandas()
    model_df = session.table("POC_NOJORONO.NOTEBOOKS.MODEL_STORAGE_TEST_MODEL_1").to_pandas()

st.success("✅ Tables loaded successfully")

# Close session after loading data
session.close()

# --- 3. Preview Training Data ---
st.header("📊 Data Preview")
st.subheader("Training Data (Top 5 Rows)")
st.dataframe(train_df.head(), use_container_width=True)

# --- 4. Row-Level Error Metrics ---
st.header("📈 Row-Level Forecast Metrics")
row_actual_v_fcst = test_pred_df.copy()

row_actual_v_fcst["ABS_ERROR"] = (row_actual_v_fcst["ACTUAL"] - row_actual_v_fcst["PREDICTED"]).abs()
row_actual_v_fcst["APE"] = np.where(
    row_actual_v_fcst["ACTUAL"] == 0,
    np.nan,
    row_actual_v_fcst["ABS_ERROR"] / row_actual_v_fcst["ACTUAL"]
)
row_actual_v_fcst["SQ_ERROR"] = (row_actual_v_fcst["ACTUAL"] - row_actual_v_fcst["PREDICTED"]) ** 2

# --- 5. Partition-Level Metrics ---
st.header("📂 Metrics Per Group Partition")

partition_metrics = row_actual_v_fcst.groupby("GROUP_IDENTIFIER_STRING").agg(
    MAPE=("APE", "mean"),
    MAE=("ABS_ERROR", "mean"),
    RMSE=("SQ_ERROR", lambda x: np.sqrt(np.mean(x)))
).reset_index()

# Optional rounding
partition_metrics[["MAPE", "MAE", "RMSE"]] = partition_metrics[["MAPE", "MAE", "RMSE"]].round(3)

st.dataframe(partition_metrics, use_container_width=True)

# --- 6. Overall Metrics (Avg + Median) ---
st.header("📌 Overall Model Performance Summary")

overall_avg_metrics = pd.DataFrame([{
    "AGGREGATION": "AVG",
    "OVERALL_MAPE": partition_metrics["MAPE"].mean(),
    "OVERALL_MAE": partition_metrics["MAE"].mean(),
    "OVERALL_RMSE": partition_metrics["RMSE"].mean()
}])

overall_median_metrics = pd.DataFrame([{
    "AGGREGATION": "MEDIAN",
    "OVERALL_MAPE": partition_metrics["MAPE"].median(),
    "OVERALL_MAE": partition_metrics["MAE"].median(),
    "OVERALL_RMSE": partition_metrics["RMSE"].median()
}])

overall_metrics = pd.concat([overall_avg_metrics, overall_median_metrics], ignore_index=True)
overall_metrics[["OVERALL_MAPE", "OVERALL_MAE", "OVERALL_RMSE"]] = overall_metrics[
    ["OVERALL_MAPE", "OVERALL_MAE", "OVERALL_RMSE"]
].round(3)

st.subheader("Average and Median Across All Partitions")
st.dataframe(overall_metrics, use_container_width=True)

# --- 7. Partition Metrics Visualizations ---
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

# Layout with two columns
col1, col2 = st.columns(2)

# Column 1: Tables
with col1:
    # Look at the best performing partitions
    st.subheader("✅ BEST Performing Partitions")
    st.dataframe(partition_metrics.sort_values(by=metric).head(), use_container_width=True)
with col2:
    # Look at the worst performing partitions
    st.subheader("❌ WORST Performing Partitions")
    st.dataframe(partition_metrics.sort_values(by=metric, ascending=False).head(), use_container_width=True)

# --- 8. Partition Metrics Visualizations ---
st.header("📊 Validation Set Visualizations")
# Create list of unique partitions
partitions = sorted(test_pred_df["GROUP_IDENTIFIER_STRING"].unique())

# Select a single partition to visualize
partition_choice = st.selectbox("Partition", partitions)

# Filter validation set for selected partition
partition_choice_df = (
    test_pred_df[test_pred_df["GROUP_IDENTIFIER_STRING"] == partition_choice]
    .sort_values("FUTURE_DTTM")
    .copy()
)
partition_choice_df["FUTURE_DTTM"] = pd.to_datetime(partition_choice_df["FUTURE_DTTM"])

# Tabs layout
tabs = st.tabs(
    [
        "Line Plot: Validation Actual & Predicted",
        "Scatter Plot: Validation Actual vs. Predicted",
        "Line Plot: Training Actuals",
    ]
)

# --- Tab 1: Line Plot for Validation ---
tabs[0].line_chart(partition_choice_df, x="FUTURE_DTTM", y=["ACTUAL", "PREDICTED"])

# --- Tab 2: Scatter Plot: Predicted vs Actual ---
fig_scatter = px.scatter(
    partition_choice_df,
    x="ACTUAL",
    y="PREDICTED",
    title="Predicted vs. Actual Visits",
    labels={"ACTUAL": "Actual Visits", "PREDICTED": "Predicted Visits"},
    opacity=0.6,
    trendline="ols",
    hover_data=["PREDICTED", "ACTUAL", "FUTURE_DTTM"],
)

# Add expected y = x trendline
min_visits = min(partition_choice_df["ACTUAL"].min(), partition_choice_df["PREDICTED"].min())
max_visits = max(partition_choice_df["ACTUAL"].max(), partition_choice_df["PREDICTED"].max())

fig_scatter.add_trace(
    go.Scatter(
        x=[min_visits, max_visits],
        y=[min_visits, max_visits],
        mode="lines",
        line=dict(color="black", dash="dash"),
        name="Expected Trend (y = x)",
        showlegend=True,
    )
)
tabs[1].plotly_chart(fig_scatter, use_container_width=True)

# --- Tab 3: Training Actuals ---
train_partition_df = (
    final_sdf_df[final_sdf_df["GROUP_IDENTIFIER_STRING"] == partition_choice]
    .sort_values("ORDER_TIMESTAMP")
    .copy()
)
train_partition_df["ORDER_TIMESTAMP"] = pd.to_datetime(train_partition_df["ORDER_TIMESTAMP"])

# Remove quotes from target column if any
TARGET_COLUMN = "SALES_VOLUME"
if (TARGET_COLUMN.startswith('"') and TARGET_COLUMN.endswith('"')) or (
    TARGET_COLUMN.startswith("'") and TARGET_COLUMN.endswith("'")
):
    y_name = TARGET_COLUMN[1:-1]
else:
    y_name = TARGET_COLUMN

tabs[2].line_chart(train_partition_df, x="ORDER_TIMESTAMP", y=y_name)

# --- 9. Feature Importance ---
st.set_page_config(page_title="Feature Importance Explorer")

# Sample load: Replace with your actual model_df
# --- PREPROCESS FUNCTIONS ---
def preprocess_model_data(df):
    df["FEATURE_IMPORTANCE"] = df["METADATA"].apply(
        lambda x: (
            json.loads(x).get("feature_importance", {})
            if isinstance(x, str)
            else x.get("feature_importance", {})
        )
    )

    feature_rows = []
    for _, row in df.iterrows():
        for feature, importance in row["FEATURE_IMPORTANCE"].items():
            feature_rows.append({
                "MODEL_NAME": row["MODEL_NAME"],
                "GROUP_IDENTIFIER_STRING": row["GROUP_IDENTIFIER_STRING"],
                "FEATURE": feature,
                "IMPORTANCE": importance,
            })

    feature_df = pd.DataFrame(feature_rows)
    return df, feature_df


def calculate_average_rank(feature_df):
    feature_df = feature_df.copy()
    feature_df["RANK"] = feature_df.groupby("GROUP_IDENTIFIER_STRING")["IMPORTANCE"].rank(ascending=False)

    avg_rank_df = (
        feature_df.groupby("FEATURE")
        .agg(AVERAGE_RANK=("RANK", "mean"), AVERAGE_IMPORTANCE=("IMPORTANCE", "mean"))
        .reset_index()
        .sort_values("AVERAGE_RANK")
    )
    return feature_df, avg_rank_df


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

# --- LOAD DATA ---
# Simulate or assume model_df is already in memory
# Replace with actual DataFrame
# Ensure the "METADATA" column exists and is string/dict with "feature_importance"
model_df, feature_df = preprocess_model_data(model_df)

st.title("🔍 Feature Importance Visualization")
col1, col2 = st.columns([2, 1])
with col1:
    partition_models = sorted(model_df["GROUP_IDENTIFIER_STRING"].unique())
    selected_partition = st.selectbox("Select Partition (optional)", [None] + partition_models)
with col2:
    top_n = st.slider("Top N Features", min_value=5, max_value=50, value=5)

if selected_partition:
    filtered_df = feature_df[feature_df["GROUP_IDENTIFIER_STRING"] == selected_partition]
    fig = plot_feature_importance(filtered_df, is_aggregated=False, top_n=top_n)
else:
    feature_df, avg_rank_df = calculate_average_rank(feature_df)
    fig = plot_feature_importance(avg_rank_df, is_aggregated=True, top_n=top_n)

st.plotly_chart(fig, use_container_width=True)

# --- DATA TABLES ---
with st.expander("Show Underlying Data"):
    if selected_partition:
        st.dataframe(filtered_df.sort_values("IMPORTANCE", ascending=False))
    else:
        tabs = st.tabs(["Average Importance", "All Features"])
        tabs[0].dataframe(avg_rank_df.sort_values("AVERAGE_RANK", ascending=True))
        tabs[1].dataframe(feature_df.sort_values("IMPORTANCE", ascending=False))