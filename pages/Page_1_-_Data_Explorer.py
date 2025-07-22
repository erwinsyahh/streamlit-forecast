import streamlit as st
from utils.load_data import load_all_tables
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Load Data
st.title("Training Data Trends")

try:
    train_df, _, _, _ = load_all_tables()
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

# Select partition columns
partition_cols = st.multiselect(
    "Select Partition Columns",
    options=[col for col in train_df.columns if train_df[col].dtype == object or train_df[col].dtype.name == 'category'],
    default=["STORE_ID", "PRODUCT_ID"]
)

if not partition_cols:
    st.warning("Please select at least one partition column.")
    st.stop()

if "ORDER_TIMESTAMP" not in train_df.columns:
    st.error("ORDER_TIMESTAMP column missing.")
    st.stop()

train_df["ORDER_TIMESTAMP"] = pd.to_datetime(train_df["ORDER_TIMESTAMP"], errors="coerce")
train_df.dropna(subset=["ORDER_TIMESTAMP"], inplace=True)

# GROUP_IDENTIFIER
train_df["GROUP_IDENTIFIER"] = train_df[partition_cols].astype(str).agg("_".join, axis=1)

# Top groups
visualize_count = st.slider("Number of Partitions to Visualize", min_value=1, max_value=20, value=5)
top_groups = train_df["GROUP_IDENTIFIER"].drop_duplicates().head(visualize_count).tolist()
filtered_df = train_df[train_df["GROUP_IDENTIFIER"].isin(top_groups)].copy()

available_partitions = sorted(filtered_df["GROUP_IDENTIFIER"].unique())
if not available_partitions:
    st.warning("No data available.")
    st.stop()

chosen_partition = st.selectbox("Select Partition to Visualize", available_partitions)
partition_df = filtered_df[filtered_df["GROUP_IDENTIFIER"] == chosen_partition].copy()

# Show sample
st.subheader(f"Sample Records for: {chosen_partition}")
st.dataframe(partition_df.head(5))

# Aggregation
agg_choice = st.selectbox("Aggregate by", options=["Daily", "Weekly", "Monthly"])

if agg_choice == "Daily":
    partition_df["PERIOD"] = partition_df["ORDER_TIMESTAMP"].dt.date
elif agg_choice == "Weekly":
    partition_df["PERIOD"] = partition_df["ORDER_TIMESTAMP"].dt.to_period("W").dt.start_time
elif agg_choice == "Monthly":
    partition_df["PERIOD"] = partition_df["ORDER_TIMESTAMP"].dt.to_period("M").dt.start_time

agg_df = partition_df.groupby("PERIOD")["SALES_VOLUME"].sum().reset_index()

# Plot
fig = px.line(
    agg_df,
    x="PERIOD",
    y="SALES_VOLUME",
    title=f"{agg_choice} Sales Volume - {chosen_partition}"
)
st.plotly_chart(fig, use_container_width=True)

# --- Static Correlation Matrix ---
st.subheader("Correlation Matrix for Selected Partition")
numeric_cols = partition_df.select_dtypes(include=["number"]).columns.tolist()

if len(numeric_cols) < 2:
    st.warning("Not enough numerical columns to compute correlation.")
else:
    corr = partition_df[numeric_cols].corr()
    fig_corr, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig_corr)

# --- Stationarity & Time Series Diagnostics ---
from matplotlib import rcParams
rcParams.update({"font.size": 8})

if len(agg_df) < 30:
    st.warning("Not enough data points for time series analysis (need at least 30).")
else:
    time_series = agg_df.set_index("PERIOD")["SALES_VOLUME"].astype(float)

    # --- Seasonal Decomposition ---
    st.subheader(f"Seasonal Decomposition (Period = 30) for: {chosen_partition}")
    try:
        result = seasonal_decompose(time_series, model="additive", period=30)
        fig_decomp, axs = plt.subplots(4, 1, figsize=(7, 6), sharex=True)

        axs[0].plot(result.observed, label="Observed", color="blue")
        axs[0].legend(loc="upper left", fontsize=7)

        axs[1].plot(result.trend, label="Trend", color="orange")
        axs[1].legend(loc="upper left", fontsize=7)

        axs[2].plot(result.seasonal, label="Seasonal", color="green")
        axs[2].legend(loc="upper left", fontsize=7)

        axs[3].plot(result.resid, label="Residuals", color="red")
        axs[3].legend(loc="upper left", fontsize=7)

        for ax in axs:
            ax.tick_params(axis="both", labelsize=7)

        plt.tight_layout()
        st.pyplot(fig_decomp)

    except Exception as e:
        st.error(f"Decomposition failed: {e}")
        
    # --- Augmented Dickey-Fuller Test ---
    adf_result = adfuller(time_series.dropna())
    statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    # Compose readable Markdown without using join
    st.markdown(f"""
    **ADF Statistic:** `{statistic:.4f}`  
    **p-value:** `{p_value:.4f}`  

    **Critical Values:**  
    - **1%**: `{critical_values['1%']:.4f}`  
    - **5%**: `{critical_values['5%']:.4f}`  
    - **10%**: `{critical_values['10%']:.4f}`  

    **Conclusion:** {"✅ The series is likely *stationary*" if p_value < 0.05 else "❌ The series is likely *not stationary*"}
    """)

    # --- ACF and PACF ---
    st.subheader(f"ACF and PACF Analysis for: {chosen_partition}")
    fig_acf_pacf, ax = plt.subplots(2, 1, figsize=(7, 4.5))
    plot_acf(time_series.dropna(), lags=40, ax=ax[0])
    ax[0].set_title("Autocorrelation (ACF)", fontsize=9)
    ax[0].tick_params(axis="both", labelsize=7)

    plot_pacf(time_series.dropna(), lags=40, ax=ax[1])
    ax[1].set_title("Partial Autocorrelation (PACF)", fontsize=9)
    ax[1].tick_params(axis="both", labelsize=7)

    plt.tight_layout()
    st.pyplot(fig_acf_pacf)
