import pandas as pd
import numpy as np
from snowflake.snowpark import Session
import streamlit as st
import json 
from snowflake.snowpark.context import get_active_session

@st.cache_data()
def load_all_tables():
    session = get_active_session()
    # session = Session.builder.config("connection_name", "poc_nojo").create() # Comment unnecessary 1
    print(session)
    train_df = session.table("POC_NOJORONO.ML_FORECAST.DAILY_PARTITIONED_SAMPLE_DATA").to_pandas()
    test_pred_df = session.table("POC_NOJORONO.NOTEBOOKS.VALIDATION_PREDS_FROM_TEST_MODEL_1").to_pandas()
    result_df = session.table("POC_NOJORONO.NOTEBOOKS.FORECAST_RESULTS").to_pandas()
    model_df = session.table("POC_NOJORONO.NOTEBOOKS.MODEL_STORAGE_TEST_MODEL_1").to_pandas()
    session.close()

    return train_df, test_pred_df, result_df, model_df

def compute_row_metrics(df):
    df = df.copy()
    df["ABS_ERROR"] = (df["ACTUAL"] - df["PREDICTED"]).abs()
    df["APE"] = np.where(df["ACTUAL"] == 0, np.nan, df["ABS_ERROR"] / df["ACTUAL"])
    df["SQ_ERROR"] = (df["ACTUAL"] - df["PREDICTED"]) ** 2
    return df

def compute_partition_metrics(df):
    grouped = df.groupby("GROUP_IDENTIFIER_STRING").agg(
        MAPE=("APE", "mean"),
        MAE=("ABS_ERROR", "mean"),
        RMSE=("SQ_ERROR", lambda x: np.sqrt(np.mean(x)))
    ).reset_index()
    grouped[["MAPE", "MAE", "RMSE"]] = grouped[["MAPE", "MAE", "RMSE"]].round(3)
    return grouped

def compute_overall_metrics(partition_metrics):
    overall_avg = pd.DataFrame([{
        "AGGREGATION": "AVG",
        "OVERALL_MAPE": partition_metrics["MAPE"].mean(),
        "OVERALL_MAE": partition_metrics["MAE"].mean(),
        "OVERALL_RMSE": partition_metrics["RMSE"].mean(),
    }])

    overall_median = pd.DataFrame([{
        "AGGREGATION": "MEDIAN",
        "OVERALL_MAPE": partition_metrics["MAPE"].median(),
        "OVERALL_MAE": partition_metrics["MAE"].median(),
        "OVERALL_RMSE": partition_metrics["RMSE"].median(),
    }])

    overall = pd.concat([overall_avg, overall_median], ignore_index=True)
    overall[["OVERALL_MAPE", "OVERALL_MAE", "OVERALL_RMSE"]] = overall[["OVERALL_MAPE", "OVERALL_MAE", "OVERALL_RMSE"]].round(3)
    return overall

def preprocess_model_data(df):
    df = df.copy()
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
