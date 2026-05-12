"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2026/3/3 11:28
Description:


"""

import pandas as pd
import sys
import os

folder_address = ""
if sys.platform == "darwin":
    folder_address = "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/others/ezgi_data"
else:
    folder_address = (
        "C:/Users/Administrator/OneDrive - Brunel University London/others/ezgi_data"
    )
file_name = "survey24_JP.csv"
data_address = os.path.join(folder_address, file_name)

df_raw = pd.read_csv(data_address)
needed_columns = [
    "trust_ai_6",
    "trust_ai_3",
    "trust_ai_1",
    "trust_ai_4",
    "trust_ai_5",
    "trust_ai_7",
    "trust_ai_8",
    "trust_ai_9",
    "used_ai",
    "marital_status",
    "working_status",
    "children_in_HH",
    "household_size",
    "urban/rural",
    "education_level",
    "work_industry",
    "social_media_usage_15",
    "gross_HH_income",
    "sex",
    "age",
    "pref",
]
df_raw2 = df_raw[needed_columns].copy()
mean_columns = [
    "trust_ai_1",
    "trust_ai_4",
    "trust_ai_5",
    "trust_ai_7",
    "trust_ai_8",
    "trust_ai_9",
]
df_raw2["mean_ai"] = 7 - df_raw2[mean_columns].mean(axis=1, skipna=False)  # 7 - mean
df_linear_regression = df_raw2[
    [
        "trust_ai_6",
        "trust_ai_3",
        "mean_ai",
        "used_ai",
        "marital_status",
        "working_status",
        "children_in_HH",
        "household_size",
        "urban/rural",
        "education_level",
        "work_industry",
        "social_media_usage_15",
        "gross_HH_income",
        "sex",
        "age",
        "pref",
    ]
]
df_linear_regression = df_linear_regression.dropna(
    subset=["trust_ai_6", "trust_ai_3", "mean_ai"], how="any"
)
file_name = "JP_linear_regression_data.csv"
data_address = os.path.join(folder_address, file_name)
df_linear_regression.to_csv(data_address, index=False)
