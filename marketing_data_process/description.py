"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2026/6/9 11:41
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
file_name = "JP_linear_regression_data.csv"
data_address = os.path.join(folder_address, file_name)

df1 = pd.read_csv(data_address)
df1.groupby(by="sex").size()
df1 = df1.assign(
    sex=lambda x: x["sex"].map({1.0: "Male", 2.0: "Female"}).astype("category")
)
df1["education_level"] = df1["education_level"].map(
    {
        1.0: "No university degree",
        2.0: "No university degree",
        3.0: "No university degree",
        4.0: "University degree and above",
        5.0: "University degree and above",
        6.0: "No university degree",
        7.0: "No university degree",
    }
)
df1.rename(columns={"children_in_HH": "household_children"}, inplace=True)
df1["household_children"] = df1["household_children"].map(
    {
        0.0: "No children",
        1.0: "Having children",
        2.0: "Having children",
        3.0: "Having children",
        4.0: "Having children",
        5.0: "Having children",
        6.0: "Having children",
    }
)
df1.rename(columns={"gross_HH_income": "household_income"}, inplace=True)
df1["household_income"] = df1["household_income"].map(
    {
        1.0: "Under 5m Yen",
        2.0: "Under 5m Yen",
        3.0: "Under 5m Yen",
        4.0: "Under 5m Yen",
        5.0: "5m~7m",
        6.0: "5m~7m",
        7.0: "7m~9m",
        8.0: "7m~9m",
        9.0: "9m~12m",
        10.0: "9m~12m",
        11.0: "Over 12m",
        12.0: "Over 12m",
        13.0: "Over 12m",
    }
)
target_cols = ["sex", "education_level", "household_children", "household_income"]
frames = [df1.groupby(by=col).size() for col in target_cols]
result_df = pd.concat(frames, keys=target_cols).reset_index()
pass
