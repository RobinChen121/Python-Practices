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
file_name2 = "UK_linear_regression_data.csv"
data_address2 = os.path.join(folder_address, file_name2)

df1 = pd.read_csv(data_address)
df2 = pd.read_csv(data_address2)

# groupby() 默认会忽略 NaN
# 使用 map 时，不在里面的会输出 NaN
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
df2.rename(columns={"profile_education_level": "education_level"}, inplace=True)
df2["education_level"] = df2["education_level"].map(
    {
        15.0: "University degree and above",
        16.0: "University degree and above",
        17.0: "University degree and above",
        1.0: "No university degree",
        2.0: "No university degree",
        3.0: "No university degree",
        4.0: "No university degree",
        5.0: "No university degree",
        6.0: "No university degree",
        7.0: "No university degree",
        8.0: "No university degree",
        9.0: "No university degree",
        10.0: "No university degree",
        11.0: "No university degree",
        12.0: "No university degree",
        13.0: "No university degree",
        14.0: "No university degree",
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
df2.rename(columns={"profile_household_children": "household_children"}, inplace=True)
df2["household_children"] = df2["household_children"].map(
    {
        1.0: "No children",
        2.0: "Having children",
        3.0: "Having children",
        4.0: "Having children",
        5.0: "Having children",
        6.0: "Having children",
        7.0: "Having children",
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
df2.rename(columns={"profile_gross_household": "household_income"}, inplace=True)
df2 = df2.dropna(subset="household_income").copy()
df2["household_income"] = df2["household_income"].map(
    {
        1.0: "Under 25k pounds",
        2.0: "Under 25k pounds",
        3.0: "Under 25k pounds",
        4.0: "Under 25k pounds",
        5.0: "Under 25k pounds",
        6.0: "25k-35k",
        7.0: "25k-35k",
        8.0: "35k-45k",
        9.0: "35k-45k",
        10.0: "45k-60k",
        11.0: "45k-60k",
        12.0: "Over 60k",
        13.0: "Over 60k",
        14.0: "Over 60k",
        15.0: "Over 60k",
    }
)

df1["marital_status"] = df1["marital_status"].map(
    {
        1.0: "Married",
        2.0: "Married",
        3.0: "Married",
        4.0: "Married",
        5.0: "Never married",
        6.0: "Married",
        7.0: "Married",
    }
)
df2.rename(columns={"profile_marital_stat": "marital_status"}, inplace=True)
df2["marital_status"] = df2["marital_status"].map(
    {
        1.0: "Married",
        2.0: "Married",
        3.0: "Married",
        4.0: "Married",
        5.0: "Married",
        6.0: "Never married",
        7.0: "Married",
    }
)

df1["sex"] = df1["sex"].map({1.0: "Male", 2.0: "Female"})
df2.rename(columns={"profile_gender": "sex"}, inplace=True)
df2["sex"] = df2["sex"].map(
    {
        1.0: "Male",
        2.0: "Female",
    }
)

df1.rename(columns={"social_media_usage_15": "social_media_active"}, inplace=True)
df1.fillna({"social_media_active": 0}, inplace=True)
df1["social_media_active"] = df1["social_media_active"].map(
    {
        1.0: "Yes",
        0.0: "No",
    }
)
df2.rename(
    columns={"social_media_activemember_97": "social_media_active"}, inplace=True
)
df2["social_media_active"] = df2["social_media_active"].map(
    {
        1.0: "Yes",
        2.0: "No",
    }
)

df1.rename(columns={"urban/rural": "urban_rural"}, inplace=True)
df1["urban_rural"] = df1["urban_rural"].map(
    {
        1.0: "Cities",
        2.0: "Cities",
        6.0: "Others",
    }
)
df2.rename(columns={"ONS_urban": "urban_rural"}, inplace=True)
df2["urban_rural"] = df2["urban_rural"].map(
    {
        1.0: "Cities",
        2.0: "Others",
        3.0: "Others",
    }
)

df1["work_industry"] = df1["work_industry"].map(
    {
        1.0: "Agriculture, forestry, fishery",
        2.0: "Mining, energy, construction",
        3.0: "Manufacturing",
        4.0: "Finance, insurance, real estate",
        5.0: "Education",
        6.0: "Medicare and nursing service",
        7.0: "Transportation, tourism",
        8.0: "Other service industry",
        9.0: "Public services",
        10.0: "Other",
        11.0: "Not working",
        12.0: "Newspaper publishing",
    }
)
df2.rename(columns={"work_sector": "work_industry"}, inplace=True)
df2["work_industry"] = df2["work_industry"].map(
    {
        1.0: "Private sector",
        2.0: "Public sector",
        3.0: "Third/voluntary sector",
    }
)

target_cols = [
    "education_level",
    "household_children",
    "household_income",
    "marital_status",
    "sex",
    "social_media_active",
    "urban_rural",
    "work_industry",
]

# frames = [df1.groupby(by=col).size() for col in target_cols]
# result_df = pd.concat(frames, keys=target_cols).reset_index()
# frames2 = [df2.groupby(by=col).size() for col in target_cols]
# result_df2 = pd.concat(frames2, keys=target_cols).reset_index()
# result_df.to_csv("jp.csv")
# result_df2.to_csv("uk.csv")

## t test
from scipy import stats
print(df1['age'].mean(), df2['age'].mean())
print(stats.ttest_ind(df1['age'], df2['age']), end='\n\n')

print(df1['mean_ai'].mean(), df2['mean_ai'].mean())
print(stats.ttest_ind(df1['mean_ai'], df2['mean_ai']), end='\n\n')

print(df1['trust_ai_3'].mean(), df2['trust_ai_3'].mean())
print(stats.ttest_ind(df1['trust_ai_3'], df2['trust_ai_3']), end='\n\n')

print(df1['trust_ai_6'].mean(), df2['trust_ai_6'].mean())
print(stats.ttest_ind(df1['trust_ai_6'], df2['trust_ai_6']), end='\n\n')
