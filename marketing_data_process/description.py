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
        1.0: "Elementary or junior high school",
        2.0: "High school",
        3.0: "Vocational college, junior college, training college",
        4.0: "University (Four years)",
        5.0: "University (Six years)",
        6.0: "Graduate school",
        7.0: "Other",
    }
)
df2.rename(columns={"profile_education_level": "education_level"}, inplace=True)
df2["education_level"] = df2["education_level"].map(
    {
        15.0: "University diploma",
        16.0: "University or CNAA first degree (e.g. BA, B.Sc, B.Ed)",
        17.0: "University or CNAA higher degree (e.g. M.Sc, Ph.D)",
        18.0: "Other technical, professional or higher qualification",
        1.0: "No formal qualifications",
        2.0: "Youth training certificate/skill seekers",
        3.0: "Recognised trade apprenticeship completed",
        4.0: "Clerical and commercial",
        5.0: "City & Guilds certificate",
        6.0: "City & Guilds certificate - advanced",
        7.0: "ONC",
        8.0: "CSE grades 2-5",
        9.0: "CSE grade 1, GCE O level, GCSE, School Certificate",
        10.0: "Scottish Ordinary/ Lower Certificate",
        11.0: "GCE A level or Higher Certificate",
        12.0: "Scottish Higher Certificate",
        13.0: "Nursing qualification (e.g. SEN, SRN, SCM, RGN)",
        14.0: "Teaching qualification (not degree)",
    }
)

df1.rename(columns={"children_in_HH": "household_children"}, inplace=True)
df1["household_children"] = df1["household_children"].map(
    {
        0.0: "No children",
        1.0: "1 children",
        2.0: "2 children",
        3.0: "3 children",
        4.0: "4 children",
        5.0: "5 children or more",
    }
)
df2.rename(columns={"profile_household_children": "household_children"}, inplace=True)
df2["household_children"] = df2["household_children"].map(
    {
        1.0: "No children",
        2.0: "1 child",
        3.0: "2 children",
        4.0: "3 children",
        5.0: "4 children",
        6.0: "5 children or more",
    }
)

df1.rename(columns={"gross_HH_income": "household_income"}, inplace=True)
df1["household_income"] = df1["household_income"].map(
    {
        1.0: "Under 2m Yen",
        2.0: "2m~3m",
        3.0: "3m~4m",
        4.0: "4m~5m",
        5.0: "5m~6m",
        6.0: "6m~7m",
        7.0: "7m~8m",
        8.0: "8m~9m",
        9.0: "9m~10m",
        10.0: "10m~12m",
        11.0: "11m~14m",
        12.0: "14m~16m",
        13.0: "Over 16m",
    }
)
df2.rename(columns={"profile_gross_household": "household_income"}, inplace=True)
df2 = df2.dropna(subset="household_income").copy()
df2["household_income"] = df2["household_income"].map(
    {
        1.0: "under 5k pounds",
        2.0: "5k~10k",
        3.0: "10k~15k",
        4.0: "15k~20k",
        5.0: "20k~25k",
        6.0: "25k-30k",
        7.0: "30k-35k",
        8.0: "35k-40k",
        9.0: "40k-45k",
        10.0: "45k-50k",
        11.0: "50k-60k",
        12.0: "60k~70k",
        13.0: "70k~100k",
        14.0: "100k~150k",
        15.0: "Over 150k",
    }
)

df1["marital_status"] = df1["marital_status"].map(
    {
        1.0: "Civil partnership",
        2.0: "Divorced",
        3.0: "Living as married",
        4.0: "Married",
        5.0: "Never married",
        6.0: "Separated after being married",
        7.0: "Widowed",
    }
)
df2.rename(columns={"profile_marital_stat": "marital_status"}, inplace=True)
df2["marital_status"] = df2["marital_status"].map(
    {
        1.0: "Married",
        2.0: "Living as married",
        3.0: "Separated after being married",
        4.0: "Divorced",
        5.0: "Widowed",
        6.0: "Never married",
        7.0: "Civil partnership",
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
        1.0: "Tokyo or designated cities",
        2.0: "Other cities",
        6.0: "Towns and villages",
    }
)
df2.rename(columns={"ONS_urban": "urban_rural"}, inplace=True)
df2["urban_rural"] = df2["urban_rural"].map(
    {
        1.0: "Urban",
        2.0: "Town and Fringe",
        3.0: "Rural",
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

df1["used_ai"] = df1["used_ai"].map(
    {
        1.0: "No",
        2.0: "Yes"
    }
)
df2["used_ai"] = df2["used_ai"].map(
    {
        1.0: "No",
        2.0: "Yes"
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
    "used_ai"
]

frames = [df1.groupby(by=col).size() for col in target_cols]
result_df = pd.concat(frames, keys=target_cols).reset_index()
frames2 = [df2.groupby(by=col).size() for col in target_cols]
result_df2 = pd.concat(frames2, keys=target_cols).reset_index()
result_df.to_csv("jp.csv")
result_df2.to_csv("uk.csv")

## t test
from scipy import stats
print(df1['age'].mean(), df2['age'].mean())
print(df1['age'].count(), df2['age'].count())
print(stats.ttest_ind(df1['age'], df2['age']), end='\n\n')

print(df1['mean_ai'].mean(), df2['mean_ai'].mean())
print(df1['mean_ai'].count(), df2['mean_ai'].count())
print(stats.ttest_ind(df1['mean_ai'], df2['mean_ai']), end='\n\n')

print(df1['trust_ai_3'].mean(), df2['trust_ai_3'].mean())
print(df1['mean_ai'].count(), df2['mean_ai'].count())
print(stats.ttest_ind(df1['trust_ai_3'], df2['trust_ai_3']), end='\n\n')

print(df1['trust_ai_6'].mean(), df2['trust_ai_6'].mean())
print(df1['mean_ai'].count(), df2['mean_ai'].count())
print(stats.ttest_ind(df1['trust_ai_6'], df2['trust_ai_6']), end='\n\n')
