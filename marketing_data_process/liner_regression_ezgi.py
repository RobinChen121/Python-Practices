"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2026/3/3 14:11
Description:


"""

import pandas as pd
import sys
import os
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer

folder_address = ""
if sys.platform == "darwin":
    folder_address = "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/others/ezgi_data"
else:
    folder_address = (
        "C:/Users/Administrator/OneDrive - Brunel University London/others/ezgi_data"
    )
file_name = "JP_linear_regression_data.csv"
data_address = os.path.join(folder_address, file_name)

df_raw = pd.read_csv(data_address)

model0 = smf.ols(formula="trust_ai_6 ~ trust_ai_3 + mean_ai", data=df_raw).fit()
print(model0.summary())

df1 = df_raw.dropna(subset="used_ai").copy()
df1["used_ai"] = df1["used_ai"].map({1.0: "No", 2.0: "Yes"})
df1["used_ai"] = df1["used_ai"].astype("category")
model1 = smf.ols(
    formula="trust_ai_6 ~ C(used_ai, Treatment(reference='No')) + trust_ai_3 + mean_ai",
    data=df1,
).fit()
print(model1.summary())

df2 = df_raw.dropna(subset="marital_status").copy()
df2["marital_status"] = df2["marital_status"].map(
    {
        1.0: "Civil Partnership",
        2.0: "Divorced",
        3.0: "Living as married",
        4.0: "Married",
        5.0: "Never married",
        6.0: "Separated after married",
        7.0: "Widowed",
    }
)
df2["marital_status"] = df2["marital_status"].astype("category")
model2 = smf.ols(
    formula="trust_ai_6 ~ C(marital_status, Treatment(reference='Never married')) + trust_ai_3 + mean_ai",
    data=df2,
).fit()
print(model2.summary())

df3 = df_raw.dropna(subset="working_status").copy()
df3["working_status"] = df3["working_status"].map(
    {
        1.0: "Full time>40",
        2.0: "Part time8-40",
        3.0: "Part time<8",
        4.0: "Full time student",
        5.0: "Retired",
        6.0: "Self-employed",
        7.0: "Unemployed",
        8.0: "Other",
    }
)
df3["working_status"] = df3["working_status"].astype("category")
model3 = smf.ols(
    formula="trust_ai_6 ~ C(working_status, Treatment(reference='Full time>40')) + trust_ai_3 + mean_ai",
    data=df3,
).fit()
print(model3.summary())

df4 = df_raw.dropna(subset="children_in_HH").copy()
df4["children_in_HH"] = df4["children_in_HH"].map(
    {
        1.0: "0",
        2.0: "1",
        3.0: "2",
        4.0: "3",
        5.0: "4",
        6.0: ">=5",
        7.0: "Don't know",
        8.0: "Prefer not to say",
    }
)
df4["children_in_HH"] = df4["children_in_HH"].astype("category")
model4 = smf.ols(
    formula="trust_ai_6 ~ C(children_in_HH, Treatment(reference='0')) + trust_ai_3 + mean_ai",
    data=df4,
).fit()
print(model4.summary())

df5 = df_raw.dropna(subset="household_size").copy()
df5["household_size"] = df5["household_size"].map(
    {
        1.0: "1",
        2.0: "2",
        3.0: "3",
        4.0: "4",
        5.0: "5",
        6.0: "6",
        7.0: "7",
        8.0: "8 or more",
        9.0: "Don't know",
        10.0: "Prefer not to say",
    }
)
df5["household_size"] = df5["household_size"].astype("category")
model5 = smf.ols(
    formula="trust_ai_6 ~ C(household_size, Treatment(reference='1')) + trust_ai_3 + mean_ai",
    data=df5,
).fit()
print(model5.summary())

df_raw.rename(columns={"urban/rural": "urban_rural"}, inplace=True)
df6 = df_raw.dropna(subset="urban_rural").copy()
df6["urban_rural"] = df6["urban_rural"].map(
    {
        1.0: "Tokyo or designated cities",
        2.0: "Other cities",
        6.0: "Towns and villages",
    }
)
df6["urban_rural"] = df6["urban_rural"].astype("category")
model6 = smf.ols(
    formula="trust_ai_6 ~ C(urban_rural, Treatment(reference='Towns and villages')) + trust_ai_3 + mean_ai",
    data=df6,
).fit()
print(model6.summary())

df_raw.rename(columns={"education level": "education_level"}, inplace=True)
df7 = df_raw.dropna(subset="education_level").copy()
df7["education_level"] = df7["education_level"].map(
    {
        1.0: "Elementary or junior high",
        2.0: "High school",
        3.0: "Vocational/junior/training college",
        4.0: "University(4 years)",
        5.0: "University(6 years)",
        6.0: "Graduate school",
        7.0: "Other",
        8.0: "Prefer not to answer",
    }
)
df7["education_level"] = df7["education_level"].astype("category")
model7 = smf.ols(
    formula="trust_ai_6 ~ C(education_level, Treatment(reference='University(4 years)')) + trust_ai_3 + mean_ai",
    data=df7,
).fit()
print(model7.summary())

df8 = df_raw.dropna(subset="work_industry").copy()
df8["work_industry"] = df8["work_industry"].map(
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
df8["work_industry"] = df8["work_industry"].astype("category")
model8 = smf.ols(
    formula="trust_ai_6 ~ C(work_industry, Treatment(reference='Not working')) + trust_ai_3 + mean_ai",
    data=df8,
).fit()
print(model8.summary())

df_raw["social_media_usage_15"].fillna(0, inplace=True)
df9 = df_raw
df9["social_media_usage_15"] = df9["social_media_usage_15"].map(
    {
        1.0: "Yes",
        0.0: "No",
    }
)
df9["social_media_usage_15"] = df9["social_media_usage_15"].astype("category")
model9 = smf.ols(
    formula="trust_ai_6 ~ C(social_media_usage_15, Treatment(reference='Yes')) + trust_ai_3 + mean_ai",
    data=df9,
).fit()
print(model9.summary())

df10 = df_raw.dropna(subset="sex").copy()
df10["sex"] = df10["sex"].map(
    {
        1.0: "Male",
        2.0: "Female",
    }
)
df10["sex"] = df10["sex"].astype("category")
model10 = smf.ols(
    formula="trust_ai_6 ~ C(sex, Treatment(reference='Male')) + trust_ai_3 + mean_ai",
    data=df10,
).fit()
print(model10.summary())

df_raw.rename(columns={"gross_HH_income": "household_income"}, inplace=True)
df11 = df_raw.dropna(subset="household_income").copy()
df11["household_income"] = df11["household_income"].map(
    {
        1.0: "Under 2m",
        2.0: "2m~3m",
        3.0: "3m~4m",
        4.0: "4m~5m",
        5.0: "5m~6m",
        6.0: "6m~7m",
        7.0: "7m~8m",
        8.0: "8m~9m",
        9.0: "9m~10m",
        10.0: "10m~12m",
        11.0: "12m~14m",
        12.0: "14m~16m",
        13.0: "Over 16m",
        18.0: "Do not know",
        19.0: "Prefer not to answer",
    }
)
df11["household_income"] = df11["household_income"].astype("category")
model11 = smf.ols(
    formula="trust_ai_6 ~ C(household_income, Treatment(reference='Under 2m')) + trust_ai_3 + mean_ai",
    data=df11,
).fit()
print(model11.summary())

stargazer = Stargazer(
    [
        model0,
        model1,
        model2,
        model3,
        model4,
        model5,
        model6,
        model7,
        model8,
        model9,
        model10,
        model11,
    ],
)
stargazer.custom_columns(
    [
        "model0",
        "model1",
        "model2",
        "model3",
        "model4",
        "model5",
        "model6",
        "model7",
        "model8",
        "model9",
        "model10",
        "model11",
    ]
)
stargazer.show_model_numbers(False)
stargazer.title("Regression Results")
stargazer.significance_levels([0.1, 0.05, 0.01])
html = stargazer.render_html()

with open("regression.html", "w", encoding="utf-8") as f:
    f.write(html)
