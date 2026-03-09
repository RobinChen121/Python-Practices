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
file_name = "UK_linear_regression_data.csv"
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

df_raw.rename(columns={"profile_marital_stat": "marital_status"}, inplace=True)
df2 = df_raw.dropna(subset="marital_status").copy()
df2["marital_status"] = df2["marital_status"].map(
    {
        1.0: "Married",
        2.0: "Living as married",
        3.0: "Separated after married",
        4.0: "Divorced",
        5.0: "Widowed",
        6.0: "Never married",
        7.0: "Civil Partnership",
    }
)
df2["marital_status"] = df2["marital_status"].astype("category")
model2 = smf.ols(
    formula="trust_ai_6 ~ C(marital_status, Treatment(reference='Married')) + trust_ai_3 + mean_ai",
    data=df2,
).fit()
print(model2.summary())

# df3 = df_raw.dropna(subset="working_status").copy()
# df3["working_status"] = df3["working_status"].map(
#     {
#         1.0: "Full time>40",
#         2.0: "Part time8-40",
#         3.0: "Part time<8",
#         4.0: "Full time student",
#         5.0: "Retired",
#         6.0: "Self-employed",
#         7.0: "Unemployed",
#         8.0: "Other",
#     }
# )
# df3["working_status"] = df3["working_status"].astype("category")
# model3 = smf.ols(
#     formula="trust_ai_6 ~ C(working_status, Treatment(reference='Full time>40')) + trust_ai_3 + mean_ai",
#     data=df3,
# ).fit()
# print(model3.summary())

df_raw.rename(
    columns={"profile_household_children": "household_children"}, inplace=True
)
df4 = df_raw.dropna(subset="household_children").copy()
df4 = df4[~df4["household_children"].isin([9.0, 8.0])]
df4["household_children"] = df4["household_children"].map(
    {
        1.0: "0",
        2.0: "1",
        3.0: "2",
        4.0: "3",
        5.0: "4",
        6.0: ">=5",
        7.0: ">=5",
    }
)
df4["household_children"] = df4["household_children"].astype("category")
model4 = smf.ols(
    formula="trust_ai_6 ~ C(household_children, Treatment(reference='0')) + trust_ai_3 + mean_ai",
    data=df4,
).fit()
print(model4.summary())

df_raw.rename(columns={"profile_household_size": "household_size"}, inplace=True)
df5 = df_raw.dropna(subset="household_size").copy()
df5 = df5[~df5["household_size"].isin([9.0, 10.0])]
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
    }
)
df5["household_size"] = df5["household_size"].astype("category")
model5 = smf.ols(
    formula="trust_ai_6 ~ C(household_size, Treatment(reference='1')) + trust_ai_3 + mean_ai",
    data=df5,
).fit()
print(model5.summary())

df_raw.rename(columns={"ONS_urban": "urban_rural"}, inplace=True)
df6 = df_raw.dropna(subset="urban_rural").copy()
df6["urban_rural"] = df6["urban_rural"].map(
    {
        1.0: "Cities",
        2.0: "Others",
        3.0: "Others",
    }
)
df6["urban_rural"] = df6["urban_rural"].astype("category")
model6 = smf.ols(
    formula="trust_ai_6 ~ C(urban_rural, Treatment(reference='Others')) + trust_ai_3 + mean_ai",
    data=df6,
).fit()
print(model6.summary())

df_raw.rename(columns={"profile_education_level": "education_level"}, inplace=True)
df7 = df_raw.dropna(subset="education_level").copy()
df7["education_level"] = df7["education_level"].map(
    {
        15.0: "University degree",
        16.0: "University degree",
        17.0: "University degree",
        1.0: "Others",
        2.0: "Others",
        3.0: "Others",
        4.0: "Others",
        5.0: "Others",
        6.0: "Others",
        7.0: "Others",
        8.0: "Others",
        9.0: "Others",
        10.0: "Others",
        11.0: "Others",
        12.0: "Others",
        13.0: "Others",
        14.0: "Others",
    }
)
df7["education_level"] = df7["education_level"].astype("category")
model7 = smf.ols(
    formula="trust_ai_6 ~ C(education_level, Treatment(reference='Others')) + trust_ai_3 + mean_ai",
    data=df7,
).fit()
print(model7.summary())

df_raw.rename(columns={"work_sector": "work_industry"}, inplace=True)
df8 = df_raw.dropna(subset="work_industry").copy()
df8["work_industry"] = df8["work_industry"].map(
    {
        1.0: "Private sector",
        2.0: "Public sector",
        3.0: "Third/voluntary sector",
    }
)
df8["work_industry"] = df8["work_industry"].astype("category")
model8 = smf.ols(
    formula="trust_ai_6 ~ C(work_industry, Treatment(reference='Public sector')) + trust_ai_3 + mean_ai",
    data=df8,
).fit()
print(model8.summary())

df9 = df_raw
df9["social_media_activemember_97"] = df9["social_media_activemember_97"].map(
    {
        1.0: "Yes",
        2.0: "No",
    }
)
df9["social_media_activemember_97"] = df9["social_media_activemember_97"].astype(
    "category"
)
model9 = smf.ols(
    formula="trust_ai_6 ~ C(social_media_activemember_97, Treatment(reference='No')) + trust_ai_3 + mean_ai",
    data=df9,
).fit()
print(model9.summary())

df_raw.rename(columns={"profile_gender": "sex"}, inplace=True)
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

df_raw.rename(columns={"profile_gross_household": "household_income"}, inplace=True)
df11 = df_raw.dropna(subset="household_income").copy()
df11["household_income"] = df11["household_income"].map(
    {
        1.0: "Under 5k pounds",
        2.0: "5k-10k",
        3.0: "10k-15k",
        4.0: "15k-20k",
        5.0: "20k-25k",
        6.0: "25k-30k",
        7.0: "30k-35k",
        8.0: "35k-40k",
        9.0: "40k-45k",
        10.0: "45k-50k",
        11.0: "50k-60k",
        12.0: "60k-70k",
        13.0: "70k-100k",
        14.0: "100k-150k",
        15.0: "> 150k",
    }
)
df11["household_income"] = df11["household_income"].astype("category")
model11 = smf.ols(
    formula="trust_ai_6 ~ C(household_income, Treatment(reference='Under 5k pounds')) + trust_ai_3 + mean_ai",
    data=df11,
).fit()
print(model11.summary())

df_raw.rename(columns={"profile_gross_personal": "personal_income"}, inplace=True)
df12 = df_raw.dropna(subset="personal_income").copy()
df12["personal_income"] = df12["personal_income"].map(
    {
        1.0: "Under 5k pounds",
        2.0: "5k-10k",
        3.0: "10k-15k",
        4.0: "15k-20k",
        5.0: "20k-25k",
        6.0: "25k-30k",
        7.0: "30k-35k",
        8.0: "35k-40k",
        9.0: "40k-45k",
        10.0: "45k-50k",
        11.0: "50k-60k",
        12.0: "60k-70k",
        13.0: "70k-100k",
        14.0: ">100k",
    }
)
df12["personal_income"] = df12["personal_income"].astype("category")
model12 = smf.ols(
    formula="trust_ai_6 ~ C(personal_income, Treatment(reference='Under 5k pounds')) + trust_ai_3 + mean_ai",
    data=df12,
).fit()
print(model12.summary())

stargazer = Stargazer(
    [
        model0,
        model1,
        model2,
        # model3,
        model4,
        model5,
        model6,
        model7,
        model8,
        model9,
        model10,
        model11,
        model12,
    ],
)
stargazer.custom_columns(
    [
        "model0",
        "model1",
        "model2",
        # "model3",
        "model4",
        "model5",
        "model6",
        "model7",
        "model8",
        "model9",
        "model10",
        "model11",
        "model12",
    ]
)
stargazer.show_model_numbers(False)
stargazer.title("Regression Results")
stargazer.significance_levels([0.1, 0.05, 0.01])
html = stargazer.render_html()

with open("regression_uk.html", "w", encoding="utf-8") as f:
    f.write(html)
