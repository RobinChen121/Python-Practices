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

# mean_ai as IV
# trust_ai_6 as DV
# 2 model: trust_ai_3 as DV
# mean_ai as DV
DV = "trust_ai_3"
IV = " + mean_ai + trust_ai_6"
# DV = "trust_ai_6"
# IV = " + mean_ai + trust_ai_3"
# DV = "mean_ai"
# IV = " + trust_ai_3 + trust_ai_6"
model0 = smf.ols(formula=DV + " ~ age" + IV, data=df_raw).fit()
print(model0.summary())

df_raw.rename(columns={"profile_education_level": "education_level"}, inplace=True)
df1 = df_raw.dropna(subset="education_level").copy()
df1["education_level"] = df1["education_level"].map(
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
df1["education_level"] = df1["education_level"].astype("category")
model1 = smf.ols(
    formula=DV
    + " ~ C(education_level, Treatment(reference='No university degree')) + age"
    + IV,
    data=df1,
).fit()
print(model1.summary())

df_raw.rename(
    columns={"profile_household_children": "household_children"}, inplace=True
)
df2 = df_raw.dropna(subset="household_children").copy()
df2 = df2[~df2["household_children"].isin([9.0, 8.0])]
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
df2["household_children"] = df2["household_children"].astype("category")
model2 = smf.ols(
    formula=DV
    + " ~ C(household_children, Treatment(reference='No children')) + age"
    + IV,
    data=df2,
).fit()
print(model2.summary())

df_raw.rename(columns={"profile_gross_household": "household_income"}, inplace=True)
df3 = df_raw.dropna(subset="household_income").copy()
df3["household_income"] = df3["household_income"].map(
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
df3["household_income"] = df3["household_income"].astype("category")
model3 = smf.ols(
    formula=DV
    + " ~ C(household_income, Treatment(reference='Under 25k pounds')) + age"
    + IV,
    data=df3,
).fit()
print(model3.summary())

df_raw.rename(columns={"profile_marital_stat": "marital_status"}, inplace=True)
df4 = df_raw.dropna(subset="marital_status").copy()
df4["marital_status"] = df4["marital_status"].map(
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
df4["marital_status"] = df4["marital_status"].astype("category")
model4 = smf.ols(
    formula=DV
    + " ~ C(marital_status, Treatment(reference='Never married')) + age"
    + IV,
    data=df4,
).fit()
print(model4.summary())

df_raw.rename(columns={"profile_gender": "sex"}, inplace=True)
df5 = df_raw.dropna(subset="sex").copy()
df5["sex"] = df5["sex"].map(
    {
        1.0: "Male",
        2.0: "Female",
    }
)
df5["sex"] = df5["sex"].astype("category")
model5 = smf.ols(
    formula=DV + " ~ C(sex, Treatment(reference='Male')) + age" + IV,
    data=df5,
).fit()
print(model5.summary())

df6 = df_raw
df6["social_media_activemember_97"] = df6["social_media_activemember_97"].map(
    {
        1.0: "Yes",
        2.0: "No",
    }
)
df6["social_media_activemember_97"] = df6["social_media_activemember_97"].astype(
    "category"
)
model6 = smf.ols(
    formula=DV
    + " ~ C(social_media_activemember_97, Treatment(reference='No')) + age"
    + IV,
    data=df6,
).fit()
print(model6.summary())

df_raw.rename(columns={"ONS_urban": "urban_rural"}, inplace=True)
df7 = df_raw.dropna(subset="urban_rural").copy()
df7["urban_rural"] = df7["urban_rural"].map(
    {
        1.0: "Cities",
        2.0: "Others",
        3.0: "Others",
    }
)
df7["urban_rural"] = df7["urban_rural"].astype("category")
model7 = smf.ols(
    formula=DV + " ~ C(urban_rural, Treatment(reference='Others')) + age" + IV,
    data=df7,
).fit()
print(model7.summary())

df8 = df_raw.dropna(subset="used_ai").copy()
df8["used_ai"] = df8["used_ai"].map({1.0: "No", 2.0: "Yes"})
df8["used_ai"] = df8["used_ai"].astype("category")
model8 = smf.ols(
    formula=DV + " ~ C(used_ai, Treatment(reference='No')) + age" + IV,
    data=df8,
).fit()
print(model8.summary())


# df_raw.rename(columns={"profile_household_size": "household_size"}, inplace=True)
# df5 = df_raw.dropna(subset="household_size").copy()
# df5 = df5[~df5["household_size"].isin([9.0, 10.0])]
# df5["household_size"] = df5["household_size"].map(
#     {
#         1.0: "1",
#         2.0: "2",
#         3.0: "3",
#         4.0: "4",
#         5.0: "5",
#         6.0: "6",
#         7.0: "7",
#         8.0: "8 or more",
#     }
# )
# df5["household_size"] = df5["household_size"].astype("category")
# model5 = smf.ols(
#     formula=DV + " ~ C(household_size, Treatment(reference='1')) + trust_ai_3 + mean_ai",
#     data=df5,
# ).fit()
# print(model5.summary())


df_raw.rename(columns={"work_sector": "work_industry"}, inplace=True)
df9 = df_raw.dropna(subset="work_industry").copy()
df9["work_industry"] = df9["work_industry"].map(
    {
        1.0: "Private sector",
        2.0: "Public sector",
        3.0: "Third/voluntary sector",
    }
)
df9["work_industry"] = df9["work_industry"].astype("category")
model9 = smf.ols(
    formula=DV + " ~ C(work_industry, Treatment(reference='Public sector')) + age" + IV,
    data=df9,
).fit()
print(model9.summary())


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
    ]
)
stargazer.show_model_numbers(False)
stargazer.title("Regression Results")
stargazer.significance_levels([0.1, 0.05, 0.01])
html = stargazer.render_html()

out_name = "DV" + "-" + DV + "-regression_uk.html"
with open(out_name, "w", encoding="utf-8") as f:
    f.write(html)
