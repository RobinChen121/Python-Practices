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

# mean_ai as IV
# trust_ai_6 as DV
# 2 model: trust_ai_3 as DV
# mean_ai as DV
# DV = "trust_ai_3"
# IV = " + mean_ai + trust_ai_6"
# DV = "trust_ai_6"
# IV = " + mean_ai + trust_ai_3"
DV = "mean_ai"
IV = " + trust_ai_3 + trust_ai_6"
model0 = smf.ols(formula=DV + " ~ age" + IV, data=df_raw).fit()
print(model0.summary())

df_raw.rename(columns={"education level": "education_level"}, inplace=True)
df1 = df_raw.dropna(subset="education_level").copy()
df1 = df1[~df1["household_size"].isin([8.0])]
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
df1["education_level"] = df1["education_level"].astype("category")
model1 = smf.ols(
    formula=DV
    + " ~ C(education_level, Treatment(reference='No university degree')) + age"
    + IV,
    data=df1,
).fit()
print(model1.summary())

df_raw.rename(columns={"children_in_HH": "household_children"}, inplace=True)
df2 = df_raw.dropna(subset="household_children").copy()
df2 = df2[~df2["household_children"].isin([7.0, 8.0])]
df2["household_children"] = df2["household_children"].map(
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

df2["household_children"] = df2["household_children"].astype("category")
model2 = smf.ols(
    formula=DV
    + " ~ C(household_children, Treatment(reference='No children')) + age"
    + IV,
    data=df2,
).fit()
print(model2.summary())

# df2 = df_raw.dropna(subset="household_size").copy()
# df2 = df2[~df2["household_size"].isin([9.0, 10.0])]
# df2["household_size"] = df2["household_size"].map(
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
# df2["household_size"] = df2["household_size"].astype("category")
# model2 = smf.ols(
#     formula=DV + " ~ C(household_size, Treatment(reference='1')) + trust_ai_3 + mean_ai",
#     data=df2,
# ).fit()
# print(model2.summary())

df_raw.rename(columns={"gross_HH_income": "household_income"}, inplace=True)
df3 = df_raw.dropna(subset="household_income").copy()
df3 = df3[~df3["household_income"].isin([18.0, 19.0])]
df3["household_income"] = df3["household_income"].map(
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
df3["household_income"] = df3["household_income"].astype("category")
model3 = smf.ols(
    formula=DV
    + " ~ C(household_income, Treatment(reference='Under 5m Yen')) + age"
    + IV,
    data=df3,
).fit()
print(model3.summary())

df4 = df_raw.dropna(subset="marital_status").copy()
df4["marital_status"] = df4["marital_status"].map(
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
df4["marital_status"] = df4["marital_status"].astype("category")
model4 = smf.ols(
    formula=DV
    + " ~ C(marital_status, Treatment(reference='Never married')) + age"
    + IV,
    data=df4,
).fit()
print(model4.summary())

df5 = df_raw.dropna(subset="sex").copy()
# 一次性处理映射和类型转换
df5 = df5.assign(
    sex=lambda x: x["sex"].map({1.0: "Male", 2.0: "Female"}).astype("category")
)
df5["sex"] = df5["sex"].astype("category")
model5 = smf.ols(
    formula=DV + " ~ C(sex, Treatment(reference='Male')) + age" + IV,
    data=df5,
).fit()
print(model5.summary())

df_raw["social_media_usage_15"].fillna(0, inplace=True)
df6 = df_raw
df6["social_media_usage_15"] = df6["social_media_usage_15"].map(
    {
        1.0: "Yes",
        0.0: "No",
    }
)
df6["social_media_usage_15"] = df6["social_media_usage_15"].astype("category")
model6 = smf.ols(
    formula=DV + " ~ C(social_media_usage_15, Treatment(reference='No')) + age" + IV,
    data=df6,
).fit()
print(model6.summary())

df_raw.rename(columns={"urban/rural": "urban_rural"}, inplace=True)
df7 = df_raw.dropna(subset="urban_rural").copy()
df7["urban_rural"] = df7["urban_rural"].map(
    {
        1.0: "Cities",
        2.0: "Cities",
        6.0: "Others",
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

df9 = df_raw.dropna(subset="work_industry").copy()
df9["work_industry"] = df9["work_industry"].map(
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
df9["work_industry"] = df9["work_industry"].astype("category")
model9 = smf.ols(
    formula=DV
    + " ~ C(work_industry, Treatment(reference='Public services')) + age"
    + IV,
    data=df9,
).fit()
print(model9.summary())

# df10 = df_raw.dropna(subset="working_status").copy()
# df10["working_status"] = df10["working_status"].map(
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
# df10["working_status"] = df10["working_status"].astype("category")
# model10 = smf.ols(
#     formula=DV + " ~ C(working_status, Treatment(reference='Full time>40')) + age" +IV,
#     data=df10,
# ).fit()
# print(model10.summary())


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
        # model10,
        # model11,
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
        # "model10",
        # "model11",
    ]
)
stargazer.show_model_numbers(False)
stargazer.title("Regression Results")
stargazer.significance_levels([0.1, 0.05, 0.01])
html = stargazer.render_html()

out_name = "DV" + "-" + DV + "-regression_jp.html"
with open(out_name, "w", encoding="utf-8") as f:
    f.write(html)
