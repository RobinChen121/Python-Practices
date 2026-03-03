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
file_name = "JP_linear_regression_data.csv"
data_address = os.path.join(folder_address, file_name)

df_raw = pd.read_csv(data_address)

df1 = df_raw.dropna(subset="used_ai").copy()
df1["used_ai"] = df1["used_ai"].map({1.0: "No", 2.0: "Yes"})
df1["used_ai"] = df1["used_ai"].astype("category")
model1 = smf.ols(
    formula="trust_ai_6 ~ C(used_ai) + trust_ai_3 + mean_ai", data=df1
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
    formula="trust_ai_6 ~ C(marital_status) + trust_ai_3 + mean_ai", data=df2
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
    formula="trust_ai_6 ~ C(working_status) + trust_ai_3 + mean_ai", data=df3
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
    formula="trust_ai_6 ~ C(children_in_HH) + trust_ai_3 + mean_ai", data=df4
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
    formula="trust_ai_6 ~ C(household_size) + trust_ai_3 + mean_ai", data=df5
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
    formula="trust_ai_6 ~ C(urban_rural) + trust_ai_3 + mean_ai", data=df6
).fit()
print(model6.summary())

stargazer = Stargazer([model1, model2, model3, model4, model5, model6])
stargazer.custom_columns(["model1", "model2", "model3", "model4", "model5", "model6"])
stargazer.show_model_numbers(False)
stargazer.title("Regression Results")
stargazer.significance_levels([0.1, 0.05, 0.01])
html = stargazer.render_html()

with open("regression.html", "w", encoding="utf-8") as f:
    f.write(html)
