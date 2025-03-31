"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/3/26 10:18
Description: 
    

"""

import pandas as pd

df = pd.read_excel(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/teaching/MG5580/attendence/2025/total.xlsx"
)
df["Checked in"] = df["Status"].apply(lambda x: 1 if "Attended" in x else 0)
df1 = df[
    [
        "First Name",
        "Surname",
        "Student Number",
        "Check-in time",
        "week",
        "lecture",
        "Checked in",
    ]
]
df2 = df1[["First Name", "Surname", "Student Number", "Checked in"]]
df3 = df2.groupby(["First Name", "Surname", "Student Number"]).sum().reset_index()
required_num = df3["Checked in"].max()
df3["attendance rate"] = df3["Checked in"] / required_num
df3["attendance rate"] = df3["attendance rate"].round(2)
group1 = [2406897, 2454912, 2436420, 2438218, 1943033, 2435784]
group2 = [2437658, 2450399, 2449619, 2443543, 2450549, 2502612]
group3 = [2501206, 2508923, 2354444, 2446520, 2510061]
group4 = [2164307, 2502152, 2149478, 2446815, 2438646, 2450391]
group6 = [2509187, 2433622, 2440245, 2505489, 2502998, 2504856]
group5 = [2303493, 2502848, 2339055, 2401087]
# group1 = list(map(str, group1))
# group2 = list(map(str, group2))
# group3 = list(map(str, group3))
# group4 = list(map(str, group4))
# group5 = list(map(str, group5))
# group6 = list(map(str, group6))
df3["group"] = df3["Student Number"].apply(
    lambda x: (
        1
        if x in group1
        else (
            2
            if x in group2
            else 3 if x in group3 else 4 if x in group4 else 5 if x in group5 else  6 if x in group6 else None
        )
    )
)
df3 = df3.sort_values(by = 'group')
df3.to_excel('/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/teaching/MG5580/attendence/2025/attendance_final.xlsx')
