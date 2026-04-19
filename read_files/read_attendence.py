"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 15/04/2026, 12:09
@Desc:

"""

import pandas as pd
import os
import sys

file_dir = ""
if sys.platform == "win32":
    file_dir = "C:/Users/Administrator/OneDrive - Brunel University London/teaching/MG5580/attendence/2026/"
file_list = os.listdir(file_dir)
df_list = []
for file in file_list:
    if "lecture" in file or "seminar" in file:
        df = pd.read_csv(file_dir + file)
        df = df[["Student Number", "Status", "Check-in time", "Check In Mode"]]
        week_num = file[4]
        lecture_num = 1 if "lecture" in file else 0
        df["Check-in time"] = pd.to_datetime(df["Check-in time"], format="%H:%M")
        df["week"] = week_num
        df["lecture"] = lecture_num
        df["attended"] = (df["Status"] == "Attended").astype(int)
        if "lecture" in file:
            df["not_late"] = (df["Check-in time"].dt.hour <= 9).astype(int)
        else:
            df["not_late"] = (df["Check-in time"].dt.hour <= 10).astype(int)
        df_list.append(df)

df_total = pd.concat(df_list, ignore_index=True)
df_total.to_csv(file_dir + "total.csv", index=False)
df_attendance = df_total[["Student Number", "attended", "not_late"]]
df_attendance = df_attendance.groupby(["Student Number"]).sum()
df_attendance.to_csv(file_dir + "attendance_rate.csv")
df_group = pd.read_excel(file_dir + "groups-2026.xlsx")
df_final = df_group.merge(df_attendance, on="Student Number", how="left")
df_final.to_csv(file_dir + "final.csv", index=False)
