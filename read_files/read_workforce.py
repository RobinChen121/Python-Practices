"""
Python version: 3.13
Author: Zhen Chen
Date: 2026/9/8
Description:
    Read the workforce testing CSV file and report confidence intervals for
    gap and runtime metrics by parameter group.
"""

import pandas as pd


FILE_PATH = (
    "/Users/zhenchen/Library/CloudStorage/"
    "OneDrive-BrunelUniversityLondon/Numerical-tests/workforce/c++/"
    "12periods_testing_2segments.csv"
)

GROUP_COLUMNS = [
    "turnover pattern",
    "min worker",
    "fix cost",
    "salary",
    "penalty",
]

METRIC_COLUMNS = ["SDPtime", "opt gap%", " MIP time", "MIP sS gap%", " MIP-sS time"]

Z_VALUE_95 = 1.96


def confidence_interval(series: pd.Series) -> pd.Series:
    clean_series = series.dropna()
    sample_size = len(clean_series)
    mean_value = clean_series.mean()
    std_value = clean_series.std(ddof=1) if sample_size > 1 else 0.0
    margin = Z_VALUE_95 * std_value / (sample_size ** 0.5) if sample_size > 1 else 0.0
    return pd.Series(
        {
            "mean": mean_value,
            "margin": margin,
        }
    )


df = pd.read_csv(FILE_PATH)

for group_column in GROUP_COLUMNS:
    summary = (
        df.groupby(group_column)[METRIC_COLUMNS]
        .apply(lambda group: group.apply(confidence_interval))
        .unstack()
        .swaplevel(axis=1)
        .sort_index(axis=1, level=0)
    )
    print(f"\n===== Grouped by {group_column.strip()} =====")
    header = [
        group_column.strip(),
        "SDPtime", "opt gap%", " MIP time", "MIP sS gap%", " MIP-sS time"
    ]
    print(" & ".join(header) + r" \\")
    for group_value in summary.index:
        formatted_metrics = [str(group_value)]
        for metric_column in METRIC_COLUMNS:
            mean_text = f"{summary.loc[group_value, ('mean', metric_column)]:.2f}"
            margin_text = f"{summary.loc[group_value, ('margin', metric_column)]:.2f}"
            formatted_metrics.append(f"{mean_text} $\\pm$ {margin_text}")
        print(" & ".join(formatted_metrics) + " & 270" + r" \\")

overall_summary = df[METRIC_COLUMNS].apply(confidence_interval)
print("\n===== Overall =====")
print("overall & SDPtime & opt gap% & MIP time & sS gap% & MIP-sS time " + r"\\")
overall_metrics = ["overall"]
for metric_column in METRIC_COLUMNS:
    mean_text = f"{overall_summary.loc['mean', metric_column]:.2f}"
    margin_text = f"{overall_summary.loc['margin', metric_column]:.2f}"
    overall_metrics.append(f"{mean_text} $\\pm$ {margin_text}")
print(" & ".join(overall_metrics) + " & 1080" + r" \\")
