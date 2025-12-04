"""
@Python version: 3.12
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 03/12/2025, 17:02
@Desc:

"""

import pandas as pd
import os
import sys


def read_data():
    if sys.platform != "win32":
        file_path = "/Users/zhenchen/Documents/machine learning data/walmart-recruiting-store-sales-forecasting"
    else:
        file_path = "D:/chenzhen/data/walmart-recruiting-store-sales-forecasting/"
    # 读取数据
    sales = pd.read_csv(os.path.join(file_path, "train.csv"), parse_dates=["Date"])
    features = pd.read_csv(
        os.path.join(file_path, "features.csv"), parse_dates=["Date"]
    )
    stores = pd.read_csv(os.path.join(file_path, "stores.csv"))

    # 合并数据
    df = pd.merge(sales, features, on=["Store", "Date", "IsHoliday"])
    df = pd.merge(df, stores, on="Store")

    # 按时间排序
    df = df.sort_values(["Store", "Dept", "Date"])

    return df


if __name__ == "__main__":
    data = read_data()
    pass
