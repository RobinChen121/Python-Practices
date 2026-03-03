"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 25/11/2025, 19:04
@Desc    : this is my python coding to test the multi horizontal quantile regression
forecaster in the paper: arxiv-2017-a multi horizon quantile recurrent forcast.

"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config
import pytorch_lightning as pl


# 在绝大多数的时间序列预测场景中，时间划分是强制性的。
# 用户划分是可选的，取决于你是否需要评估模型对全新用户或实体的泛化能力
def read_data():
    data_name = "LD2011_2014.txt"
    if sys.platform == 'win32':
        data_path = os.path.join('D:/chenzhen/data/', data_name)
    else:
        data_path = os.path.join('/Users/zhenchen/Documents/machine learning data/', data_name)

    df_raw = pd.read_csv(data_path,
                         parse_dates=[0],  # specify data column index
                         delimiter=";",  # data separator
                         decimal=",")  # character that recognized as decimal
    df_raw.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)  # rename the name of dataframe by mapper
    # convert kw to kwh by taking mean()
    df_hour_agg = df_raw.resample("1h", on="timestamp").mean().reset_index()
    df_hour_agg.set_index("timestamp", inplace=True)

    # create calender features
    df_hour_agg["yearly_cycle"] = np.sin(2 * np.pi * df_hour_agg.index.dayofyear / 366)
    df_hour_agg["weekly_cycle"] = np.sin(2 * np.pi * df_hour_agg.index.dayofweek / 7)
    df_hour_agg["daily_cycle"] = np.sin(2 * np.pi * df_hour_agg.index.hour / 24)

    # split households to enable model generalization for predicting unknown households
    households = df_hour_agg.columns[1:]
    train_households, test_households = train_test_split(households, train_size=config.households_train_ratio)

    df_full_train = df_hour_agg[train_households]
    df_full_test = df_hour_agg[test_households]
    date_train_size = int(df_full_train.shape[0]*config.date_train_ratio)
    df_train = df_full_train[:date_train_size]
    df_val = df_full_train[date_train_size:]
    if not os.path.isfile("data_loder/test_data.csv"):
        df_full_test.to_csv("data_loder/test_data.csv", index=False)
    return df_train, df_val


if __name__ == '__main__':
    data_train, data_val = read_data()
