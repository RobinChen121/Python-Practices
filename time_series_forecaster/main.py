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
from sklearn.model_selection import train_test_split


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

    # split households to enable model generalization for predicting unknown households
    households = df_hour_agg.columns[1:]
    train_households, test_households = train_test_split(households, test_size=0.5)

    df_train = df_hour_agg[train_households]
    df_test = df_hour_agg[test_households]

    pass


if __name__ == '__main__':
    read_data()
