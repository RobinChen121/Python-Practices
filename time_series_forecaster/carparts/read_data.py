"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 01/03/2026, 13:07
@Desc:

"""

import os
import pandas
import pyreadr
import pandas as pd
import torch.nn as nn
import torch


def get_raw_data():
    file_path = os.path.dirname(os.getcwd())
    file_path = os.path.join(file_path, "data", "carparts.rda")

    # result is an ordered dict object
    result = pyreadr.read_r(file_path)
    df = result[list(result.keys())[0]]

    # cull the data based on some conditions:
    # (a) Possessed fewer than ten positive monthly demands
    # (b) Had no positive demand in the first 15 and final 15 months
    # sum() is calculating the sum of conditional counts
    bad_count = (df > 0).sum()
    df1 = df.loc[:, bad_count >= 10]

    # any() is about any one that satisfies the condition
    df2 = df1.loc[:, (df1.iloc[:15] > 0).any() & (df1.iloc[-15:] > 0).any()]
    df2.index = pd.period_range("1998-01", periods=51, freq="M")
    return df2


def create_sequence(
    raw_data: pandas.DataFrame, sequence_length: int, embedding_layers=None
):
    month_num, item_num = raw_data.shape

    # scaling
    raw_data = raw_data / raw_data.mean()

    y_sequence = []
    x_sequence = []
    for j in range(item_num):
        for i in range(month_num - 2 * sequence_length):
            y = raw_data.iloc[i + sequence_length : i + 2 * sequence_length, j].values
            x = raw_data.iloc[i : i + sequence_length, j].values
            age = i
            month = raw_data.index[i].month

            covariate = [age, month, embedding_layers[j]]
            x_sequence.append(covariate)
            y_sequence.append(y)
    return torch.tensor(y_sequence, dtype=torch.float32), torch.tensor(
        x_sequence, dtype=torch.float32
    )


if __name__ == "__main__":
    df_ = get_raw_data()
    month_num, item_num = df_.shape
    sequence_length = 8
    embedding_dim = 32
    create_sequence(df_, sequence_length)
    embedding_layers = nn.Embedding(item_num, embedding_dim)
    print(df_.head())
    print(df_.shape)
