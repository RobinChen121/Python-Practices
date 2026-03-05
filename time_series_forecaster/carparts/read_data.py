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


def z_score(x: torch.Tensor, dim=0, eps=1e-8):
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + eps)  # 避免除零


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
    scaled_raw_data = raw_data / raw_data.mean()

    # float() will transform the data to torch.float32 by default
    ages = torch.arange(0, month_num).unsqueeze(1).repeat(1, item_num).float()
    months = (
        torch.tensor([raw_data.index[i].month for i in range(month_num)])
        .unsqueeze(1)
        .repeat(item_num, 1)
        .float()
    )
    scaled_ages = z_score(ages)
    scaled_months = z_score(months)

    y_sequence = []
    x_sequence = []
    for j in range(item_num):
        # 获取该 item 的 embedding，必须用 tensor 索引访问
        item_emb = embedding_layers(torch.tensor(j))
        for i in range(month_num - 2 * sequence_length):
            y = torch.tensor(
                scaled_raw_data.iloc[i + sequence_length : i + 2 * sequence_length, j]
            )
            x = torch.tensor(scaled_raw_data.iloc[i : i + sequence_length, j])
            age = scaled_ages[i, j].flatten()  # 用 flatten() 才有维度
            month = scaled_months[i, j].flatten()

            covariate = torch.cat([age, month, item_emb], dim=0)  # 必须有维度才能拼接
            x_sequence.append(torch.cat([x, covariate], dim=0))
            y_sequence.append(y)
    return x_sequence, y_sequence


if __name__ == "__main__":
    df_ = get_raw_data()
    month_num, item_num = df_.shape
    sequence_length = 8
    embedding_dim = 32

    embedding_layers = nn.Embedding(item_num, embedding_dim)
    create_sequence(df_, sequence_length, embedding_layers)
    print(df_.head())
    print(df_.shape)
