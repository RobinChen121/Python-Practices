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
    raw_data: pandas.DataFrame,
    embedding_layers: torch.nn.Embedding,
    train_length: int,
    encoder_length: int,
    decoder_length: int,
):
    month_num, item_num = raw_data.shape

    # scaling
    scaled_raw_data = (raw_data / raw_data.mean()).astype("float32")

    # float() will transform the data to torch.float32 by default
    ages = torch.arange(0, month_num).unsqueeze(1).repeat(1, item_num).float()
    months = (
        torch.tensor([raw_data.index[i].month for i in range(month_num)])
        .unsqueeze(1)
        .repeat(1, item_num)
        .float()
    )
    scaled_ages = z_score(ages)
    scaled_months = z_score(months)

    y_train = []
    x_train = []
    y_test = []
    x_test = []
    for j in range(item_num):
        # 获取该 item 的 embedding，必须用 tensor 索引访问
        item_emb = embedding_layers(torch.tensor(j))
        for i in range(train_length - encoder_length - decoder_length + 1):
            y = torch.tensor(
                scaled_raw_data.iloc[
                    i + encoder_length : i + encoder_length + decoder_length, j
                ].values
            ).unsqueeze(1)
            x = torch.tensor(
                scaled_raw_data.iloc[i : i + encoder_length, j].values
            ).unsqueeze(1)
            age = scaled_ages[i : i + encoder_length, j].unsqueeze(
                1
            )  # 用 flatten() 才有维度
            month = scaled_months[i : i + encoder_length, j].unsqueeze(1)
            emb = item_emb.repeat(encoder_length, 1)

            # dim=0 按行拼接，dim=1 按列拼接
            x_train.append(torch.cat([x, age, month, emb], dim=1))  # 必须有维度才能拼接
            y_train.append(y)

        x = torch.tensor(
            scaled_raw_data.iloc[
                train_length - encoder_length : train_length,
                j,
            ].values
        ).unsqueeze(1)
        age = scaled_ages[train_length - encoder_length : train_length, j].unsqueeze(
            1
        )  # 用 flatten() 才有维度
        month = scaled_months[
            train_length - encoder_length : train_length, j
        ].unsqueeze(1)
        emb = item_emb.repeat(encoder_length, 1)
        x_test.append(torch.cat([x, age, month, emb], dim=1))

        y = torch.tensor(
            scaled_raw_data.iloc[train_length:month_num, j].values
        ).unsqueeze(1)
        y_test.append(y)
    return torch.stack(x_train), torch.stack(y_train), torch.stack(x_test), torch.stack(y_test)


if __name__ == "__main__":
    df_ = get_raw_data()
    month_num_, item_num_ = df_.shape
    sequence_length_ = 8
    embedding_dim = 32

    embedding_layers_ = nn.Embedding(item_num_, embedding_dim)
    x_train_, y_train_, x_test_, y_test_ = create_sequence(
        df_, embedding_layers_, train_length=43, encoder_length=8, decoder_length=8
    )
    print(df_.head())
    print(df_.shape)
