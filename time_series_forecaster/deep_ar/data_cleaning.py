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
import numpy as np


def z_score(x: torch.Tensor, dim=0, eps=1e-8):
    # dim=0 表示沿第 0 维求平均，0维表示行，即对列求平均
    # keepdim=true 表示保留维度，不然会把那个指定的dim删除
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


# embedding 应该放到模型的 forward 函数里
def create_sequence(
    raw_data: pandas.DataFrame,
    # embedding_layers: torch.nn.Embedding,
    train_length: int,
    encoder_length: int,
    decoder_length: int,
):
    month_num, item_num = raw_data.shape

    # scaling
    # scaled_raw_data = (raw_data / raw_data.mean()).astype("float32")

    # float() will transform the data to torch.float32 by default
    ages = torch.arange(0, month_num).unsqueeze(1).float()
    m = torch.tensor([raw_data.index[i].month for i in range(month_num)]).float()

    # scaled_ages = z_score(ages) # 到底是否需要对age, month 进行z标准化，存疑
    # scaled_months = z_score(months)

    scaled_ages = ages / (month_num - 1)
    scaled_months = (torch.sin(2 * torch.pi * m), torch.cos(2 * torch.pi * m))

    y_train = []
    x_train = []
    y_test = []
    x_test = []
    v_train = []
    v_test = []
    emb_train = []
    emb_test = []
    data_np = raw_data.to_numpy(dtype=np.float32)  # 比频繁调用 pandas iloc 快多倍
    for j in range(item_num):
        # 获取该 item 的 embedding，必须用 tensor 索引访问
        # device = embedding_layers.weight.device
        # item_emb = embedding_layers(torch.tensor([j], device=device)).detach()
        # torch.long 是 int64 整数型
        item_id = torch.tensor(j, dtype=torch.long)
        for i in range(train_length - encoder_length - decoder_length + 1):
            # from_numpy 比 torch.tensor 快，浅拷贝，但是必须让numpy数组首先可写
            v = 1 + np.mean(data_np[i : i + encoder_length, j])
            # torch.full 与 repeat 功能类似，只不过作用于单个值
            v_window = torch.full((encoder_length, 1), float(v))
            v_train.append(v_window)
            x = torch.tensor(data_np[i : i + encoder_length, j] / v).unsqueeze(1)
            y = torch.tensor(
                data_np[i + encoder_length : i + encoder_length + decoder_length, j]
            ).unsqueeze(
                1
            )  # y 要拟合负二项分布，为整数，不需要标准化

            # emb = item_emb.repeat(encoder_length, 1)

            # dim=0 按行拼接，dim=1 按列拼接
            x_train.append(torch.cat([x, age, month], dim=1))  # 必须有维度才能拼接
            y_train.append(y)
            emb_train.append(item_id)

        v = 1 + np.mean(data_np[train_length - encoder_length : train_length, j])
        v_window = torch.full((encoder_length, 1), float(v), dtype=torch.float32)
        v_test.append(v_window)
        x = (
            torch.from_numpy(
                data_np[train_length - encoder_length : train_length, j] / v
            )
            .float()
            .unsqueeze(1)
        )
        age = scaled_ages[train_length - encoder_length : train_length, j].unsqueeze(
            1
        )  # 用 flatten() 才有维度
        month = scaled_months[
            train_length - encoder_length : train_length, j
        ].unsqueeze(1)
        # emb = item_emb.repeat(encoder_length, 1)
        x_test.append(torch.cat([x, age, month], dim=1))

        y = torch.from_numpy(data_np[train_length:month_num, j]).float().unsqueeze(1)
        y_test.append(y)
        emb_test.append(item_id)
    return (
        # pytorch 中，第0维为行，即数据最外面（最左边）的维度
        # stack 在指定 dim上堆叠，默认 dim = 0
        torch.stack(x_train),
        torch.stack(x_test),
        torch.stack(y_train),
        torch.stack(y_test),
        torch.stack(v_train),
        torch.stack(v_test),
        torch.stack(emb_train),
        torch.stack(emb_test),
    )


if __name__ == "__main__":
    df_ = get_raw_data()
    month_num_, item_num_ = df_.shape
    sequence_length_ = 8
    embedding_dim = 32

    # embedding_layers_ = nn.Embedding(item_num_, embedding_dim)
    x_train_, y_train_, x_test_, y_test_, _, _ = create_sequence(
        df_, train_length=43, encoder_length=8, decoder_length=8
    )
    print(df_.head())
    print(df_.shape)
