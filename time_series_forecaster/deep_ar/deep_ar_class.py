"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2026/3/11 17:39
Description:


"""

import torch.nn as nn
import torch
import torch.distributions as Dist


class DeepARLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        layer_num=1,
        batch_first=True,
        bidirectional=False,
        distribution="negative-binomial",
    ):
        super(DeepARLSTM, self).__init__()
        self.output_size = output_size
        self.dist = distribution

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            layer_num,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        # mu or sigma
        # mu or alpha
        # 每个时间步 h_t 形状 [num_layers, batch, hidden_size]
        # 用每个时间步最后那个 layer 的h_t 参与下面的映射
        self.linear = nn.Linear(hidden_size, 2 * output_size)

    def forward(self, x):
        # 返回所有时间步的隐藏状态及最后一个时间步的（h_n, c_m）
        # out 的形状是 (batch_size, seq_size, hidden_size)
        out, _ = self.lstm(x)  # out.shape = [batch, seq_len, 2*output_size]
        parameters = self.linear(out)
        mu = parameters[..., : self.output_size]  # tensor 的切片, 不论前面的维度是什么
        sigma_or_alpha = parameters[..., self.output_size :]
        if self.dist == "negative-binomial":
            mu = (
                torch.nn.functional.softplus(mu) + 1e-6
            )  # 加上 1e-6 是防止出现 0 的情况
        sigma_or_alpha = torch.nn.functional.softplus(sigma_or_alpha) + 1e-6

        if self.dist != "negative-binomial":
            dist = Dist.normal(mu, sigma_or_alpha)
        else:
            prob = mu / sigma_or_alpha
            num = mu * prob / (1 - prob)
            dist = Dist.negative_binomial(num, prob)

        return dist
