"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2026/3/11 17:39
Description:


"""

import torch.nn as nn
import torch
import torch.distributions as distribution


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

    def forward(self, x, v):
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

        # ---- 原始尺度还原 ----
        if v is not None:
            mu = mu * v
            if self.dist == "negative-binomial":
                # 负二项 alpha 不乘 v，要除以 sqrt(v)
                sigma_or_alpha = sigma_or_alpha / torch.sqrt(v)

        if self.dist != "negative-binomial":
            dist = distribution.normal.Normal(mu, sigma_or_alpha)
        else:
            # 负二项分布用 shape parameter 和均值拟合
            var = mu + sigma_or_alpha * (mu**2)
            p = mu / var
            p = torch.clamp(p, min=1e-6, max=1.0 - 1e-6)  # 让数值稳定，防止极端值
            r = mu * p / (1 - p)
            dist = distribution.negative_binomial.NegativeBinomial(
                total_count=r, probs=p
            )

        return dist
