"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2026/3/11 17:39
Description:


"""

import torch.nn as nn


class DeepARLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
        distribution="negative-binomial",
    ):
        super(DeepARLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
