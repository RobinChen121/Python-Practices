"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 26/11/2025, 20:38
@Desc    : electricity data set

"""
import torch
from torch.utils.data import Dataset

class ElDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self, df):
        return len(self.df)

    def __getitem__(self, idx):
        """ Yield one sample"""
        row = self.df.iloc[idx]