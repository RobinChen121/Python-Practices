"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/11/7 15:13
Description: 
    

"""
import torch
import torch.nn as nn

# 假设词汇表大小为10000，embedding维度为300
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=300)

# 输入一个词的索引
word_id = torch.tensor([42])
vector = embedding(word_id)  # shape: [1, 300]
print(vector)
