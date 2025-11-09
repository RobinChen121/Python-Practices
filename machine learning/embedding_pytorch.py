"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 08/11/2025, 17:13
@Desc    : 

"""
import torch
import torch.nn as nn

# 假设：
# season 有 4 种类别（春=0, 夏=1, 秋=2, 冬=3）
# weekday 有 7 种类别（周一=0, 周二=1, ... 周日=6）
season_embedding = nn.Embedding(num_embeddings=4, embedding_dim=2)  # 4类 → 2维向量
weekday_embedding = nn.Embedding(num_embeddings=7, embedding_dim=3) # 7类 → 3维向量

# 举例：今天是“夏季”(1)，星期三(2)
season_id = torch.tensor([1])
weekday_id = torch.tensor([2])

# 获取 embedding 向量
season_vec = season_embedding(season_id)
weekday_vec = weekday_embedding(weekday_id)

print("Season embedding:", season_vec)
print("Weekday embedding:", weekday_vec)

# 拼接成最终时间特征向量
# dim=-1 表示“最后一个维度"，等价于按列拼接 dim = 1
time_feature = torch.cat([season_vec, weekday_vec], dim= -1)
print("Combined time feature:", time_feature)
