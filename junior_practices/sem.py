"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/6/27 20:28
Description: 
    

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from semopy import Model

# ======================
# 1. 模拟数据生成
# ======================
np.random.seed(123)
n = 300  # 样本量

# 潜变量真实得分（用于生成题项）
latent_ad = np.random.normal(5, 1, n)  # 广告吸引力潜变量
latent_trust = 0.6 * latent_ad + np.random.normal(0, 1, n)  # 品牌信任（受广告影响）
latent_purchase = 0.7 * latent_trust + 0.3 * latent_ad + np.random.normal(0, 1, n)  # 购买意愿

# 生成观测题项（带测量误差）
data = pd.DataFrame({
    # 广告吸引力题项（3个）
    'ad1': latent_ad + np.random.normal(0, 0.5, n),
    'ad2': latent_ad + np.random.normal(0, 0.5, n),
    'ad3': latent_ad + np.random.normal(0, 0.5, n),

    # 品牌信任题项（3个）
    'trust1': latent_trust + np.random.normal(0, 0.5, n),
    'trust2': latent_trust + np.random.normal(0, 0.5, n),
    'trust3': latent_trust + np.random.normal(0, 0.5, n),

    # 购买意愿题项（2个）
    'purchase1': latent_purchase + np.random.normal(0, 0.5, n),
    'purchase2': latent_purchase + np.random.normal(0, 0.5, n),

    # 调节变量（个人创新性，0=低，1=高）
    'group': np.random.binomial(1, 0.5, n)  # 随机分组
})

# ======================
# 2. 定义SEM模型
# ======================
model_spec = '''
    # 测量模型（CFA部分）
    广告吸引力 =~ ad1 + ad2 + ad3
    品牌信任 =~ trust1 + trust2 + trust3
    购买意愿 =~ purchase1 + purchase2

    # 结构模型（路径分析）
    品牌信任 ~ 广告吸引力
    购买意愿 ~ 品牌信任 + 广告吸引力
'''

# ======================
# 3. 模型拟合与结果
# ======================
model = Model(model_spec)
result = model.fit(data)
print("=== 模型拟合结果 ===")
print(result)
print("\n=== 标准化系数 ===")
print(model.inspect(std_est=True))