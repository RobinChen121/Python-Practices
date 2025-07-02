"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/7/2 20:15
Description: 
    

"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd

df = pd.read_excel(r'/Users/zhenchen/Downloads/git/Statistics-book/datas/data-pca.xlsx', index_col=0)  # 读取数据
data = scale(df.values)  # 标准化，标准化之后就自动根据协方差矩阵进行主成分分析了
pca = PCA(n_components=3)  # the number of principal components
pca.fit(data)  # fit the data by PCA
print(
    f"explained variance of each principal component: {pca.explained_variance_ratio_}"
)  # output the explained variance ratio

principal_components = pca.fit_transform(data)
print(f"principal components shown the first 10 rows:\n {principal_components[0:10]}")
