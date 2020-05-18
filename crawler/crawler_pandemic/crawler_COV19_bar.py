# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:29:55 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: crawl the datas from qq news about COV-19
    
"""

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'https://api.inews.qq.com/newsqa/v1/automation/foreign/country/ranklist'
headers = {
         'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
}

res = requests.get(url, headers = headers, timeout = 10)
#res.encoding = 'gbk'  # 防止抓取的是乱码
content = res.text
data_jason = json.loads(content)
data = data_jason['data']

df = pd.DataFrame(data)
df = df[['name', 'continent', 'date', 'confirm', 'nowConfirm', 'confirmAdd', 'dead', 'heal']]
df.loc[len(df)] = ['中国', '亚洲', '05.15', 84471, 167, 7, 4644, 79660]
df.sort_values(by = 'confirmAdd', ascending = False, inplace = True)
df.reset_index(drop = True, inplace = True)
df = df.iloc[0:10, 0:10]

#数据计算
radius = df['confirmAdd']
n = radius.count()
theta = np.arange(0, 2*np.pi, 2*np.pi/n)    # 360度分成 n 份，偏移：+ 2*np.pi/(2*n)
radius_maxmin = (radius-radius.min())/(radius.max()-radius.min())  #x-min/max-min   归一化 

# 画图
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(20,20))
ax = plt.subplot(111, projection='polar')    #一行一列第一个，启用极坐标
ax.set_theta_direction(-1) # 逆时针
ax.set_theta_zero_location('N') # 正上方为零度

bar = ax.bar(theta, 
             radius,
             width = 2*np.pi/n,
             color = np.random.random((n, 3)), # 随机颜色
             align = 'edge', # 从指定位置
             #bottom = 5  # 偏离圆心距离
           )

ax.set_title('新冠肺炎各国新增确诊数 ' + str(df['date'][0]), fontdict={'fontsize':15})   #设置标题

#设置文字
for i in range(n):
    ax.text(theta[i] + 0.03, radius[i] + 10, df['name'][i] + ':' + str(radius[i]))


plt.axis('off') # 不显示网络线
plt.tight_layout # 紧凑布局
plt.show()
plt.savefig('test.png')
