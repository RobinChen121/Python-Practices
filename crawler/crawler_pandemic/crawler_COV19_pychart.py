# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:45:02 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyecharts.charts import Pie
from pyecharts import options as opts

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
df.sort_values(by = 'dead', ascending = False, inplace = True)
df.reset_index(drop = True, inplace = True)
df = df.iloc[0:15, 0:15]


a = df['name'].values.tolist()
b = df['dead'].values.tolist()
n = len(a)
rosechart = Pie(init_opts = opts.InitOpts(width='1350px', height='750px'))

color_series = np.random.random((n, 3))
# rosechart.set_colors(color_series)
# Add data, set the radius of pie chart, and show it as Nightingale chart or not
rosechart.add("", [list(z) for z in zip(a, b)],
        radius=["5%", "95%"],
        center=["30%", "60%"],
        rosetype="area"
        )
# Set global configuration item
rosechart.set_global_opts(title_opts=opts.TitleOpts(title='新冠肺炎各国累计死亡数 '+ str(df['date'][0])),
                     legend_opts=opts.LegendOpts(is_show=False),
                     toolbox_opts=opts.ToolboxOpts())

# Set the serial configuration item to be labeled outside
rosechart.set_series_opts(label_opts=opts.LabelOpts(is_show=True, 
                                               position="outside",  # inside, outside
                                               font_size=14,
                                               formatter="{b}:{c}", font_style="italic",
                                               font_weight="bold", font_family="Microsoft YaHei"
                                               ),
                     )
rosechart.render('a.html')
rosechart.render_notebook()