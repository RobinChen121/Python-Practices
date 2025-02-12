# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:34:22 2019

@author: zhen chen

Python version: 3.7

Description: adopt pyechart to draw maps. Note that the latest package has different
syntaxes with older ones， 使用 snapshot-selenium 渲染图片比较麻烦
    
"""

from pyecharts.charts import Bar
from pyecharts.charts import Map
from example.commons import Faker
from pyecharts import options as opts


bar = Bar()
bar.add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
bar.add_yaxis("商家1", [5, 20, 36, 10, 75, 90])

bar.set_global_opts(title_opts={"text": "主标题", "subtext": "副标题"})


# render 会生成本地 HTML 文件，默认会在当前目录生成 render.html 文件
# 也可以传入路径参数，如 bar.render("mycharts.html")
bar.render()


# 世界地图数据
#value = [95.1, 23.2, 43.3, 66.4, 88.5]
#attr= ["China", "Canada", "Brazil", "Russia", "United States"]
map = Map()
map.add('中国地图', [list(z) for z in zip(Faker.provinces, Faker.values())], "china")
map.render('china_map.html')

map2 = Map()

map2.add('中国地图', [list(z) for z in zip(Faker.country, Faker.values())], "world")
map2.set_series_opts(label_opts=opts.LabelOpts(is_show=False)) # not show each country's name
map2.set_global_opts(
            title_opts=opts.TitleOpts(title="Map-世界地图"),
            visualmap_opts=opts.VisualMapOpts(max_=200),
        )
map2.render('world_map.html')