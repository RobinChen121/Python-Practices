# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:21:36 2019

@author: zhen chen

Python version: 3.7

Description: 
    
"""

from example.commons import Faker
from pyecharts import options as opts
from pyecharts.charts import Map



def map_base() -> Map:
    c = (
        Map()
        .add("商家A", [list(z) for z in zip(Faker.provinces, Faker.values())], "china")
        .set_global_opts(title_opts=opts.TitleOpts(title="Map-基本示例"),
                         visualmap_opts=opts.VisualMapOpts(max_=200, is_piecewise=True), #加颜色
        )
    )
    return c

map_base().render('example.html')


districts = ['北碚区', '江北区', '沙坪坝区']
values = [30, 50, 40]
def map_chongqing() -> Map:
    c = (
        Map(init_opts = opts.InitOpts(height = '500px', width = '900px'))
        .add("商家A", [list(z) for z in zip(districts, values)], "重庆")
        .set_global_opts(     
            title_opts = opts.TitleOpts(title="Map-重庆地图"),
            visualmap_opts = opts.VisualMapOpts(max_=200, is_piecewise=True),  
           # datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
        )     
    )
    return c

map_chongqing().render('chongqing.html')