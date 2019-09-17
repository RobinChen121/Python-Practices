#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/9/15 19:59
# @Author  : Zhen Chen

# Python version: 3.7
# Description: 用 pandas 创建一个 DataFrame 类型

"""

import pandas as pd
import numpy as np

df = pd.DataFrame(np.array([[85, 68, 90], [82, 63, 88], [84, 90, 78]]), columns=['统计学', '高数', '英语'], index=['张三', '李四', '王五'])
print(df)
print(df.sample(2))