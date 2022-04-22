# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:39:08 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: 
    
"""
import pandas as pd
import pingouin as pg

#enter survey responses as a DataFrame
df = pd.DataFrame({'Q1': [1, 2, 2, 3, 2, 2, 3, 3, 2, 3],
                   'Q2': [1, 1, 1, 2, 3, 3, 2, 3, 3, 3],
                   'Q3': [1, 1, 2, 1, 2, 3, 3, 3, 2, 3]})

result = pg.cronbach_alpha(data = df)
print(result)