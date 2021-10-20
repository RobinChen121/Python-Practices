# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 16:13:23 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

fo = open('test.txt', 'w+', encoding='utf-8')
ls = ['唐诗\n', '宋词\n', '元曲\n']
fo.writelines(ls)
fo.close()