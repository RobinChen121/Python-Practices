# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:20:47 2023

@author: chen
"""
import requests
from bs4 import BeautifulSoup

res = requests.get('https://so.gushiwen.cn/shiwenv_987daf622b7e.aspx')
res.encoding = 'utf-8'
soup = BeautifulSoup(res.text, 'lxml')
poem = soup.find('textarea').get_text()
print(poem)