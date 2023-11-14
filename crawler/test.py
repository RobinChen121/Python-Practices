# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 19:21:22 2021

@author: zhen chen

MIT Licence.

Python version: 3.8


Description: 
    
"""
import requests
from bs4 import BeautifulSoup

r = requests.get('http://www.baidu.com/')  #requests.get('https://so.gushiwen.cn/guwen/book_46653FD803893E4F7F702BCF1F7CCE17.aspx') 
r.encoding='utf-8'
text = r.text
soup = BeautifulSoup(r.text, features="lxml")
contents = soup.find_all('div', class_='contson')
poems = []
for i in range(len(contents)):
    str = contents[i].get_text()
    poems.append(str)
