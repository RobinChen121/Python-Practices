# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:05:36 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: read local html file
    
"""

from bs4 import BeautifulSoup
import re
import requests




soup = BeautifulSoup(open('E:\\爬虫练习\\天眼查\\a.html', encoding='UTF-8'), 'lxml')
state_list = soup.find_all('div', class_ = 'tag-common -normal-bg')
boss_list = soup.find_all('div', class_ = 'title -wider text-ellipsis')
capital_list = soup.find_all('div', class_ = 'title -narrow text-ellipsis')
name_list = soup.find_all('div', class_ = 'logo -w88')
soup.select('.tag-common -normal-bg')

