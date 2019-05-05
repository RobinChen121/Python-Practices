# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:08:23 2019

@author: zhen chen

Python version: 3.7

Description: This a practice for crawler in the movie reviewing website Maoyan
    
"""
import requests  # a package for requesting from websites
import xlswt  # a package for reading and writing in excel
from bs4 import BeautifulSoup # a package for webstie data analysis


headers = {
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36',
    'Host':'movie.douban.com'
    }
