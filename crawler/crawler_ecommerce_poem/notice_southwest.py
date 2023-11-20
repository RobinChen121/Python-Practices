# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 08:18:23 2023

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.11


Description: should use F12 to find where the notices website of Southwest university really locates
    
"""
import requests  
from bs4 import BeautifulSoup
import re


r = requests.get('https://coa.swu.edu.cn/api/customDoc/v1/getDocList?sg8l3mmA=n2MOnqlqEhT6s0d9jBy6NEFA5bwjVYYBi5x2KMleaaR4kHvV4lH1AcH8BDZuDFfQ5aeSxsjW6Dt8HwIP8TsVCvM2TS.PGd682I4shZ8KhP4qqDGG0WghzSkn3Wrld.3CtcouTlWEFG3')
soup = BeautifulSoup(r.text, 'lxml')
docids = re.compile('(?<="docid":")\d{5}').findall(soup.text)
dates = re.compile('(?<="doccreatedate":").*?(?=","docsubject")').findall(soup.text)
dates2 = re.compile('(?<="doccreatedate":")\d{4}-\d{2}-\d{2}').findall(soup.text) # same with above 
titles = re.compile('(?<=docsubject":").*?(?="})').findall(soup.text)


r2 = requests.get('https://coa.swu.edu.cn/api/customDoc/v1/getDocDetail?sg8l3mmA=68wmQalqEoQhg_F_rKlnr3GGO69hQsJohI2jOQBT7tLQKYSrWzowWhAYGkWZGhN8y1r7Q2H5PpQoWE9tQBNqZ3YV5D2MdTFo1F68PBcjQhwX1fhqnBt6QG')
soup2 = BeautifulSoup(r2.text, 'lxml')