# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 23:27:14 2021

@author: zhen chen

MIT Licence.

Python version: 3.8


Description: 
    
"""
import requests
from bs4 import BeautifulSoup
import re
import json

def getKeywordResult(keyword):
    url = 'http://www.baidu.com/s?wd='+keyword
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = 'utf-8'
        return r.text
    except:
        return ""
def parserLinks(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for div in soup.find_all('div', {'data-tools': re.compile('title')}): # 字典类型定义特殊的标签
        data = div.attrs['data-tools']  #获得属性值
        d = json.loads(data)        #将属性值转换成字典
        links.append(d['title'])    #将返回链接的题目返回
    return links
def main():
    html = getKeywordResult('Python语言程序设计基础(第2版)')
    ls = parserLinks(html)
    count = 1
    for i in ls:
        print("[{:^3}]{}".format(count, i))
        count += 1
        
main()