# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:32:42 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description:  crawling company data from tianyancha，construction companes in Beibei District, Chonqing 
    
"""

import re
import requests
from bs4 import BeautifulSoup
import time
import random

# 获取网页信息
def get_url_content(url):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36'
    }  # 主要是模拟浏览器登录，防止反爬
    r = requests.get(url, headers=headers, timeout=30)  # 获取网页的内容，并返回给r变量，timeout 为超时时间
    r.raise_for_status()  # 检查返回网页的内容的状态码，200表示成功
    r.encoding = r.apparent_encoding  # 识别编码
    return r.text  # 返回网页中的文本内容，数据内容为 r.content


# 解析出网页中想要的信息，爬虫的关键步骤
def filter_info(url_text):
    content = 
    soup = BeautifulSoup(url_text, "lxml")  # lxml 是一个解码方式，lxml is one decoding model for Beautifulsoup
    search_results = soup.find_all(name='div', class_ = 'search-item sv-search-company') # 查询的显示结果
    name_list = re.findall('alt="(.*?)"', search_resultssearch_results)
    # title = re.sub(r'\s', '', title, count=2)  # 将前两个空格替换
    # 使用 get_text 可以读取网页里面的换行，而 text 不能
    chapter_content = soup.select('.chapter_content')[0].get_text(
        separator="\n")  # select(.类名) 查找网页中的类，因为返回的是列表，所以跟 [0]
    # chapter_content2 = soup.find_all('div', class_ = 'chapter_content')[0].text
    chapter_content = chapter_content.lstrip()  # 去除左边的空格
    chapter_content = chapter_content.rstrip()  # 去除右边的空格
    this_chapter = [title, chapter_content]
    return this_chapter


# 主函数
def main():
    # 一共只显示 250 页搜索结果
    for i in range(250):
        try:
            page_num = str(i + 1)
            url = 'https://www.tianyancha.com/search/ocE/p' + page_num + '?key=%E5%8C%97%E7%A2%9A&companyType=normal_company&base=cq&areaCode=500109'  # 各网页网址
            url_content = get_url_content(url)  # 获取网页
            page_content = filter_info(url_content)  # 解析网页内容
            time.sleep(random.random() * 2)  # 每抓一个网页休息0~2秒，防止被反爬措施封锁 IP
        except:  # 第 28 回缺失，跳过错误，继续抓
            continue


main()