# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:59:20 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: crawling the online novle "hong lou meng"
    
"""

import requests  # 联系网络的包，a package for requesting from websites
import re  # 正则表达式包，for cutting the punctuations
from bs4 import BeautifulSoup  # 分析网页数据的包，a package for webstie data analysis
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
    soup = BeautifulSoup(url_text, "lxml")  # 解析网页返回内容，lxml 是一个解码方式，效率比较快，被推荐使用
    # title = soup.find('h1').get_text() # 寻找便签 h1 中的内容，作为标题
    title = soup.title.text  # 也可以这样获取 title
    # title = re.sub(r'\s', '', title, count=2)  # 将前两个空格替换
    # 使用 get_text 可以读取网页里面的换行，而 text 不能
    chapter_content = soup.select('.chapter_content')[0].get_text(
        separator="\n")  # select(.类名) 查找网页中的类，因为返回的是列表，所以跟 [0]
    # chapter_content2 = soup.find_all('div', class_ = 'chapter_content')[0].text
    chapter_content = chapter_content.lstrip()  # 去除左边的空格
    chapter_content = chapter_content.rstrip()  # 去除右边的空格
    this_chapter = [title, chapter_content]
    return this_chapter


# 将每章内容输出到 txt 文档里
def write_txt(string_array):
    file_address = 'E:/爬虫练习/红楼梦/'  # txt 存放地址
    file_name = string_array[0]
    f = open(file_address + file_name + '.txt', 'w', encoding='utf-8')  # 必须跟解码形式，不然有的网页中文内容写不到txt里
    f.write(string_array[1])
    f.close()


# 主函数
def main():
    # 一共 120 回，每一回一个网页
    for i in range(120):
        try:
            chapter_num = str(i + 1)
            url = 'http://www.shicimingju.com/book/hongloumeng/' + chapter_num + '.html'  # 各回的网页网址
            url_content = get_url_content(url)  # 获取网页
            chapter_content = filter_info(url_content)  # 解析网页内容，提取小说
            write_txt(chapter_content)  # 输出小说内容到 txt
            time.sleep(random.random() * 2)  # 每抓一个网页休息0~2秒，防止被反爬措施封锁 IP
        except:  # 第 28 回缺失，跳过错误，继续抓
            continue


main()
