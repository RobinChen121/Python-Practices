# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 20:00:51 2021

@author: zhen chen

MIT Licence.

Python version: 3.8


Description: 
    
"""
import requests  # 联系网络的包，a package for requesting from websites
from bs4 import BeautifulSoup  # 分析网页数据的包，a package for webstie data analysis
import time
import random
import pandas as pd


# 获取单个网页信息
def get_url_content(url):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36'
    }  # 主要是定义浏览器登录，防止反爬，一般的网站没有这个也行
    r = requests.get(url, headers=headers, timeout=30)  # 获取网页的内容，并返回给r变量，timeout 为超时时间
    r.raise_for_status()  # 检查返回网页的内容的状态码，200表示成功
    r.encoding = r.apparent_encoding  # 统一编码方式
    return r.text  # 返回网页中的文本内容，数据内容为 r.content


# 解析出网页中想要的信息，爬虫的关键步骤
def filter_info(chapter_num, url_text):
    soup = BeautifulSoup(url_text, "lxml")  # 解析网页返回内容，lxml 是一个解码方式，效率比较快，被推荐使用
    contents = soup.find('div', class_='contson')
    # 使用 get_text 可以读取网页里面的换行，而 text 不能
    chapter_title = contents.find('p').get_text()  # 第一句话是本回的标题
    chapter_titel = '第' + str(chapter_num) + '回 ' + chapter_title.lstrip()
    chapter_content = contents.get_text(separator="\n")  # find 返回的不是列表，不用跟 [0]
    chapter_content = chapter_content.lstrip() # 截掉字符串左边的空字符
    this_chapter = [chapter_titel, chapter_content]
    return this_chapter

# 从网页中找到每一回的链接
def bs_single_text(url_head_content):
    soup = BeautifulSoup(url_head_content, "lxml")  # 解析网页返回内容，lxml 是一个解码方式，效率比较快，被推荐使用
    items = soup.find('div', class_='list-box')  # 每一回的链接都在类 span 里面
    items = items.findAll('li')
    for each in items:
        date = each.span.get_text()
        title = each.a.get_text()
        all_list.append([date, title])  # 获取每一回的链接，存储到列表里


# 将每章内容输出到 txt 文档里
def write_txt(string_array):
    file_address = 'E:/爬虫练习/三国演义/'  # txt 存放地址
    file_name = string_array[0]
    with open(file_address + file_name + '.txt', 'w', encoding='utf-8') as f: # 必须跟解码形式，不然有的网页中文内容写不到txt里
        f.write(string_array[1])

# 主函数
def main():
    for i in range(1, 30):
        str1 = 'http://tempnews.swu.edu.cn/seeyon/xndxNewsData.do?method=getNewsMoreList&curPage='
        str2 = '&typeId=26851391992318884pageSize=35'
        url = str1 + str(i) + str2
        single_text = get_url_content(url)
        bs_single_text(single_text)
    datas = pd.DataFrame(all_list)
    datas.to_excel(r'D:\西大新闻.xlsx', encoding = 'utf-8')
    print()
        
    # url_head_content = get_url_content(url)  # 获取网页
    # links = get_url_links(url_head_content)  # 获取每一回的链接地址
    # for index, each in enumerate(links):
    #     url_link_content = get_url_content(each)  # 获取每一回的网页内容
    #     chapter_content = filter_info(index + 1, url_link_content)  # 解析每一回的网页内容，获取小说文本
    #     write_txt(chapter_content)  # 输出小说内容到 txt
    #     time.sleep(random.random() * 1)  # 每抓一个网页休息0~1秒，防止被反爬措施封锁 IP

# 运行函数
all_list = []
main()


