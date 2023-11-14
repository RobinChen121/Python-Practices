# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:59:17 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: crawl comments data from a jd store: https://mall.jd.com/view_page-92839543.html
    
"""

import requests
import re
import time
import random
import math
from bs4 import BeautifulSoup
import datetime

# 宏变量存储目标js的URL列表，全局变量
COMMENT_PAGE_URL = []
ITEM_ID = 4571301  # 商品代码 4676059 为钛度毁灭者RGB鼠标， 4571301 为黑色磨砂鼠标
ITEM_ID = 100003986945 # 商品代码 100003986945 为暗鸦之眼3.5usb 耳机
#ITEM_ID = 100004023595 #暗鸦 M3.3M 带麦版本黑色


# 生成链接列表
def get_Url(num):
    #productPageComments
    urlFront = 'https://club.jd.com/comment/skuProductPageComments.action?callback=fetchJSON_comment98&productId='
    urlMid = '&score=0&sortType=6&page='
    urlRear = '&pageSize=10&isShadowSku=0&rid=0&fold=1'
    for i in range(0, num):
        COMMENT_PAGE_URL.append(urlFront + str(ITEM_ID) + urlMid + str(i) + urlRear)


# 获取单个商品的评论数据
def getInfo(num):
    # 定义需要的字段
    auctionSku = []  # 产品型号
    score = []
    rateContent = []
    rateDate = []

    # 头文件，没有头文件会返回错误的js
    headers = {
         'cookie':'shshshfpa=7dd23290-b0c4-77f9-c0ba-8323f0eda385-1572009534; shshshfpb=s3E8V9JSA50vxJEFgiWICuw%3D%3D; pinId=yk7rerKxt2oO0EHGPEYlDA; areaId=4; pin=robert_chen1988; unick=robinChen123; _tp=H%2BAfxttNz9M%2FYJBc7tmOuQ%3D%3D; _pst=robert_chen1988; ipLocation=%u91cd%u5e86; __jdu=1861334361; ipLoc-djd=4-48203-54798-0.1895914109; user-key=165e3e25-25e8-4eda-9e5f-43ba00ffbd8a; jwotest_product=99; cn=2; PCSYCityID=CN_500000_500100_500109; unpl=V2_ZzNsbRUFQhYmXEIHfR5YBmJRQFVKVEYQdFpOXCkZCAFgBhdeXhYtF3UPQWR4GF8EewIXQUNeUhZ0A0ZccBpcNWczElxDVkERcQlHUUsaXQNvBhczQFdEEkU4RlZLGGwFbgQSX0BSQRJ1OBEGehoPUWNQDltFU0AJcFoUXGcRXwBiHxMPSl8RFSEMQVF%2BGmwFYAU%3D; __jdv=122270672|news.ifeng.com|t_1000355004_207944_3166|tuiguang|fc02be5b76524cc992440c99c1e56442-p_3166|1586348361945; mt_xid=V2_52007VwMbVV1aUF8dThlsATQAFgdcWFVGHR4bCRliAEJWQVBbCR1VSlQAMgNBV1oMWloYeRpdBmYfE1dBWVtLHEgSXA1sABdiX2hRahxIH1QAYjMRUF5b; __jdc=122270672; 3AB9D23F7A4B3C9B=RYYF55B6RCE6A4LEX4H3C7NJMTQD6LEVJWHYJXTHAUH4VYRCFJCR7THNFZYR5ZPTI4IGZM6LQOYQOL2NKZCAWMRUU4; cid=NWxFNTEwNnBWNDk2N2RINDk4OGpTMjUyMWZOMjE0MndXNzM4M2pGMzkwNHBZMzM4; wlfstk_smdl=2f5dbnogl63w0ru37rt3l12rq632heqj; TrackID=1vWoN9s9x5cyZRaVdCnRfMvVgWp1ov3MOXravmuBnWrU4qhI-J7WQAe7uIxPkVcueQS3LoY3BkZ_q01q4JbJn3kH8b2e0TQFiOOl5uMJn57oY8iheQzPV6VTfaNu8FBvK; ceshi3.com=201; __jda=122270672.1861334361.1572009533.1586501762.1586505075.197; thor=4C6CC15539A1E7CDD5AD240701CE814E0887E17BA257A6BA450ADF9CABEBE1B045A4AC12AD1BEC52520DD8C585F429ABA7AC83A0060CFAF28B795E01A4F62A52B45A39749DEEF09F88FEDB1BF86F4B1EA509FD72FA1C2C6B5984D2D171DA70731AEEFE5FD80226916A92FEF947F8EC80B4B5036CB0F6D9D667444F875A73DB00833F22F0CCA4568EBAB39AFE77C64815F90E53447EAB34CB41C12240149571AC; shshshfp=8e8ee8fc1a3555a2c4c7ba3bc3b56a2e; JSESSIONID=59047F177BA263B4D60B646A11E3FC64.s1; shshshsID=7622a3e36e22d181c860fd948c53b62f_3_1586505186517; __jdb=122270672.3.1861334361|197.1586505075',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
        # 'referer': 'https://item.jd.com/4571299.html',
        #'accept': '*/*',
        #'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'zh-CN,zh;q=0.9'
    }

    # 循环获取每一页评论
    for i in range(num):
        # 解析JS文件内容，用了正则表达式
        url = COMMENT_PAGE_URL[i]
        response = requests.get(url, headers=headers)  # ,
        response.encoding = 'gbk'  # 防止抓取的是乱码
        content = response.text

        # soup = BeautifulSoup(content, "lxml") # soup 适合解析 html 网页，不适合解析纯文本的 json 网页

        _, _, comment_content = content.partition('"comments":')
        # 用正则表达式搜索所需的信息
        results = re.findall('"content":"(.*?)".*?"creationTime":"(.*?)".*?"score":(\d).*?"productColor":"(.*?)"',
                             comment_content, re.S)  # 评论者账号, re.S 表示可以换行查找
        # 如果网页为空，表示评论数没那么多，则跳过循环
        if len(results) == 0:
            break
        else:
            time.sleep(random.random())  # 每抓一个网页休息0-1秒，防止被反爬措施封锁 IP
            if i % 10 == 0:
                time.sleep(1)  # 每抓 10页休息两秒
        for result in results:
            rateContent.append(result[0])
            rateDate.append(result[1])
            score.append(result[2])
            auctionSku.append(result[3])

    # 将数据写入csv文件中
    address = 'E:\爬虫练习\京东评论\商品' + str(ITEM_ID) + '-' + str(datetime.date.today()) + '.csv'
    f = open(address, 'a+', encoding='utf-8-sig')
    # 首先读取列标题
    f.write(','.join(['score', 'rateDate', 'auctionSku', 'rateContent']) + '\n')  # 用逗号分隔，csv读取时视作不同的单元格
    for i in list(range(len(score))):
        text = ','.join((score[i], rateDate[i], auctionSku[i], rateContent[i])) + '\n'
        f.write(text + ' ')
        #print(i + 1, ":写入成功")
    f.close()


tic = time.process_time()
print(tic)
Page_Num = 2500  # 最大支持 Page_Num*10 个评论
get_Url(Page_Num)
getInfo(Page_Num)
print(time.process_time())
toc = time.process_time() - tic
print(f"time used: {toc}s")
