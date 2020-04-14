# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:52:08 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: crawling transaction datas of a tmall store
    
"""


#导入需要的库
import requests
from bs4 import  BeautifulSoup as bs
import json
import csv
import re
import time
import random

#宏变量存储目标js的URL列表，全局变量
COMMENT_PAGE_URL = []
ITEM_ID = 591120591607 # 商品代码

#生成链接列表
def Get_Url(num):
    # 对于不同商品，urlFront 的内容需要修改，注意翻页 num
    urlFront = 'https://rate.tmall.com/list_detail_rate.htm?itemId='
    urlMid = '&spuId=1190907782&sellerId=2455371463&order=3&currentPage='
    urlRear = '&append=0&content=1&tagId=&posi=&picture=&groupId=&ua=098%23E1hvbpvRvpWvUpCkvvvvvjiPn2dw0jEmP25ZtjYHPmPWlji8n25yljECRFSZzjnmRUhCvCB4caYOoY147DIhIJwwgDST75MNh46Cvvyv2bJmlrOvOSgPvpvhvv2MMgyCvvOUvvVvJhkivpvUvvmvK2bMWektvpvIvvvvvhCvvvvvvUHaphvUoQvvvQCvpvACvvv2vhCv2RvvvvWvphvWgvyCvhQvJE6vCzbEKOmD5dUf8zcGzWkfeciP1WLW%2B2Kz8Z0vQRAn%2BbyDCJLIAXZTKFEw9Exr08TJEcqpafk1%2BFw5ZTVQD70wd56Ofa1lDb8rvphvC9vhvvCvp2yCvvpvvvvviQhvCvvv9U8jvpvhvvpvv86Cvvyv2h4mSbQvhbArvpvEvvAnmWLhv8bGRphvCvvvvvv%3D&needFold=0&_ksTS=1586769049242_598&callback=jsonp599'
    for i in range(0,num):
        COMMENT_PAGE_URL.append(urlFront+str(ITEM_ID)+urlMid+str(1+i)+urlRear)

#获取单个商品的评论数据
def GetInfo(num):
    #定义需要的字段
    nickname = []
    auctionSku = []
    ratecontent = []
    rateDate = []
    #循环获取每一页评论
    for i in range(num):
        #头文件，没有头文件会返回错误的js
        headers = {
            'cookie':'=wE04FhTJ4DYCAd6yymobnvhk; hng=CN%7Czh-CN%7CCNY%7C156; lid=kkrobert; enc=c2o38MOsusbWbqQSfoOUPsj%2BByoX38ioNWr4jvjiV1U08cfjrU77O0vrt26HBaanb4LhsrxnT9S77VozIVoipg%3D%3D; sm4=500100; t=8182d0a22f3c08fc0ca729d34a939249; tracknick=kkrobert; _tb_token_=7fe676791eee5; cookie2=1f263acd1b1bb718051e3422ab2bc905; _m_h5_tk=fef1b48d562648fc8e433c520801f419_1586775572350; _m_h5_tk_enc=41a8cb772659eaa633f036ec205d9c66; x5sec=7b22726174656d616e616765723b32223a223238383036646361663737323538366661313635393236316130386165393863434f48593050514645502f516e704439696637327967453d227d; l=dBxWGoB4QcdQPvJMBOfZd8YREn79aIObzsPrrLelrICPOR1vqkaVWZXRWYLJCnGVnsiBR3oGfmN0B2Y_nyznhZXRFJXn9MpOLdTh.; isg=BOfnx94YdWop6fEJ_yEGhlTkdhuxbLtOQ8Ix8rlVUHb1qAVqwT4Ln5Duyqg2QJPG',
            'user-agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
            'referer': 'https://detail.tmall.com/item.htm?spm=a1z10.3-b-s.w4011-18708102967.41.1e3274683nIS7a&id=591120591607&rn=911d5e11064605ecc45b9726063b5a13&abbucket=15&skuId=4221626110476',
            'accept': '*/*',
            'accept-encoding':'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9'
        }
        #解析JS文件内容，用了正则表达式
        #url = 'https://rate.tmall.com/list_detail_rate.htm?itemId=591120591607&spuId=1190907782&sellerId=2455371463&order=3&currentPage=2&append=0&content=1&tagId=&posi=&picture=&groupId=&ua=098%23E1hvbpvRvpWvUpCkvvvvvjiPn2dw0jEmP25ZtjYHPmPWlji8n25yljECRFSZzjnmRUhCvCB4caYOoY147DIhIJwwgDST75MNh46Cvvyv2bJmlrOvOSgPvpvhvv2MMgyCvvOUvvVvJhkivpvUvvmvK2bMWektvpvIvvvvvhCvvvvvvUHaphvUoQvvvQCvpvACvvv2vhCv2RvvvvWvphvWgvyCvhQvJE6vCzbEKOmD5dUf8zcGzWkfeciP1WLW%2B2Kz8Z0vQRAn%2BbyDCJLIAXZTKFEw9Exr08TJEcqpafk1%2BFw5ZTVQD70wd56Ofa1lDb8rvphvC9vhvvCvp2yCvvpvvvvviQhvCvvv9U8jvpvhvvpvv86Cvvyv2h4mSbQvhbArvpvEvvAnmWLhv8bGRphvCvvvvvv%3D&needFold=0&_ksTS=1586769049242_598&callback=jsonp599'
        content = requests.get(COMMENT_PAGE_URL[i],headers=headers).text # COMMENT_PAGE_URL[i]
        nk = re.findall('"displayUserNick":"(.*?)"', content) # 评论者账号
        # 如果网页为空，表示评论数没那么多，则跳过循环
        if len(nk) == 0:
            break
        else:
            time.sleep(random.random())  # 每抓一个网页休息0-1秒，防止被反爬措施封锁 IP
        nickname.extend(nk) # extend 添加一串数
        #print(nk)
        auctionSku.extend(re.findall('"auctionSku":"(.*?)"', content)) # 产品型号
        ratecontent.extend(re.findall('"rateContent":"(.*?)"', content))
        rateDate.extend(re.findall('"rateDate":"(.*?)"', content))
    #将数据写入csv文件中
    address = 'E:\爬虫练习\天猫评论\商品' + str(ITEM_ID) + '.csv'
    f = open(address, 'a+', encoding='utf-8-sig')
    # 首先读取列标题
    f.write(','.join(('nickname', 'rateDate', 'auctionSku', 'ratecontent')) +'\n') # 用逗号分隔，csv读取时视作不同的单元格
    for i in list(range(0, len(nickname))):
        text = ','.join((nickname[i], rateDate[i], auctionSku[i], ratecontent[i])) + '\n'
        f.write(text + ' ')
        print(i+1,":写入成功")
    f.close()

#主函数
if __name__ == "__main__":
    Page_Num = 30  # 最大支持 Page_Num*20 个评论
    Get_Url(Page_Num)
    GetInfo(Page_Num)