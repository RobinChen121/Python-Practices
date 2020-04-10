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
    urlRear = '&append=0&content=1&tagId=&posi=&picture=&groupId=&ua=098%23E1hvHQvRvpQvUpCkvvvvvjiPRLqp0jlbn2q96jD2PmPWsjn2RL5wQjnhn2cysjnhR86CvC8h98KKXvvveSQDj60x0foAKqytvpvhvvCvp86Cvvyv9PPQt9vvHI4rvpvEvUmkIb%2BvvvRCiQhvCvvvpZptvpvhvvCvpUyCvvOCvhE20WAivpvUvvCC8n5y6J0tvpvIvvCvpvvvvvvvvhZLvvvvtQvvBBWvvUhvvvCHhQvvv7QvvhZLvvvCfvyCvhAC03yXjNpfVE%2BffCuYiLUpVE6Fp%2B0xhCeOjLEc6aZtn1mAVAdZaXTAdXQaWg03%2B2e3rABCCahZ%2Bu0OJooy%2Bb8reEyaUExreEKKD5HavphvC9vhphvvvvGCvvpvvPMM3QhvCvmvphmCvpvZzPQvcrfNznswOiaftlSwvnQ%2B7e9%3D&needFold=0&_ksTS=1552466697082_2019&callback=jsonp2020'
    for i in range(0,num):
        COMMENT_PAGE_URL.append(urlFront+str(ITEM_ID)+urlMid+str(1+i)+urlRear)

#获取单个商品的评论数据
def GetInfo(num):
    #定义需要的字段
    nickname = []
    auctionSku = []
    ratecontent = []
    ratedate = []
    #循环获取每一页评论
    for i in range(num):
        #头文件，没有头文件会返回错误的js
        headers = {
            'cookie':'cna=qMU/EQh0JGoCAW5QEUJ1/zZm; enc=DUb9Egln3%2Fi4NrDfzfMsGHcMim6HWdN%2Bb4ljtnJs6MOO3H3xZsVcAs0nFao0I2uau%2FbmB031ZJRvrul7DmICSw%3D%3D; lid=%E5%90%91%E6%97%A5%E8%91%B5%E7%9B%9B%E5%BC%80%E7%9A%84%E5%A4%8F%E5%A4%A9941020; otherx=e%3D1%26p%3D*%26s%3D0%26c%3D0%26f%3D0%26g%3D0%26t%3D0; hng=CN%7Czh-CN%7CCNY%7C156; x=__ll%3D-1%26_ato%3D0; t=2c579f9538646ca269e2128bced5672a; _m_h5_tk=86d64a702eea3035e5d5a6024012bd40_1551170172203; _m_h5_tk_enc=c10fd504aded0dc94f111b0e77781314; uc1=cookie16=V32FPkk%2FxXMk5UvIbNtImtMfJQ%3D%3D&cookie21=U%2BGCWk%2F7p4mBoUyS4E9C&cookie15=UtASsssmOIJ0bQ%3D%3D&existShop=false&pas=0&cookie14=UoTZ5bI3949Xhg%3D%3D&tag=8&lng=zh_CN; uc3=vt3=F8dByEzZ1MVSremcx%2BQ%3D&id2=UNcPuUTqrGd03w%3D%3D&nk2=F5RAQ19thpZO8A%3D%3D&lg2=U%2BGCWk%2F75gdr5Q%3D%3D; tracknick=tb51552614; _l_g_=Ug%3D%3D; ck1=""; unb=3778730506; lgc=tb51552614; cookie1=UUBZRT7oNe6%2BVDtyYKPVM4xfPcfYgF87KLfWMNP70Sc%3D; login=true; cookie17=UNcPuUTqrGd03w%3D%3D; cookie2=1843a4afaaa91d93ab0ab37c3b769be9; _nk_=tb51552614; uss=""; csg=b1ecc171; skt=503cb41f4134d19c; _tb_token_=e13935353f76e; x5sec=7b22726174656d616e616765723b32223a22393031623565643538663331616465613937336130636238633935313935363043493362302b4d46454e76646c7243692b34364c54426f4d4d7a63334f44637a4d4455774e6a7378227d; l=bBIHrB-nvFBuM0pFBOCNVQhjb_QOSIRYjuSJco3Wi_5Bp1T1Zv7OlzBs4e96Vj5R_xYB4KzBhYe9-etui; isg=BDY2WCV-dvURoAZdBw3uwj0Oh2yUQwE5YzQQ9qAfIpm149Z9COfKoZwV-_8q0HKp',
            'user-agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
            'referer': 'https://detail.tmall.com/item.htm?spm=a1z10.5-b-s.w4011-17205939323.51.30156440Aer569&id=41212119204&rn=06f66c024f3726f8520bb678398053d8&abbucket=19&on_comment=1&sku_properties=134942334:3226348',
            'accept': '*/*',
            'accept-encoding':'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9'
        }
        #解析JS文件内容，用了正则表达式
        content = requests.get(COMMENT_PAGE_URL[i],headers=headers).text
        nk = re.findall('"displayUserNick":"(.*?)"', content)
        # 如果网页为空，表示评论数没那么多，则跳过循环
        if len(nk) == 0:
            break
        else:
            time.sleep(random.random())  # 每抓一个网页休息0-1秒，防止被反爬措施封锁 IP
        nickname.extend(nk) # extend 添加一串数
        #print(nk)
        auctionSku.extend(re.findall('"auctionSku":"(.*?)"', content))
        ratecontent.extend(re.findall('"rateContent":"(.*?)"', content))
        ratedate.extend(re.findall('"rateDate":"(.*?)"', content))
    #将数据写入csv文件中
    address = 'E:\爬虫练习\天猫评论\商品' + str(ITEM_ID) + '.csv'
    f = open(address, 'a+', encoding='utf-8-sig')
    # 首先读取列标题
    f.write(','.join(('nickname', 'ratedate', 'auctionSku', 'ratecontent')) +'\n') # 用逗号分隔，csv读取时视作不同的单元格
    for i in list(range(0, len(nickname))):
        text = ','.join((nickname[i], ratedate[i], auctionSku[i], ratecontent[i])) + '\n'
        f.write(text + ' ')
        print(i+1,":写入成功")
    f.close()

#主函数
if __name__ == "__main__":
    Page_Num = 30  # 最大支持 30*20 个评论
    Get_Url(Page_Num)
    GetInfo(Page_Num)