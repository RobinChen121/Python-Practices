# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:08:23 2019

@author: zhen chen

MIT Licence.

Python version: 3.7

Email: robinchen@swu.edu.cn

Description: This a practice for crawler in the movie reviewing website douban, 
                crawling for top250 movies, and analyze their result
    
"""
import requests  # 联系网络的包，a package for requesting from websites
import xlwt  # 读写 excel 的包，a package for reading and writing in excel, not supporting xlsx writing
from bs4 import BeautifulSoup # 分析网页数据的包，a package for webstie data analysis
from collections import Counter # 计算列表中元素的包，counter the num of each element in a list
import collections
import matplotlib.pyplot as plt # 画图的包
from pylab import mpl  # 设置图形中字体样式与大小的包
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['font.size'] = 6.0

import time 
import random

import jieba # 中文分词包
from wordcloud import WordCloud # 词云包
import re  # 正则表达式包，for cutting the punctuations


headers = {
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36',
    'Host':'movie.douban.com'
    }

## data needed
movie_list_english_name = []
movie_list_chinese_name = []
director_list = []
time_list = []
star_list = []
reviewNum_list = []
quote_list = []
nation_list = []
category_list = []

num = 0
for i in range(0, 10):
    link = 'https://movie.douban.com/top250?start=' + str(i*25) # 250部电影一共 10个网页， 10 pages for total 250 movies    
    res = requests.get(link, headers = headers, timeout = 10)
    time.sleep(random.random()*3) # 每抓一个网页休息2~3秒，防止被反爬措施封锁 IP，avoid being blocked of the IP address
    
    # res.text is the content of the crawler
    soup = BeautifulSoup(res.text, "lxml")  # lxml 是一个解码方式，lxml is one decoding model for Beautifulsoup
    div_title_list = soup.find_all('div', class_ = 'hd') # 寻找 hd 类型的类，find classes whose tag are hd
    div_info_list = soup.find_all('div', class_ = 'bd')
    div_star_list = soup.find_all('div', class_ = 'star')
    div_quote_list = soup.find_all('p', class_ = 'quote')
    
    for each in div_title_list:
        # a表示 html 中的超链接，a is href link of html
        # strip 去掉收尾的空格，strip() is for stripping spacing at the beginning and end of a string
        movie = each.a.span.text.strip() # 只能得到第一个字段，only get the first span of text this method
        movie_list_chinese_name.append(movie)
        
    # 通过css 定位得到第二个字段，从而得到英文名字，get second span by css location   
    div_title_list2 = soup.select('div.hd > a > span:nth-of-type(2)')
    for each in div_title_list2:
        movie = each.text
        #movie = movie.replace(u'\xa0', u' ')
        movie = movie.strip('\xa0/\xa0') # 去掉英文名字中的空格，strip the extra string in the english name
        movie_list_english_name.append(movie)
    
    for each in div_info_list:
        num += 1
        info = each.p.text.strip()
        if len(info) < 3: # 筛选掉不符合条件的信息，skip the information not needed
            continue
        
        # 搜索电影上映年代，find the movie year
        lines = info.split('\n')  # 将信息按照换行符分割成不同句子，split the info into two lines
        time_start = lines[1].find('20')
        if time_start < 0:
            time_start = lines[1].find('19')
        time_len = lines[1][time_start : time_start + 4]
        time_list.append(time_len)
        time_end = time_start + 4
        
        # find the director English name. some director name string strange, so drop this
#        for i in range(len(info)):
#            if info[i].encode( 'UTF-8' ).isalpha():
#                break
#        if i != len(info) - 1:
#            start = i
#            end = info.find('主')
#            director = info[start : end - 3]
#            director_list_english_name.append(director)
        
        # 搜索电影导演中文名，find the director name
        end = info.find('主')
        if end < 0:
            end = info.find('...')
        director = info[4 : end - 3]
        director_list.append(director)
        
        # 搜索电影来源地，find the nation of the movie
        frequent = 0
        start = 0
        end = 0
        line2 = lines[1]
        for j in range(len(line2)):
            if line2[j] == '\xa0':
                frequent += 1
            if frequent == 2 and start == 0:
                start = j + 1
            if frequent == 3:
                end = j
                break
        nation = line2[start : end]
        nation_list.append(nation)
        
        
        # 搜索电影类型，find the category of the movie
        frequent = 0
        start = 0
        for j in range(len(line2)):
            if line2[j] == '\xa0':
                frequent += 1
            if frequent == 4 and start == 0:
                start = j + 1
        category = line2[start : len(line2)]
        category_list.append(category)
    
    # 搜索电影评分，find the star of each movie    
    for each in div_star_list:
        info = each.text.strip()
        star = float(info[0 : 3])
        star_list.append(star)
        end = info.find('人')
        reviewNum = int(info[3 : end])
        reviewNum_list.append(reviewNum)
    
    # 搜索电影代表评论，find the best quote for each movie
    for each in div_quote_list:
        info = each.text.strip()
        quote_list.append(info)
    if len(quote_list) == 249: # 第250部电影没有代表性评论，单独处理。the 250th movie has no quote, so add a blank one
        quote_list.append(' ') 
    
file = xlwt.Workbook()

table = file.add_sheet('sheet1', cell_overwrite_ok = True)

table.write( 0, 0, "排名")
table.write( 0, 1, "电影中文名")
table.write( 0, 2, "电影其他名")
table.write( 0, 3, "时间")
table.write( 0, 4, "导演")
table.write( 0, 5, "国家或地区")
table.write( 0, 6, "评分")
table.write( 0, 7, "评分人数")
table.write( 0, 8, "电影类型")
table.write( 0, 9, '代表性评论')

for i in range(len(nation_list)):
    table.write(i + 1, 0, i + 1)
    table.write(i + 1, 1, movie_list_chinese_name[i])
    table.write(i + 1, 2, movie_list_english_name[i])
    table.write(i + 1, 3, time_list[i])
    table.write(i + 1, 4, director_list[i])
    table.write(i + 1, 5, nation_list[i])
    table.write(i + 1, 6, star_list[i])
    table.write(i + 1, 7, reviewNum_list[i])
    table.write(i + 1, 8, category_list[i])
    table.write(i + 1, 9, quote_list[i])

 # 导出到 xls 文件里，save to xls file    
file.save('豆瓣 top 250 电影爬虫抓取.xls')

# 分析电影来源地，analysis nations
locations = []
for i in range(len(nation_list)):
    nations = nation_list[i].split(' ') 
    for j in range(len(nations)):
        if nations[j] == '西德':
            nations[j] = '德国'
        locations.append(nations[j])

result = Counter(locations)
result_sort = sorted(result.items(), key = lambda x: x[1], reverse = True) # order descending and by x[1]
result_sort = collections.OrderedDict(result_sort)
othervalue = 0
for i in range(10, len(result)):
    othervalue += list(result_sort.values())[i]
    
# 画饼状图，draw the pie picture using matplotlib
def make_autopct(values): # 定义饼状图中数字显示方式， define the values formats in the pie
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%({v:d})'.format(p = pct, v = val)
    return my_autopct
    
values = []
labels = []
for i in range(10):
    values.append(list(result_sort.values())[i])
    labels.append(list(result_sort.keys())[i])
values.append(othervalue)
labels.append('其他地区')
plt.rcParams['savefig.dpi'] = 200 # 定义图形清晰度，set dpi for figure, affect the figure's size
plt.rcParams['figure.dpi'] = 200 #set dpi for figure
w, l, p  = plt.pie(values, explode = [0.02 for i in range(11)], labels = labels, pctdistance = 0.8, 
                           radius = 1, rotatelabels = True, autopct = make_autopct(values))
[t.set_rotation(315) for t in p] # 设置标签旋转，rotate the text for the labels
plt.title('豆瓣 TOP250 电影来源地', y = -0.1)
plt.show()

# 分析电影类型，analysis categories
categories = []
for i in range(len(category_list)):
    category = category_list[i].split(' ') 
    for j in range(len(category)):
        categories.append(category[j])
result = Counter(categories)
result_sort = sorted(result.items(), key = lambda x: x[1], reverse = True) #排序 order descending and by x[1]
result_sort = collections.OrderedDict(result_sort)
othervalue = 0
for i in range(15, len(result)):
    othervalue += list(result_sort.values())[i]
# draw the pie picture using matplotlib
values = []
labels = []
for i in range(15):
    values.append(list(result_sort.values())[i])
    labels.append(list(result_sort.keys())[i])
values.append(othervalue)
labels.append('其他类型')
plt.rcParams['savefig.dpi'] = 200 # 定义图形清晰度，set dpi for figure, affect the figure's size
plt.rcParams['figure.dpi'] = 200 #set dpi for figure
w, l, p  = plt.pie(values, explode = [0.02 for i in range(16)], labels = labels, pctdistance = 0.8, 
                           radius = 1, rotatelabels = True, autopct = make_autopct(values))
[t.set_rotation(315) for t in p] # rotate the text for the labels
plt.title('豆瓣 TOP250 电影种类', y = -0.1)
plt.show()


# word cloud
jieba.add_word('久石让')
jieba.add_word('谢耳朵')
# 一些语气词和没有意义的词
del_words = ['的', ' ', '人', '就是', '一个', '被', 
             '不是', '也', '最', '了', '才', '给', '要', 
             '就', '让', '在', '都', '是', '与', '和', 
             '不', '有', '我', '你', '能', '每个',  '不会', '中', '没有',
             '这样', '那么', '不要', '如果', '来', '它', '对', '当', '比',
             '不能', '却', '一种', '而', '不过', '只有', '不得不', '再',
             '不得不', '比', '一部', '啦', '他', '像', '会', '得', '里']
all_quotes = ''.join(quote_list) # 将所有代表性评论拼接为一个文本
# 去掉标点符号
all_quotes = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ", all_quotes)
words = jieba.lcut(all_quotes)
words_final = []
for i in range(len(words)): # 去掉一些语气词，没有意义的词。 
    if words[i] not in del_words:
        words_final.append(words[i])
text_result = Counter(words_final)
cloud = WordCloud(
    font_path = 'FZSTK.TTF',
    background_color = 'white',
    width = 1000,
    height = 860,
    max_words = 40   
 )

#wc = cloud.generate(words) # 这种方法对中文支持不太好，this mehtod is better for only english string
wc = cloud.generate_from_frequencies(text_result)
wc.to_file("豆瓣 TOP 250 词云.jpg") 
plt.figure()
plt.imshow(wc)
plt.axis('off')
plt.title('豆瓣 TOP 250 电影代表性评论的词云分析')
plt.show()

# 评分最高的十部电影
star_dict = dict(zip(movie_list_chinese_name, star_list))
star_sort = sorted(star_dict.items(), key = lambda x: x[1], reverse = True) #排序 order descending and by x[1]
star_sort = collections.OrderedDict(star_sort)
values = []
labels=[]
for i in range(10):
    labels.append(list(star_sort.keys())[i])
    values.append(list(star_sort.values())[i])
bar = plt.barh(range(10), width = values, tick_label = labels, color = 'rgbycmrgby')
for i, v in enumerate(values): # 柱状图添加数字
    plt.text(v + 0.05, i - 0.1, str(v), color = 'blue', fontweight = 'bold')
plt.xlim(xmax = 10, xmin = 8)
plt.title('评分最高的十部电影')
plt.show()

# 评分人数最多的十部电影
review_dict = dict(zip(movie_list_chinese_name, reviewNum_list))
review_sort = sorted(review_dict.items(), key = lambda x: x[1], reverse = True) #排序 order descending and by x[1]
review_sort = collections.OrderedDict(review_sort)
values = []
labels=[]
for i in range(10):
    labels.append(list(review_sort.keys())[i])
    values.append(list(review_sort.values())[i])
bar = plt.barh(range(10), width = values, tick_label = labels, color = 'rgbycmrgby')
for i, v in enumerate(values): # 柱状图添加数字
    plt.text(v + 10000, i - 0.1, str(v), color = 'blue', fontweight = 'bold')
plt.xlim(xmax = 1450000, xmin = 400000)
plt.title('评分人数最多的十部电影')
plt.show()