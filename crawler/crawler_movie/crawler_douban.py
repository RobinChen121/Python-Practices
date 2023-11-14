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
import pandas as pd
from bs4 import BeautifulSoup # 分析网页数据的包，a package for webstie data analysis
import matplotlib.pyplot as plt # 画图的包
import time 
import random
import jieba # 中文分词包
from wordcloud import WordCloud # 词云包
import re  # 正则表达式包
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei'] 

headers = {
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36',
    'Host':'movie.douban.com'
    }

## store the information
rank_list = []
name_list = [] # 用来存储电影名字的一个二维列表
director_list = [] # 用来存储导演名字的一个列表，下面几个变量类似
actor_list = []
year_list = []
rating_list = []
ratingNum_list = []
quote_list = []
place_list = []
category_list = []

num = 0
for i in range(0, 10):
    link = 'https://movie.douban.com/top250?start=' + str(i*25) # 250部电影一共 10个网页， 10 pages for total 250 movies    
    res = requests.get(link, headers = headers, timeout = 10)
    time.sleep(random.random()*3) # 每抓一个网页休息2~3秒，防止被反爬措施封锁 IP，avoid being blocked of the IP address
    
    # res.text is the content of the crawler
    soup = BeautifulSoup(res.text, "lxml")  # lxml 是一个解码方式，lxml is one decoding model for Beautifulsoup
    movie_list = soup.find_all('div', class_ = 'item') # 检索所有类型为 item 的 div 标签
    
    for each in movie_list:
        # 排名在 em 标签里
        rank = each.em.text
        rank_list.append(rank)
        
        # 电影名字在标签 <span class="title"> 与 <span class="other"> 里
        other_name = each.find('span', class_ = 'other').get_text()
        other_name = other_name.replace(' ', '') # 去掉空格
        other_name = other_name.replace('\xa0/\xa0', '') # 去掉多余字符 \xa0/\xa0
        cn_name = each.find('span', class_ = 'title').get_text()
        eg_name = each.find('span', class_ = 'title').find_next().get_text() # find_next() 查找满足条件的下一个标签
        eg_name = eg_name.replace('\xa0/\xa0', '') # 去掉多余字符 \xa0/\xa0
        name_list.append([cn_name, eg_name, other_name])   
              
        # 评分信息在标签  <span class="rating_num"> 里
        rating =  each.find('span', class_ = 'rating_num').get_text()
        rating_list.append(float(rating))
        
        # 评价人数通过关键词'评价'检索
        rating_num = each.find(string = re.compile('评价')).get_text()
        rating_num = rating_num.replace('人评价', '')
        ratingNum_list.append(int(rating_num))
        
        # 代表性评价在  <p class="quote"> 里  
        try:
            quote =  each.find('p', class_ = 'quote').get_text()
        except Exception: # 有的电影代表性评价没有显示
            quote =  ''
        quote_list.append(quote)
        
        info = each.p.get_text(strip = True)
        # 定义正则表达式提取出导演，主演，上映时间，地点，电影类型信息        
        try: 
            # (?<=导演: ) match any character begins after character '导演: '
            # .*? match any character (.), zero or more times (*) but as less as possible (?)
            # (?=主) match any character before character '主'
            director = re.compile('(?<=导演: ).*?(?=主)').findall(info)[0]
            actor = re.compile('(?<=主演: ).*?(?=/)').findall(info)[0]
        except Exception: # 有的电影导演名字太长，主演没有显示出来
            director = re.compile('(?<=导演: ).*?(?=\xa0)').findall(info)[0]
            actor = ''  
        director_list.append(director)
        actor_list.append(actor)
        
        # \d{4} is a four digit number
        year = re.compile('(?<=...)\d{4}').findall(info)[0]
        year_list.append(year)
        place_category = re.compile('(?<=\xa0/\xa0).*').findall(info)[0]
        place_category = place_category.replace('\xa0', '')
        place_category = place_category.split('/')
        place = place_category[0]
        category = place_category[1]
        place_list.append(place)
        category_list.append(category)
          
# 将数据存到 pandas 里
df = pd.DataFrame(rank_list, columns = ['排名'])    
df['电影名字'] = [i[0] for i in name_list]
df['外文名字'] = [i[1] for i in name_list]
df['其他名字'] = [i[2] for i in name_list]
df['评分'] = rating_list
df['评价人数'] = ratingNum_list
df['导演'] = director_list
df['主演'] = actor_list
df['上映日期'] = year_list
df['地区'] = place_list
df['类型'] = category_list
df['代表性评论'] = quote_list

# 导出到 xls 文件里，save to xls file    
df.to_csv('豆瓣 top 250 电影爬虫抓取.csv')

# 分析电影来源地并画饼图
locations = []
for i in range(len(place_list)):
    nations = place_list[i].split(' ') 
    for j in range(len(nations)):
        if nations[j] == '西德':
            nations[j] = '德国'
        locations.append(nations[j])

df_location = pd.DataFrame(locations, columns = ['地区'])
# 按照出现次数排序, size() 可以查数，生成一个 Series 类型
# 然后用 reset_index 重新命名，参数为 name
# DataFrame 类型的 reset_index 参数为 names
df2 = df_location.groupby('地区').size().reset_index(name='counts') # df2 = df2['地区'].value_counts(ascending = False).reset_index()
df2.sort_values(by = 'counts', ascending = False, inplace = True, ignore_index = True)
# 画饼状图
values = []
labels = []
other_count = 0
number = 10
for i in range(number - 1):
    values.append(df2['counts'][i])
    labels.append(df2['地区'][i])
for i in range(number - 1, df2.shape[0]):
    other_count += int(df2['counts'][i])
values.append(other_count)
labels.append('其他地区')
plt.figure(1)
plt.rcParams['figure.dpi'] = 200 # 设置图像清晰度
plt.pie(values, labels=labels, autopct='%.1f%%')
plt.legend()  # 显示标签
plt.show()


# 分析电影类型并画饼图
categories = []
for i in range(len(category_list)):
    category = category_list[i].split(' ') 
    for j in range(len(category)):
        categories.append(category[j])
        
df_category = pd.DataFrame(categories, columns = ['类型'])    
df3 = df_category.groupby('类型').size().reset_index(name='counts') 
df3.sort_values(by = 'counts', ascending = False, inplace = True, ignore_index = True)

values = []
labels = []
other_count = 0
for i in range(number - 1):
    values.append(df3['counts'][i])
    labels.append(df3['类型'][i])
for i in range(number - 1, df3.shape[0]):
    other_count += int(df3['counts'][i])
values.append(other_count)
labels.append('其他类型')
plt.figure(2)
plt.pie(values, labels=labels, autopct='%.1f%%')
plt.legend()
plt.show()

# 画词云图
jieba.add_word('久石让')
jieba.add_word('谢耳朵')
# 一些语气词和没有意义的词
del_words = [ '就是', '一个', '被', '电影', '我们',
              '不是', '每个',  '不会',  '没有', 
              '这样', '那么', '不要', '如果',
              '不能',  '一种', '不过', '只有', '不得不', 
              '不得不', '一部']
all_quotes = ''.join(quote_list) # 将所有代表性评论拼接为一个文本
# 去掉标点符号
all_quotes = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ", all_quotes)
words = jieba.lcut(all_quotes)
words_final = []
for i in range(len(words)): 
    if len(words[i]) > 1 and  words[i] not in del_words: # 去掉一些语气词，单字词 
        words_final.append(words[i])
df_text = pd.DataFrame(words_final, columns = ['词语'])
df_text2 = df_text.groupby('词语').size() # 查找词频，也可以用 from collections import Counter， 然后用 Counter(words_final) 来查找词频
cloud = WordCloud(
    font_path = 'C:\Windows\Fonts\FZSTK.TTF', # 中文字体地址 C:\Windows\Fonts\FZSTK.TTF，提前下载字体或指定，否则中文无法显示
    # mask = '' 通过 mask 参数指定一个图片地址作为词云的背景图像 
    background_color = 'white',
    width = 1000,
    height = 860,
    max_words = 25   
  )
#wc = cloud.generate(words) # 这种方法对中文支持不太好，this mehtod is better for only english string
wc = cloud.generate_from_frequencies(df_text2)
wc.to_file("豆瓣 TOP 250 词云.jpg") 
plt.figure(3)
plt.imshow(wc)
plt.axis('off')
plt.title('豆瓣 TOP 250 电影代表性评论的词云分析')
plt.show()

# 评分最高的 15 部电影
# 用 seaborn 画柱状图，因为自动将不同柱子分配不同的颜色
df_star = df.sort_values(by = '评分', ascending = False, ignore_index = True)
number = 15
df_star_top = df_star.head(number)
plt.figure(4)
sns.barplot(data = df_star_top, y = '电影名字', x = '评分', orient = 'h')
plt.title('评分最高的 ' + str(number) + ' 部电影')
plt.show()

# 评分人数最多的 15 部电影
df_num = df.sort_values(by = '评价人数', ascending = False, ignore_index = True)
df_num_top = df_num.head(number)
plt.figure(5)
sns.barplot(data = df_num_top, y = '电影名字', x = '评价人数', orient = 'h')
plt.title('评分人数最多的 ' + str(number) + ' 部电影')
plt.show()

# 电影年代画图
df_year = df.groupby('上映日期').size().reset_index(name = '部数')
df_year.sort_values('部数', ascending = False, inplace = True)
df_year_top = df_year.head(number)
df_year_top.sort_values('上映日期', inplace = True)
plt.figure(6)
sns.barplot(data = df_year_top, x = '上映日期', y = '部数')
plt.title('经典电影上映最多的 ' + str(number) +' 年')
plt.show()

