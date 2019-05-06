# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:08:23 2019

@author: zhen chen

Python version: 3.7

Description: This a practice for crawler in the movie reviewing website Maoyan
    
"""
import requests  # a package for requesting from websites
import xlwt  # a package for reading and writing in excel, not supporting xlsx writing
from bs4 import BeautifulSoup # a package for webstie data analysis


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
    link = 'https://movie.douban.com/top250?start=' + str(i*25) # 10 pages for total 250 movies
    res = requests.get(link, headers = headers, timeout = 10)
    
    # res.text is the content of the crawler
    soup = BeautifulSoup(res.text, "lxml")  # lxml is one decoding model for Beautifulsoup
    div_title_list = soup.find_all('div', class_ = 'hd') # find classes whose tag are hd
    div_info_list = soup.find_all('div', class_ = 'bd')
    div_star_list = soup.find_all('div', class_ = 'star')
    div_quote_list = soup.find_all('p', class_ = 'quote')
    
    for each in div_title_list:
        # a is href link of html
        # strip() is for stripping spacing at the beginning and end of a string
        movie = each.a.span.text.strip() # only get the first span of text this method
        movie_list_chinese_name.append(movie)
        
    # get second span by css location   
    div_title_list2 = soup.select('div.hd > a > span:nth-of-type(2)')
    for each in div_title_list2:
        movie = each.text
        #movie = movie.replace(u'\xa0', u' ')
        movie = movie.strip('\xa0/\xa0') # strip the extra string in the english name
        movie_list_english_name.append(movie)
    
    for each in div_info_list:
        num += 1
        info = each.p.text.strip()
        if len(info) < 3: # skip the information not needed
            continue
        
        # find the movie year
        lines = info.split('\n')  # split the info into two lines
        time_start = lines[1].find('20')
        if time_start < 0:
            time_start = lines[1].find('19')
        time = lines[1][time_start : time_start + 4]
        time_list.append(time)
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
        
        # find the director name
        if num == 46:
            print(info)
        end = info.find('主')
        if end < 0:
            end = info.find('...')
        director = info[4 : end - 3]
        director_list.append(director)
        
        # find the nation of the movie
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
        
        
        # find the category of the movie
        frequent = 0
        start = 0
        for j in range(len(line2)):
            if line2[j] == '\xa0':
                frequent += 1
            if frequent == 4 and start == 0:
                start = j + 1
        category = line2[start : len(line2)]
        category_list.append(category)
    
    # find the star of each movie    
    for each in div_star_list:
        info = each.text.strip()
        star = info[0 : 3]
        star_list.append(star)
        end = info.find('人')
        reviewNum = info[3 : end]
        reviewNum_list.append(reviewNum)
    
    for each in div_quote_list:
        info = each.text.strip()
        quote_list.append(info)
    
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
    
file.save('豆瓣 top 250 电影爬虫抓取.xls')