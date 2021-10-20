# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:06:10 2021

@author: zhen chen

MIT Licence.

Python version: 3.8


Description: 
    
"""
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
import pandas as pd

allUniv = []
driver = webdriver.Chrome(executable_path=r'E:\爬虫练习\chromedriver.exe')
url = "https://www.shanghairanking.cn/rankings/bcur/2021"
driver.get(url)
page_count = 1
count = 0

while True:
    # Increase page_count value on each iteration on +1
    page_count += 1
    # Do what you need to do on each page
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
    data = soup.find_all('tr') 
    data.pop(0)  # 或 del(data[0])    
    for tr in data:
        ltd = tr.find_all('td')
        if len(ltd)==0:
            continue
        count = count + 1
        rank = count
        name = ltd[1].a.get_text()
        address = ltd[2].get_text().strip()
        style = ltd[3].get_text().strip()
        score = ltd[4].get_text().strip()
        level = ltd[5].get_text().strip()
        singleUniv = [str(rank), name, address, style, score, level]
        allUniv.append(singleUniv)
    try:
        # Clicking on "2" on pagination on first iteration, "3" on second...
        driver.find_element_by_link_text(str(page_count)).click()
    except NoSuchElementException:
        # Stop loop if no more page available
        break

output = pd.DataFrame(allUniv)
output.columns = ['排名', '大学', '省份', '类型', '总分', '办学层次']
output.to_excel(r'E:\爬虫练习\大学排名.xlsx', index=False)    
    

# options = Options()
# options.add_argument("start-maximized")
# options.add_argument("disable-infobars")
# options.add_argument("--disable-extensions")