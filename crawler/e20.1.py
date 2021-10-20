# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 20:54:25 2021

@author: zhen chen

MIT Licence.

Python version: 3.8


Description: 
    
"""
import requests
from bs4 import BeautifulSoup

allUniv = []
def getHTMLText(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = 'utf-8'
        return r.text
    except:
        return ""
def fillUnivList(soup):
    data = soup.find_all('tr') 
    data.pop(0) # 或 del(data[0])
    count = 0
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
def printUnivList(num):
    print("{:^4}{:^10}{:^5}{:^8}{:^10}".format("排名","学校名称","省市","总分","办学层次"))
    for i in range(num):
        u=allUniv[i]
        print("{:^4}{:^10}{:^5}{:^8}{:^10}".format(u[0],u[1],u[2],u[3],u[4]))
def main():
    url = 'https://www.shanghairanking.cn/rankings/bcur/2021'
    html = getHTMLText(url)
    soup = BeautifulSoup(html, "lxml")
    fillUnivList(soup)
    printUnivList(10)
    
main()




