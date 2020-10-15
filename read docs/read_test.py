# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:39:37 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""


import docx  # docx 文件可以直接读取，读取07版本之前的doc文件有问题，需要先转化成 docx
import win32com.client as wc # 将 doc 转化为 docx


import os

# 读取文件夹所有文件的文件名
path = "E:\爬虫练习"  #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称




#doc文件另存为docx
#word = wc.Dispatch("Word.Application")
#doc = word.Documents.Open(r"E:\\爬虫练习\\重庆市社会科学规划项目管理办法.doc")
#
##上面的地方只能使用完整绝对地址，相对地址找不到文件，且，只能用“\\”，不能用“/”，哪怕加了 r 也不行，涉及到将反斜杠看成转义字符。
#
#doc.SaveAs(r"E:\\爬虫练习\\重庆市社会科学规划项目管理办法.docx", 12)
##注意SaveAs会打开保存后的文件，有时可能看不到，但后台一定是打开的
#doc.Close
#word.Quit   

                       
                          

file = docx.Document('E:\爬虫练习\重庆市社会科学规划项目管理办法.docx') 

for p in file.paragraphs:
    print(p.text)

print("段落数:" + str(len(file.paragraphs)))